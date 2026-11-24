"""
OCR Dispatcher.
Routes OCR requests to the appropriate engine based on source language.
"""

from __future__ import annotations

import copy
import logging
import re
import numpy as np

from . import manga_ocr
from . import easy_ocr

log = logging.getLogger("ocr.dispatcher")

# Map of UI language codes to EasyOCR language codes
EASY_OCR_LANG_MAP = {
    "en": ["en"],
    "ko": ["ko", "en"],
    "zh": ["ch_sim", "en"],  # We default to simplified chinese + english
    "zh-tw": ["ch_tra", "en"],
}

# EasyOCR only allows one non-Latin language per reader, paired with English.
# Auto mode therefore uses three compatible readers and combines their output.
EASY_OCR_AUTO_LANGS = ["ja", "en"]
EASY_OCR_ENGLISH_LANGS = ["en"]
EASY_OCR_KOREAN_LANGS = ["ko", "en"]
EASY_OCR_CHINESE_LANGS = ["ch_sim", "en"]

_ENGLISH_MARKERS = {
    "a", "and", "are", "be", "but", "do", "for", "from", "have", "he",
    "her", "his", "i", "if", "in", "is", "it", "me", "my", "no", "not",
    "of", "on", "or", "she", "that", "the", "there", "they", "this", "to",
    "was", "we", "what", "when", "who", "why", "will", "with", "you", "your",
}


def _contains_han(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text or "")


def _contains_japanese_kana(text: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" for char in text or "")


def _image_has_meaningful_color(img_rgb: np.ndarray) -> bool:
    """Identify colour webtoon pages without being fooled by antialiasing.

    Korean/Chinese webtoons are commonly colour while the manga path is
    primarily monochrome.  On colour pages a Japanese OCR hallucination must
    not short-circuit the native Korean/Chinese readers.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] < 3 or img_rgb.size == 0:
        return False
    sample = img_rgb[::8, ::8, :3].astype(np.int16, copy=False)
    channel_spread = sample.max(axis=2) - sample.min(axis=2)
    return bool(np.mean(channel_spread >= 18) >= 0.02)


def _page_script_scores(candidates: dict[str, list[dict]]) -> dict[str, int]:
    """Score each OCR pass only with characters belonging to its script."""
    scores = {"en": 0, "ja": 0, "ko": 0, "zh": 0}
    for language, regions in candidates.items():
        for region in regions:
            text = region.get("text", "") or ""
            # Isolated characters are commonly OCR hallucinations from art or
            # watermarks and must not decide the language of an entire page.
            if sum(char.isalnum() for char in text) < 2:
                continue
            if language == "en":
                scores[language] += sum(
                    ("a" <= char.lower() <= "z") for char in text
                )
            elif language == "ja":
                scores[language] += 4 * sum(
                    "\u3040" <= char <= "\u30ff" for char in text
                ) + sum("\u4e00" <= char <= "\u9fff" for char in text)
            elif language == "ko":
                scores[language] += 4 * sum(
                    "\uac00" <= char <= "\ud7af" for char in text
                )
            elif language == "zh":
                scores[language] += 3 * sum(
                    "\u4e00" <= char <= "\u9fff" for char in text
                )
    return scores


def _script_evidence(candidates: dict[str, list[dict]]) -> dict[str, int]:
    """Count decisive native-script characters produced by each reader."""
    joined = {
        language: " ".join(region.get("text", "") or "" for region in regions)
        for language, regions in candidates.items()
    }
    ko_text = joined.get("ko", "")
    return {
        # Hiragana + katakana.  This must not use the CJK Extension-A start
        # (U+3400): that made every Japanese page report kana=0 after reload.
        "kana": sum("\u3040" <= char <= "\u30ff" for char in joined.get("ja", "")),
        "hangul": sum("가" <= char <= "힯" for char in ko_text),
        "han_ja": sum("一" <= char <= "鿿" for char in joined.get("ja", "")),
        "han_zh": sum("一" <= char <= "鿿" for char in joined.get("zh", "")),
        "latin": sum(char.isascii() and char.isalpha() for char in joined.get("en", "")),
        # Non-space chars emitted by the Korean reader. On real Korean pages
        # most of them are hangul; on Japanese/English pages the reader emits
        # latin/symbol garbage with only stray hangul, so hangul density
        # separates the two cases reliably.
        "ko_len": sum(1 for char in ko_text if not char.isspace()),
    }


def _choose_auto_language(
    scores: dict[str, int], evidence: dict[str, int]
) -> str:
    """Choose native scripts before Latin hallucinations from manga artwork."""
    # Kana exists only in Japanese and is stronger evidence than a larger
    # amount of fake Latin/Hangul produced from vertical manga lettering.
    if evidence["kana"]:
        return "ja"
    if evidence["hangul"] >= 2:
        return "ko"
    if evidence["han_zh"] >= 2:
        return "zh"
    if evidence["latin"] >= 2:
        return "en"
    return max(scores, key=scores.get) if any(scores.values()) else "en"


def _english_page_is_confident(regions: list[dict]) -> bool:
    """Accept English early only when OCR produced actual English words."""
    text = " ".join(region.get("text", "") or "" for region in regions)
    words = re.findall(r"[A-Za-z]+", text.lower())
    marker_count = sum(word in _ENGLISH_MARKERS for word in words)
    alnum_count = sum(char.isalnum() for char in text)
    digit_ratio = sum(char.isdigit() for char in text) / max(1, alnum_count)
    lexical_quality = easy_ocr._english_candidate_quality(text)
    substantial_regions = sum(
        sum(char.isascii() and char.isalpha() for char in region.get("text", "")) >= 4
        for region in regions
    )
    return (
        len(words) >= 3
        and marker_count >= 2
        and substantial_regions >= 2
        and digit_ratio <= 0.12
        and lexical_quality >= 3.5
    )


def _english_page_is_plausible(regions: list[dict]) -> bool:
    """Recognize short/stylized English without accepting OCR gibberish.

    Scanlation fonts often damage punctuation or one letter in a common word,
    so the strict fast-path can miss genuine English. Requiring readable Latin
    words plus either an English marker or strong lexical quality separates
    those pages from EasyOCR's digit-heavy hallucinations on Japanese art.
    """
    text = " ".join(region.get("text", "") or "" for region in regions)
    words = re.findall(r"[A-Za-z]+", text.lower())
    marker_count = sum(word in _ENGLISH_MARKERS for word in words)
    alnum_count = sum(char.isalnum() for char in text)
    digit_ratio = sum(char.isdigit() for char in text) / max(1, alnum_count)
    lexical_quality = easy_ocr._english_candidate_quality(text)
    substantial_regions = sum(
        sum(char.isascii() and char.isalpha() for char in region.get("text", "")) >= 4
        for region in regions
    )
    return (
        len(words) >= 3
        and substantial_regions >= 1
        and digit_ratio <= 0.15
        and (
            (marker_count >= 1 and lexical_quality >= 2.5)
            or (substantial_regions >= 2 and lexical_quality >= 8.0)
        )
    )


def _english_overrides_japanese(
    regions: list[dict], scores: dict[str, int], evidence: dict[str, int]
) -> bool:
    """Prefer a coherent Latin pass over MangaOCR's scanlation hallucination.

    MangaOCR often reads stylized English as full-width Latin mixed with kana.
    The normal English fast path deliberately stays strict, so compare both
    completed passes before allowing kana alone to decide the page language.
    Real Japanese pages produce a much larger native-script score; requiring
    the English score to reach 22% of it preserves those pages.
    """
    text = " ".join(region.get("text", "") or "" for region in regions)
    words = re.findall(r"[A-Za-z]+", text)
    alnum_count = sum(char.isalnum() for char in text)
    digit_ratio = sum(char.isdigit() for char in text) / max(1, alnum_count)
    substantial_regions = sum(
        sum(char.isascii() and char.isalpha() for char in region.get("text", "")) >= 4
        for region in regions
    )
    english_score = scores.get("en", 0)
    japanese_score = scores.get("ja", 0)
    return (
        evidence.get("latin", 0) >= 24
        and len(words) >= 3
        and substantial_regions >= 2
        and digit_ratio <= 0.20
        and english_score >= max(24, japanese_score * 0.22)
    )


def _japanese_page_is_confident(regions: list[dict]) -> bool:
    """Kana from manga-ocr is decisive evidence of Japanese manga text."""
    text = " ".join(region.get("text", "") or "" for region in regions)
    kana = sum("\u3040" <= char <= "\u30ff" for char in text)
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    native_regions = sum(
        _contains_japanese_kana(region.get("text", ""))
        for region in regions
    )
    return kana >= 2 and kana + han >= 4 and native_regions >= 1


def _japanese_fast_path_is_confident(regions: list[dict]) -> bool:
    """Require broad kana evidence before skipping the Korean reader.

    MangaOCR can occasionally invent one short Japanese-looking fragment on
    Korean or Chinese artwork.  A real Japanese page normally contains kana
    across several regions, so this stricter check is safe for the fast path;
    short pages still continue through the full language comparison below.
    """
    text = " ".join(region.get("text", "") or "" for region in regions)
    kana = sum("\u3040" <= char <= "\u30ff" for char in text)
    native_regions = sum(
        _contains_japanese_kana(region.get("text", ""))
        for region in regions
    )
    return kana >= 8 and (native_regions >= 2 or kana >= 16)


def _korean_page_is_confident(
    regions: list[dict], evidence: dict[str, int]
) -> bool:
    """Accept Korean when Hangul is coherent, including short balloons."""
    text = " ".join(region.get("text", "") or "" for region in regions)
    hangul = sum("\uac00" <= char <= "\ud7af" for char in text)
    content_len = sum(1 for char in text if not char.isspace())
    native_regions = sum(
        any("\uac00" <= char <= "\ud7af" for char in region.get("text", "") or "")
        for region in regions
    )
    density = hangul / max(1, content_len)
    return (
        hangul >= 4
        and native_regions >= 1
        and hangul > evidence.get("kana", 0)
        and (density >= 0.42 or hangul >= 8)
    )


def _chinese_page_is_confident(
    regions: list[dict], scores: dict[str, int], evidence: dict[str, int]
) -> bool:
    """Prefer a dense Han pass over Japanese hallucinations on colour pages."""
    text = " ".join(region.get("text", "") or "" for region in regions)
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    content_len = sum(1 for char in text if not char.isspace())
    native_regions = sum(_contains_han(region.get("text", "")) for region in regions)
    density = han / max(1, content_len)
    return (
        han >= 2
        and native_regions >= 1
        and density >= 0.55
        and scores.get("zh", 0) >= max(6, scores.get("ja", 0) * 0.8)
        and evidence.get("han_zh", 0) >= han
    )

async def load_models(device: str = "cpu", source_lang: str = "auto"):
    """
    Load only the primary OCR engine. Other language readers are loaded lazily
    when the staged detector actually needs them, reducing idle RAM pressure.
    """
    source_lang = (source_lang or "auto").lower().strip()
    if source_lang in ("ja", "jpn", "japanese"):
        await manga_ocr.load_model(device)
    elif source_lang == "auto":
        # Both readers are shared by the whole process. Loading manga-ocr only
        # after the first Japanese page made that request wait another 20-30
        # seconds and looked like llama.cpp was stuck thinking.
        await easy_ocr.load_model(device, EASY_OCR_ENGLISH_LANGS)
        await manga_ocr.load_model(device)
    elif source_lang in EASY_OCR_LANG_MAP:
        await easy_ocr.load_model(device, EASY_OCR_LANG_MAP[source_lang])
    else:
        await easy_ocr.load_model(device, ["en"])


async def recognize_regions(
    img_rgb: np.ndarray,
    textlines: list[dict],
    source_lang: str = "auto",
    device: str = "cpu",
) -> list[dict]:
    """
    Route to manga_ocr or easyocr based on language.
    """
    source_lang = (source_lang or "auto").lower().strip()

    # manga-ocr is the best choice only when Japanese is explicitly selected.
    if source_lang in ["ja", "jpn", "japanese"]:
        log.info(f"Routing OCR to manga-ocr (lang: {source_lang})")
        await manga_ocr.load_model(device)
        return await manga_ocr.recognize(img_rgb, textlines, device)

    if source_lang == "auto":
        log.info("Routing OCR to staged automatic page-language selection")
        candidates: dict[str, list[dict]] = {}
        color_page = _image_has_meaningful_color(img_rgb)

        def _store(key: str, result, regions):
            if isinstance(result, Exception):
                log.warning("Auto recognition failed for %s: %s", key, result)
                for region in regions:
                    region["text"] = ""
                candidates[key] = regions
            else:
                candidates[key] = result

        # Staged route:
        #   1. English alone on GPU. Real English pages finish here. Always run
        #      this first: caching Japanese across pages breaks when the reader
        #      opens an English chapter in the same browser session.
        #   2. Japanese on the configured device if English was inconclusive. Real Japanese
        #      pages finish here and never start another heavyweight reader.
        #   3. Korean, then Chinese, only when the preceding script evidence is
        #      inconclusive. Keeping CPU readers sequential prevents the host
        #      from freezing under competing Torch/EasyOCR thread pools.
        detected_language = None
        evidence = {}

        if detected_language is None and not color_page:
            english_regions = copy.deepcopy(textlines)
            english_result = await easy_ocr.recognize(
                img_rgb, english_regions, device, EASY_OCR_ENGLISH_LANGS
            )
            _store("en", english_result, english_regions)
            evidence = _script_evidence(candidates)
            if _english_page_is_confident(candidates["en"]):
                detected_language = "en"
                log.info("Automatic OCR selected English (fast path)")
            elif (
                not color_page
                and _english_page_is_plausible(candidates["en"])
            ):
                detected_language = "en"
                log.info("Automatic OCR selected English (plausible latin path)")

        # Colour webtoons are overwhelmingly Korean/Chinese. Keep an empty
        # Japanese candidate for script comparisons and defer MangaOCR until
        # both native readers fail. This avoids false kana short-circuits and
        # a needless GPU inference beside a preloaded local translator.
        if detected_language is None and color_page:
            japanese_regions = copy.deepcopy(textlines)
            for region in japanese_regions:
                region["text"] = ""
            candidates["ja"] = japanese_regions

        if detected_language is None and "ja" not in candidates:
            japanese_regions = copy.deepcopy(textlines)
            try:
                # Auto mode loads only English at startup. MangaOCR must be
                # loaded lazily before its first recognition request.
                await manga_ocr.load_model(device)
                japanese_result = await manga_ocr.recognize(
                    img_rgb, japanese_regions, device
                )
            except Exception as exc:
                japanese_result = exc
            _store("ja", japanese_result, japanese_regions)

            scores = _page_script_scores(candidates)
            evidence = _script_evidence(candidates)
            if (
                not color_page
                and _english_overrides_japanese(candidates["en"], scores, evidence)
            ):
                detected_language = "en"
                log.info(
                    "Automatic OCR selected English (comparative latin path)"
                )
            elif (
                not color_page
                and _japanese_fast_path_is_confident(candidates["ja"])
            ):
                detected_language = "ja"
                log.info("Automatic OCR selected Japanese (fast kana path)")

        if detected_language is None:
            korean_regions = copy.deepcopy(textlines)
            try:
                korean_result = await easy_ocr.recognize(
                    img_rgb, korean_regions, "cpu", EASY_OCR_KOREAN_LANGS
                )
            except Exception as exc:
                korean_result = exc
            _store("ko", korean_result, korean_regions)
            evidence = _script_evidence(candidates)
            if _korean_page_is_confident(candidates["ko"], evidence):
                detected_language = "ko"
                log.info("Automatic OCR selected Korean (hangul evidence)")
            elif (
                not color_page
                and _japanese_page_is_confident(candidates["ja"])
                and evidence["kana"] >= max(2, evidence["hangul"])
            ):
                detected_language = "ja"
                log.info("Automatic OCR selected Japanese (kana evidence)")
            elif (
                not color_page
                and _english_page_is_confident(candidates["en"])
            ):
                detected_language = "en"
                log.info("Automatic OCR selected English (latin evidence)")
            else:
                chinese_regions = copy.deepcopy(textlines)
                _store(
                    "zh",
                    await easy_ocr.recognize(
                        img_rgb, chinese_regions, "cpu", EASY_OCR_CHINESE_LANGS
                    ),
                    chinese_regions,
                )
                scores = _page_script_scores(candidates)
                evidence = _script_evidence(candidates)
                if _chinese_page_is_confident(
                    candidates["zh"], scores, evidence
                ):
                    detected_language = "zh"
                    log.info("Automatic OCR selected Chinese (dense han evidence)")
                elif color_page:
                    # A colour page can still contain English or Japanese.
                    # Only pay either GPU OCR cost after Korean and Chinese
                    # have supplied no native evidence.
                    english_regions = copy.deepcopy(textlines)
                    try:
                        english_result = await easy_ocr.recognize(
                            img_rgb,
                            english_regions,
                            device,
                            EASY_OCR_ENGLISH_LANGS,
                        )
                    except Exception as exc:
                        english_result = exc
                    _store("en", english_result, english_regions)
                    if _english_page_is_plausible(candidates["en"]):
                        detected_language = "en"
                        log.info(
                            "Automatic OCR selected English "
                            "(colour-page fallback)"
                        )

                if detected_language is None and color_page:
                    japanese_regions = copy.deepcopy(textlines)
                    try:
                        await manga_ocr.load_model(device)
                        japanese_result = await manga_ocr.recognize(
                            img_rgb, japanese_regions, device
                        )
                    except Exception as exc:
                        japanese_result = exc
                    _store("ja", japanese_result, japanese_regions)
                    scores = _page_script_scores(candidates)
                    evidence = _script_evidence(candidates)

                if detected_language is None and _japanese_page_is_confident(
                    candidates["ja"]
                ):
                    detected_language = "ja"
                    log.info("Automatic OCR selected Japanese (fallback kana)")
                elif (
                    detected_language is None
                    and _english_page_is_plausible(candidates["en"])
                ):
                    detected_language = "en"
                    log.info("Automatic OCR selected English (deferred latin path)")
                elif detected_language is None:
                    detected_language = _choose_auto_language(scores, evidence)
                    log.info(
                        "Automatic OCR fell back to scoring (%s)", detected_language
                    )

        scores = _page_script_scores(candidates)
        log.info(
            "Automatic OCR language=%s, scores=%s, evidence=%s",
            detected_language,
            scores,
            evidence,
        )

        chosen = candidates[detected_language]
        original_count = len(textlines)
        for original, selected in zip(textlines, chosen[:original_count]):
            original["text"] = selected.get("text", "")
            original["ocr_confidence"] = selected.get("ocr_confidence", 0.0)
            original["detected_source_lang"] = detected_language
            if selected.get("ocr_polygons"):
                original["ocr_polygons"] = copy.deepcopy(selected["ocr_polygons"])
        # English page OCR can recover reliable lines outside CTD regions.
        # Preserve those synthetic regions after automatic language choice.
        for selected in chosen[original_count:]:
            recovered = copy.deepcopy(selected)
            recovered["detected_source_lang"] = detected_language
            textlines.append(recovered)
        log.info(
            "Automatic OCR sample (%s): %s",
            detected_language,
            [
                region.get("text", "")[:80]
                for region in textlines
                if region.get("text", "").strip()
            ][:5],
        )
        return textlines
        
    # If other supported language, use EasyOCR
    if source_lang in EASY_OCR_LANG_MAP:
        langs = EASY_OCR_LANG_MAP[source_lang]
        log.info(f"Routing OCR to easyocr (lang: {source_lang} -> {langs})")
        await easy_ocr.load_model(device, langs)
        return await easy_ocr.recognize(img_rgb, textlines, device, langs)
        
    log.warning(f"Unsupported OCR lang '{source_lang}'. Falling back to EasyOCR English.")
    await easy_ocr.load_model(device, ["en"])
    return await easy_ocr.recognize(img_rgb, textlines, device, ["en"])


def unload_all():
    """Unload all OCR models."""
    manga_ocr.unload()
    easy_ocr.unload()
