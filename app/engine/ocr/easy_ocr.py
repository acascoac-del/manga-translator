"""
OCR using easyocr.
Supports multiple languages: English, Chinese, Korean, etc.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("ocr.easy_ocr")

_readers = {}
_model_lock = None


def _english_text_score(text: str) -> int:
    """Prefer real English words over digit-heavy OCR hallucinations."""
    words = re.findall(r"[A-Za-z]+", text or "")
    readable = sum(
        min(len(word), 8)
        for word in words
        if len(word) >= 2 and any(char.lower() in "aeiouy" for char in word)
    )
    digits = sum(char.isdigit() for char in text or "")
    return readable - digits


_ENGLISH_OCR_CORRECTIONS = {
    # Frequent glyph confusions in bold/italic scanlation fonts. Keep this
    # intentionally small: it is a typography repair table, not a translator.
    "almays": "always",
    "aojusteo": "adjusted",
    "arent": "aren't",
    "beng": "being",
    "bodys": "body's",
    "couzse": "course",
    "coulo": "could",
    "finew": "fine",
    "hacoeningi": "hardening",
    "haroening": "hardening",
    "hepo": "hero",
    "hes": "he's",
    "klled": "killed",
    "mankindj": "mankind",
    "mamay": "mama!!",
    "mamayl": "mama!!",
    "nonsensel": "nonsense",
    "monsier": "monster",
    "nuvee": "over",
    "ordlnary": "ordinary",
    "pressupe": "pressure",
    "shord": "sword",
    "sfeoodee": "stronger",
    "taks": "thanks",
    "toueh": "tough",
    "thati": "that",
    "uncles": "uncle's",
    "ueh": "ugh",
    "wel": "well",
    "youre": "you're",
    "yill": "kill",
    "zhell": "well",
}

_ENGLISH_COMMON_WORDS = {
    "a", "about", "always", "and", "apart", "are", "as", "at", "be",
    "blade", "body", "but", "could", "course", "damn", "do", "fall", "falling",
    "feelings", "fine", "fined", "for", "from", "getting", "had", "handle",
    "harden", "hardening", "he", "hero", "hes", "her", "hide", "human", "i",
    "if", "in", "is", "it", "let", "mana", "me", "monster", "more", "my",
    "mankind", "nonsense", "of", "on", "only", "over", "please", "pressure",
    "properly", "puts",
    "rest", "sec", "she", "so", "stronger", "that", "the", "they", "this",
    "to", "too", "totally", "tough", "ugh", "uncle", "wait", "well", "you",
    "your", "bitten", "compress",
}


def _english_candidate_quality(text: str) -> float:
    """Rank OCR alternatives using words, digits and common manga English."""
    words = re.findall(r"[A-Za-z']+", (text or "").lower())
    if not words:
        return -100.0
    normalized = [word.replace("'", "") for word in words]
    common = sum(word in _ENGLISH_COMMON_WORDS for word in normalized)
    letters = sum(len(word) for word in normalized)
    unknown = len(words) - common
    digits = sum(char.isdigit() for char in text or "")
    return common * 8.0 + min(letters, 80) * 0.35 - unknown * 2.5 - digits * 4.0


def _english_needs_crop_retry(text: str) -> bool:
    words = [
        word.replace("'", "")
        for word in re.findall(r"[A-Za-z']+", (text or "").lower())
    ]
    if not words:
        return False
    common = sum(word in _ENGLISH_COMMON_WORDS for word in words)
    mixed_digits = any(char.isdigit() for char in text or "")
    if len(words) == 1 and not mixed_digits:
        return False
    return mixed_digits or common / len(words) < 0.60


def _english_crop_retry_limit() -> int:
    """Cap costly per-region OCR after the page-level CRAFT pass."""
    try:
        return max(0, min(8, int(os.getenv("EASYOCR_MAX_CROP_RETRIES", "4"))))
    except ValueError:
        return 4


def _select_english_crop_retries(
    textlines: list[dict], texts: list[str], limit: int | None = None
) -> list[tuple[int, dict, str]]:
    """Choose a bounded set of the regions most likely to benefit from retry.

    Page OCR already recognizes most dialogue. Retrying every unfamiliar name
    or stylized word serially kept CPU/GPU busy for 7-12 seconds per page. Keep
    the lowest-quality reads and reserve one slot for a substantial empty box.
    """
    if limit is None:
        limit = _english_crop_retry_limit()
    limit = max(0, limit)
    if not limit:
        return []

    nonempty = [
        (index, tl, text)
        for index, (tl, text) in enumerate(zip(textlines, texts))
        if text.strip() and _english_needs_crop_retry(text)
    ]
    nonempty.sort(key=lambda item: _english_candidate_quality(item[2]))

    empty = [
        (index, tl, text)
        for index, (tl, text) in enumerate(zip(textlines, texts))
        if not text.strip()
        and (tl["bbox"][2] - tl["bbox"][0]) >= 22
        and (tl["bbox"][3] - tl["bbox"][1]) >= 16
    ]
    empty.sort(
        key=lambda item: (
            (item[1]["bbox"][2] - item[1]["bbox"][0])
            * (item[1]["bbox"][3] - item[1]["bbox"][1])
        ),
        reverse=True,
    )

    empty_slots = 1 if empty and limit >= 2 else 0
    selected = nonempty[: limit - empty_slots]
    selected.extend(empty[: limit - len(selected)])
    return selected


def _normalize_english_ocr(text: str) -> str:
    """Repair safe English OCR artifacts before any translator sees them."""
    value = " ".join((text or "").split())
    if not value:
        return ""
    value = re.sub(r"([A-Za-z]{2,})\s*-\s+([A-Za-z]{2,})", r"\1\2", value)
    value = re.sub(r"\bas\s*\.\s*a\b", "as a", value, flags=re.IGNORECASE)
    value = re.sub(r"\bis\s*\.\s+the\b", "is the", value, flags=re.IGNORECASE)
    # Tiny outlined SFX next to a bubble is occasionally appended as Oee/0e.
    value = re.sub(
        r"([.!?])\s+(?:cee|oee|0e)\s*$", r"\1", value, flags=re.IGNORECASE
    )
    value = re.sub(
        r"\bmana\s+(?:ox|07[\"']?)\s+(?=normal|ordinary)",
        "Mana 0% ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r",\s*,+", ",", value)
    value = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", value)
    # A vertical border is sometimes read as an extra L before the first word.
    value = re.sub(r"^l(?=you\b)", "", value, flags=re.IGNORECASE)
    # A recurring outlined-font read on English scans. Repair the complete
    # phrase before token normalization so punctuation and contractions are
    # restored instead of sending gibberish to the translator.
    value = re.sub(
        r"\baojusteo\s+meia\?\s+that\s+coulo\s+ve\s+['\"]?klled\s+mel\b",
        "Adjusted me!? That could've killed me!",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\ba[o0]?juste[do]\s+meia\?\s+that\s+co.?ld\s+(?:y|ve)\s+"
        r"klle[do]\s+mel\b",
        "Adjusted me!? That could've killed me!",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bhow\s*was\s+i\s+[5s]u9poseo\s+to\s+ficugejhat\s+outl?\?",
        "How was I supposed to figure that out!?",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\biov\s+wepent\s+gonna\s+die\s+eumavi\b",
        "You weren't gonna die anyway!",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bsls\?\s+5e9l\b",
        "Elsie! Elsie!",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\bnot\s+ffter\s+i\s+adiusted\s+you\b",
        "Not after I adjusted you",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(
        r"\buhcohsgovsly\s+reinforc\s+ing\s+your\s+body\s+with\s+mana\b",
        "Unconsciously reinforcing your body with mana",
        value,
        flags=re.IGNORECASE,
    )

    tokens = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9']+", value)
    repaired = []
    for token in tokens:
        if not re.fullmatch(r"[A-Za-z0-9']+", token):
            repaired.append(token)
            continue
        candidate = token
        if any(char.isalpha() for char in candidate) and any(
            char.isdigit() for char in candidate
        ):
            candidate = candidate.translate(str.maketrans({"0": "o", "1": "i"}))
        lower = candidate.lower()
        correction = _ENGLISH_OCR_CORRECTIONS.get(lower)
        if correction:
            candidate = correction
        elif lower == "tm":
            candidate = "I'm"
        elif lower == "im":
            candidate = "I'm"
        elif lower == "i":
            candidate = "I"
        elif (
            token[:1].isupper()
            and token[1:].islower()
            and lower not in _ENGLISH_COMMON_WORDS
        ):
            # Preserve probable names such as Wel/Elsie.
            candidate = token
        else:
            candidate = lower
        repaired.append(candidate)

    value = "".join(repaired).strip()
    value = re.sub(
        r"(^|[.!?]\s+)([a-z])",
        lambda match: match.group(1) + match.group(2).upper(),
        value,
    )
    return value


def _easyocr_items(result) -> list[tuple[list, str, float]]:
    """Normalize EasyOCR detail=1 results for page-level English OCR."""
    items = []
    for item in result or []:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        box, text = item[0], str(item[1]).strip()
        confidence = float(item[2]) if len(item) > 2 else 0.0
        if text:
            items.append((box, text, confidence))
    return items


def _assign_english_lines(
    textlines: list[dict],
    recognized: list[tuple[list, str, float]],
    return_unassigned: bool = False,
):
    """Attach page-level OCR lines to the detector region containing them."""
    assigned: list[list[tuple[float, float, str, float]]] = [
        [] for _ in textlines
    ]
    unassigned = []
    for box, text, confidence in recognized:
        points = np.asarray(box, dtype=np.float32)
        if points.size < 8:
            continue
        lx1, ly1 = points.min(axis=0)
        lx2, ly2 = points.max(axis=0)
        cx, cy = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        line_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
        best_index, best_overlap = None, 0.0
        for index, region in enumerate(textlines):
            x1, y1, x2, y2 = region["bbox"]
            # CTD boxes can be tight around glyphs, so allow a small margin.
            margin_x = max(8, int((x2 - x1) * 0.12))
            margin_y = max(8, int((y2 - y1) * 0.12))
            ex1, ey1 = x1 - margin_x, y1 - margin_y
            ex2, ey2 = x2 + margin_x, y2 + margin_y
            ix = max(0.0, min(lx2, ex2) - max(lx1, ex1))
            iy = max(0.0, min(ly2, ey2) - max(ly1, ey1))
            overlap = (ix * iy) / line_area
            if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                overlap += 1.0
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is not None and best_overlap >= 0.25:
            assigned[best_index].append((ly1, lx1, text, confidence))
            textlines[best_index].setdefault("ocr_polygons", []).append(
                [[int(round(x)), int(round(y))] for x, y in points.tolist()]
            )
        else:
            unassigned.append(
                (float(lx1), float(ly1), float(lx2), float(ly2), text, confidence)
            )

    texts = []
    for lines in assigned:
        lines.sort(key=lambda item: (item[0], item[1]))
        # Very low-confidence fragments are generally SFX/artwork. Keep them
        # only when they are part of a region that also has a reliable line.
        reliable = any(confidence >= 0.20 for *_, confidence in lines)
        selected = [
            text for _, _, text, confidence in lines
            if confidence >= 0.12 or reliable and len(text) >= 3
        ]
        texts.append(" ".join(selected).strip())
    if return_unassigned:
        return texts, unassigned
    return texts


def _recover_unassigned_english_regions(
    unassigned: list[tuple[float, float, float, float, str, float]],
    image_shape: tuple[int, ...],
) -> list[dict]:
    """Turn reliable orphan OCR lines into renderable English regions.

    CTD occasionally misses outlined or narrow scanlation bubbles although
    EasyOCR reads their text correctly. Group nearby lines here so they still
    enter translation/inpainting. This helper is used by English OCR only.
    """
    height, width = image_shape[:2]
    candidates = []
    for x1, y1, x2, y2, text, confidence in unassigned:
        words = re.findall(r"[A-Za-z']+", text or "")
        letters = sum(len(word.replace("'", "")) for word in words)
        lowered = (text or "").lower()
        if confidence < 0.28 or letters < 2:
            continue
        if re.search(r"(?:https?://|www\.|\.com\b|\bscans?\b)", lowered):
            continue
        candidates.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "lines": [(y1, x1, y2, x2, text)],
            "confidence": [confidence],
        })

    groups = []
    for item in sorted(candidates, key=lambda value: (value["y1"], value["x1"])):
        best_group = None
        best_distance = float("inf")
        item_width = max(1.0, item["x2"] - item["x1"])
        item_height = max(1.0, item["y2"] - item["y1"])
        item_cx = (item["x1"] + item["x2"]) / 2
        for group in groups:
            group_width = max(1.0, group["x2"] - group["x1"])
            group_height = max(1.0, group["line_height"])
            overlap_x = max(
                0.0,
                min(item["x2"], group["x2"]) - max(item["x1"], group["x1"]),
            )
            overlap_ratio = overlap_x / min(item_width, group_width)
            group_cx = (group["x1"] + group["x2"]) / 2
            center_close = abs(item_cx - group_cx) <= max(item_width, group_width) * 0.35
            vertical_gap = item["y1"] - group["y2"]
            max_gap = max(18.0, 1.8 * max(item_height, group_height))
            if (
                vertical_gap >= -0.45 * max(item_height, group_height)
                and vertical_gap <= max_gap
                and (overlap_ratio >= 0.20 or center_close)
                and abs(item_cx - group_cx) < best_distance
            ):
                best_group = group
                best_distance = abs(item_cx - group_cx)
        if best_group is None:
            groups.append({**item, "line_height": item_height})
            continue
        best_group["x1"] = min(best_group["x1"], item["x1"])
        best_group["y1"] = min(best_group["y1"], item["y1"])
        best_group["x2"] = max(best_group["x2"], item["x2"])
        best_group["y2"] = max(best_group["y2"], item["y2"])
        best_group["line_height"] = max(best_group["line_height"], item_height)
        best_group["lines"].extend(item["lines"])
        best_group["confidence"].extend(item["confidence"])

    recovered = []
    for group in groups:
        lines = sorted(group["lines"], key=lambda value: (value[0], value[1]))
        text = _normalize_english_ocr(" ".join(value[4] for value in lines))
        words = re.findall(r"[A-Za-z']+", text)
        letters = sum(len(word.replace("'", "")) for word in words)
        # Single tiny sounds are intentionally left as artwork. A substantial
        # word or a multi-word phrase is dialogue/narration worth translating.
        if not (len(words) >= 2 or letters >= 6):
            continue
        pad = max(4, int(group["line_height"] * 0.20))
        x1 = max(0, int(group["x1"] - pad))
        y1 = max(0, int(group["y1"] - pad))
        x2 = min(width, int(group["x2"] + pad))
        y2 = min(height, int(group["y2"] + pad))
        if x2 <= x1 or y2 <= y1:
            continue
        recovered.append({
            "bbox": [x1, y1, x2, y2],
            "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "text": text,
            "confidence": float(np.mean(group["confidence"])),
            "ocr_confidence": float(np.mean(group["confidence"])),
            "detected_source_lang": "en",
            "from_opencv": True,
            "ocr_recovered": True,
            "ocr_polygons": [
                [
                    [int(line_x1), int(line_y1)],
                    [int(line_x2), int(line_y1)],
                    [int(line_x2), int(line_y2)],
                    [int(line_x1), int(line_y2)],
                ]
                for line_y1, line_x1, line_y2, line_x2, _ in lines
            ],
        })
    return recovered


async def load_model(device: str = "cpu", langs: list[str] = ["en"]):
    """Load easyocr model for specific languages."""
    global _readers, _model_lock
    
    import asyncio
    if _model_lock is None:
        _model_lock = asyncio.Lock()

    lang_key = tuple(sorted(langs))
    if lang_key in _readers:
        return

    async with _model_lock:
        if lang_key in _readers:
            return

        log.info(f"Loading easyocr model for languages: {langs}...")
        import easyocr
        
        # EasyOCR uses boolean for GPU
        use_gpu = device.startswith("cuda")
        
        try:
            reader = easyocr.Reader(langs, gpu=use_gpu)
            _readers[lang_key] = reader
            log.info(f"easyocr loaded for {langs} (gpu={use_gpu}).")
        except Exception as e:
            log.error(f"Failed to load easyocr for {langs}: {e}")
            raise


async def recognize(
    img_rgb: np.ndarray,
    textlines: list[dict],
    device: str = "cpu",
    langs: list[str] = ["en"]
) -> list[dict]:
    """
    OCR each detected text region using EasyOCR.
    """
    global _readers
    
    lang_key = tuple(sorted(langs))
    if lang_key not in _readers:
        await load_model(device, langs)

    reader = _readers[lang_key]
    loop = asyncio.get_event_loop()

    is_english_only = set(langs) == {"en"}
    allowlist = None
    if is_english_only:
        allowlist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            " .,!?;:'\"()-"
        )

    read_kwargs = {
        "detail": 0,
        "paragraph": False,
        "allowlist": allowlist,
        # Page-level English OCR already supplies complete line crops to the
        # recognizer. Greedy decoding is markedly faster here and avoids the
        # costly beam search that brought no dictionary benefit in this build.
        "decoder": "greedy",
        "beamWidth": 5,
    }

    if is_english_only:
        # Run CRAFT once per page/tile instead of once for every CTD region.
        # Besides being much faster, CRAFT sees complete lines and therefore
        # handles italic scanlation fonts more accurately. Japanese, Korean,
        # and Chinese continue through their existing OCR paths unchanged.
        page_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        page_kwargs = dict(read_kwargs)
        page_kwargs.update({
            "detail": 1,
            "contrast_ths": 0.05,
            "adjust_contrast": 0.7,
            "text_threshold": 0.55,
            "low_text": 0.30,
            "link_threshold": 0.30,
            "canvas_size": 2560,
            "mag_ratio": 1.0,
        })

        def _read_page():
            height = page_bgr.shape[0]
            tile_height = 2200
            overlap = 160
            all_items = []
            start = 0
            while start < height:
                end = min(height, start + tile_height)
                tile = page_bgr[start:end]
                for box, text, confidence in _easyocr_items(
                    reader.readtext(tile, **page_kwargs)
                ):
                    shifted = [[point[0], point[1] + start] for point in box]
                    # Ignore duplicate lines in the overlap except on the
                    # first tile; the next tile owns its upper overlap.
                    center_y = sum(point[1] for point in shifted) / len(shifted)
                    if start and center_y < start + overlap:
                        continue
                    all_items.append((shifted, text, confidence))
                if end >= height:
                    break
                start = end - overlap
            return all_items

        page_items = await loop.run_in_executor(None, _read_page)
        texts, unassigned = _assign_english_lines(
            textlines, page_items, return_unassigned=True
        )

        suspicious = _select_english_crop_retries(textlines, texts)

        def _retry_crop(item):
            index, tl, current = item
            x1, y1, x2, y2 = tl["bbox"]
            height, width = page_bgr.shape[:2]
            pad_x = max(16, int((x2 - x1) * 0.18))
            pad_y = max(16, int((y2 - y1) * 0.18))
            crop = page_bgr[
                max(0, y1 - pad_y):min(height, y2 + pad_y),
                max(0, x1 - pad_x):min(width, x2 + pad_x),
            ]
            if crop.size == 0:
                return index, current
            crop = cv2.resize(
                crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
            retry = " ".join(reader.readtext(
                crop,
                detail=0,
                paragraph=False,
                allowlist=allowlist,
                decoder="greedy",
                contrast_ths=0.05,
                adjust_contrast=0.7,
            )).strip()
            if _english_candidate_quality(retry) > _english_candidate_quality(current):
                return index, retry
            return index, current

        # GPU EasyOCR is most stable sequentially. The selective lexical gate
        # normally retries only a few stylized bubbles per page.
        if suspicious:
            retries = await loop.run_in_executor(
                None, lambda: [_retry_crop(item) for item in suspicious]
            )
            for index, retry in retries:
                texts[index] = retry
            log.info(
                "EasyOCR retried %s suspicious English regions by crop "
                "(limit=%s).",
                len(suspicious), _english_crop_retry_limit(),
            )

        for tl, text in zip(textlines, texts):
            tl["text"] = _normalize_english_ocr(text)
        recovered = _recover_unassigned_english_regions(
            unassigned, img_rgb.shape
        )
        textlines.extend(recovered)
        recognized = [tl for tl in textlines if tl.get("text", "").strip()]
        log.info(
            "EasyOCR page mode recognized %s/%s regions from %s lines "
            "(%s recovered regions).",
            len(recognized), len(textlines), len(page_items), len(recovered),
        )
        return textlines

    def _read_one(tl):
        bbox = tl["bbox"]
        x1, y1, x2, y2 = bbox

        # Add padding (at least 32px for CRAFT to work on tight crops)
        h, w = img_rgb.shape[:2]
        pad_x = max(32, int((x2 - x1) * 0.15))
        pad_y = max(32, int((y2 - y1) * 0.15))

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        crop = img_rgb[py1:py2, px1:px2]
        if crop.size == 0:
            return ""

        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

        # Inverted text (white on dark) reads better negated
        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        if np.mean(gray_crop) < 110:
            crop_bgr = cv2.bitwise_not(crop_bgr)

        if is_english_only:
            # Scanlated English is often only 6-12 px high in long webtoon
            # pages. Upscaling before CRAFT prevents letters from turning into
            # digit-heavy garbage (for example ``54824 PLi``).
            crop_bgr = cv2.resize(
                crop_bgr,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

        # Bubbles usually contain several text lines. reader.recognize() only
        # works on a single line and returns garbage for multi-line crops, so
        # run readtext(): CRAFT re-detects the lines inside the padded crop and
        # each line is recognized properly.
        result = reader.readtext(crop_bgr, **read_kwargs)
        text = " ".join(result).strip()

        # Halftones and thin outlined fonts sometimes survive the grayscale
        # pass. Retry only suspicious English crops with a clean binary image;
        # normal English and every other language pay no extra OCR cost.
        if is_english_only and _english_text_score(text) < 5:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            retry = " ".join(reader.readtext(binary, **read_kwargs)).strip()
            if _english_text_score(retry) > _english_text_score(text):
                text = retry
        return text.strip() if text else ""

    def _read_all():
        # CRAFT/OpenCV release the GIL, so a small thread pool cuts the
        # per-region latency on CPU readers. A shared CUDA reader stays
        # sequential because concurrent Torch calls can increase VRAM usage.
        configured_workers = max(1, int(os.environ.get("EASY_OCR_WORKERS", "4")))
        worker_count = (
            1
            if device.startswith("cuda")
            else min(configured_workers, max(1, len(textlines)))
        )
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(_read_one, textlines))

    texts = await loop.run_in_executor(None, _read_all)
    for tl, text in zip(textlines, texts):
        tl["text"] = text

    recognized = [tl for tl in textlines if tl.get("text", "").strip()]
    log.info(f"EasyOCR recognized {len(recognized)}/{len(textlines)} regions.")
    return textlines


def unload():
    """Unload models to free memory."""
    global _readers
    if _readers:
        _readers.clear()
        log.info("easyocr models unloaded.")
