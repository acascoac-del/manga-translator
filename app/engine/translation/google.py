"""
Translation using Google Translate (free, online).
No API key required.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import logging
import time

import httpx
from deep_translator import GoogleTranslator

log = logging.getLogger("translation.google")

_DIRECT_ENDPOINTS = (
    "https://clients5.google.com/translate_a/t",
    "https://clients1.google.com/translate_a/t",
)

_blocked_until = 0.0
_direct_blocked_until = 0.0
_translation_cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
_CACHE_LIMIT = 2048

_LANGUAGE_MAP = {
    "auto": "auto",
    "jpn_jpan": "ja", "jpn": "ja", "jp": "ja", "ja": "ja",
    "eng_latn": "en", "eng": "en", "en": "en",
    "kor_hang": "ko", "kor": "ko", "kr": "ko", "ko": "ko",
    "zho_hans": "zh-CN", "zho_hant": "zh-TW", "zho": "zh-CN",
    "zh": "zh-CN", "zh-cn": "zh-CN", "zh-tw": "zh-TW",
    "spa_latn": "es", "spa": "es", "es": "es",
}


def _language_code(value: str, *, allow_auto: bool = False) -> str:
    code = (value or "").strip()
    normalized = _LANGUAGE_MAP.get(code.lower(), code)
    if normalized == "auto" and not allow_auto:
        return "en"
    return normalized or ("auto" if allow_auto else "es")


def _new_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(4.0, connect=1.5),
        limits=httpx.Limits(max_connections=8, max_keepalive_connections=8),
    )


def _extract_direct_translation(data) -> str:
    # clients*.google.com/translate_a/t returns ["translation"]. Keep the
    # nested parser too in case Google serves the legacy response shape.
    if isinstance(data, list) and data:
        if isinstance(data[0], str):
            return data[0].strip()
        if isinstance(data[0], list):
            if data[0] and isinstance(data[0][0], str):
                return data[0][0].strip()
            return "".join(
                item[0]
                for item in data[0]
                if isinstance(item, list) and item and isinstance(item[0], str)
            ).strip()
    return ""


def _usable_translation(value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    lowered = value.strip().lower()
    return not (
        lowered.startswith("error ")
        or "server error" in lowered
        or "that’s an error" in lowered
        or "that's an error" in lowered
        or "<html" in lowered
    )


async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
    _client: httpx.AsyncClient | None = None,
    _skip_direct: bool = False,
) -> str:
    """Translate a single text using Google Translate."""
    if not text or not text.strip():
        return ""

    source_lang = _language_code(source_lang, allow_auto=True)
    target_lang = _language_code(target_lang)
    cache_key = (source_lang, target_lang, text)
    cached = _translation_cache.get(cache_key)
    if cached is not None:
        _translation_cache.move_to_end(cache_key)
        return cached

    global _blocked_until, _direct_blocked_until
    if time.monotonic() < _blocked_until:
        return text

    if not _skip_direct and time.monotonic() >= _direct_blocked_until:
        owns_client = _client is None
        client = _client or _new_client()
        errors: list[str] = []
        try:
            for url in _DIRECT_ENDPOINTS:
                try:
                    resp = await client.get(
                        url,
                        params={
                            "client": "dict-chrome-ex",
                            "sl": source_lang,
                            "tl": target_lang,
                            "q": text,
                        },
                    )
                    resp.raise_for_status()
                    translated = _extract_direct_translation(resp.json())
                    if _usable_translation(translated):
                        _translation_cache[cache_key] = translated
                        _translation_cache.move_to_end(cache_key)
                        while len(_translation_cache) > _CACHE_LIMIT:
                            _translation_cache.popitem(last=False)
                        return translated
                except Exception as e:
                    errors.append(f"{url}: {e}")
        finally:
            if owns_client:
                await client.aclose()

        _direct_blocked_until = time.monotonic() + 30.0
        log.warning(
            "Direct Google endpoints failed; using secondary client for 30s: %s",
            "; ".join(errors) or "empty response",
        )

    # Docker/network filters can block translate.googleapis.com while the
    # regular Google web endpoint remains reachable. Keep this fallback here
    # so every engine's shared quality repair uses the same working route.
    def _secondary_translate():
        return GoogleTranslator(
            source=source_lang or "auto", target=target_lang
        ).translate(text)

    if time.monotonic() < _blocked_until:
        return text
    try:
        translated = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(None, _secondary_translate),
            timeout=4.0,
        )
        if _usable_translation(translated):
            _translation_cache[cache_key] = translated
            _translation_cache.move_to_end(cache_key)
            while len(_translation_cache) > _CACHE_LIMIT:
                _translation_cache.popitem(last=False)
            return translated
        return text
    except Exception as e:
        _blocked_until = time.monotonic() + 90.0
        log.error(
            "Both Google translation routes failed; disabling retries for 90s: %s",
            e,
        )
        return text


async def translate_batch(
    texts: list[str],
    source_lang: str = "auto",
    target_lang: str = "es",
) -> list[str]:
    """Translate texts without multiplying a blocked Google timeout.

    Probe one uncached region first. If Docker cannot reach either Google
    route, the circuit opens and the dispatcher can immediately use its
    configured shared fallback for the complete batch. A healthy route still
    translates the remaining regions concurrently.
    """
    if not texts:
        return []

    source_lang = _language_code(source_lang, allow_auto=True)
    target_lang = _language_code(target_lang)

    results = list(texts)
    pending: list[int] = []
    for index, text in enumerate(texts):
        if not text or not text.strip():
            results[index] = ""
            continue
        cache_key = (source_lang or "auto", target_lang, text)
        cached = _translation_cache.get(cache_key)
        if cached is not None:
            results[index] = cached
        else:
            pending.append(index)

    if not pending or time.monotonic() < _blocked_until:
        return results

    probe_index = pending.pop(0)
    results[probe_index] = await translate_text(
        texts[probe_index], source_lang, target_lang
    )
    if time.monotonic() < _blocked_until:
        return results

    if time.monotonic() >= _direct_blocked_until:
        # Reuse one keep-alive connection for the complete page.
        async with _new_client() as client:
            translated = await asyncio.gather(*(
                translate_text(
                    texts[index], source_lang, target_lang, _client=client
                )
                for index in pending
            ))
        for index, value in zip(pending, translated):
            results[index] = value
    else:
        # deep-translator is synchronous. Limit its worker fan-out so a page
        # with many regions does not saturate the Docker host or open dozens
        # of identical connections after the direct endpoint has failed.
        for start in range(0, len(pending), 4):
            chunk = pending[start:start + 4]
            translated = await asyncio.gather(*(
                translate_text(
                    texts[index], source_lang, target_lang, _skip_direct=True
                )
                for index in chunk
            ))
            for index, value in zip(chunk, translated):
                results[index] = value
            if time.monotonic() < _blocked_until:
                break
    return results


async def translate_regions(
    textlines: list[dict],
    source_lang: str = "auto",
    target_lang: str = "es",
) -> list[dict]:
    """Translate all text regions."""
    texts = [tl.get("text", "") for tl in textlines]
    translations = await translate_batch(texts, source_lang, target_lang)

    for tl, trans in zip(textlines, translations):
        tl["translated_text"] = trans

    log.info(f"Translated {len(textlines)} regions ({source_lang} → {target_lang}).")
    return textlines


async def cleanup():
    global _blocked_until, _direct_blocked_until
    _blocked_until = 0.0
    _direct_blocked_until = 0.0
    _translation_cache.clear()
