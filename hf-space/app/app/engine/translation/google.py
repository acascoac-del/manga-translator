"""
Translation using Google Translate (free, online).
No API key required.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

log = logging.getLogger("translation.google")

_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
) -> str:
    """Translate a single text using Google Translate."""
    if not text or not text.strip():
        return ""

    client = await get_client()

    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {
            "client": "gtx",
            "sl": source_lang,
            "tl": target_lang,
            "dt": "t",
            "q": text,
        }
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if data and data[0]:
            return "".join(item[0] for item in data[0] if item[0])
        return text
    except Exception as e:
        log.error(f"Translation failed: {e}")
        return text


async def translate_batch(
    texts: list[str],
    source_lang: str = "auto",
    target_lang: str = "es",
) -> list[str]:
    """Translate multiple texts in parallel."""
    tasks = [translate_text(t, source_lang, target_lang) for t in texts]
    return await asyncio.gather(*tasks)


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
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
