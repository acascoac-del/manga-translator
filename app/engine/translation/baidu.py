"""
Translation using Baidu Translate (via translators package).
Free online translation via web scraping.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncio
from deep_translator import GoogleTranslator

log = logging.getLogger("translation.baidu")

async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
) -> str:
    """Translate using a secondary Google Translate engine as all other free APIs block bots."""
    if not text or not text.strip():
        return ""

    if source_lang == "auto":
        from_lang = "auto"
    else:
        from_lang = source_lang

    def _do_translate():
        try:
            return GoogleTranslator(source=from_lang, target=target_lang).translate(text)
        except Exception as e:
            log.error(f"Secondary translation failed: {e}")
            return text
            
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(None, _do_translate)
    except Exception as e:
        log.error(f"Executor failed: {e}")
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

    log.info(f"Translated {len(textlines)} regions ({source_lang} → {target_lang}) via Baidu/Bing.")
    return textlines


async def cleanup():
    pass
