"""
Translation Dispatcher - Routes translation requests to Google Translate or Ollama.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

from . import google
from . import ollama
from . import libretranslate

log = logging.getLogger("translation.dispatcher")


async def translate_regions(
    textlines: List[Dict[str, Any]],
    source_lang: str = "auto",
    target_lang: str = "es",
    translator: str = "google",
    ollama_host: str | None = None,
    ollama_model: str | None = None,
    libretranslate_host: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Route translation regions request to the appropriate translator engine.
    """
    translator = translator.lower().strip()
    
    if translator == "ollama":
        log.info(f"Dispatching translation to Ollama (model={ollama_model or 'default'})")
        return await ollama.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
            host=ollama_host,
            model=ollama_model,
        )
    elif translator == "libretranslate":
        log.info(f"Dispatching translation to LibreTranslate (host={libretranslate_host})")
        return await libretranslate.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
            host=libretranslate_host or "http://localhost:5000",
        )
    else:
        # Default fallback to Google Translate
        log.info("Dispatching translation to Google Translate")
        return await google.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
        )


async def cleanup():
    """Cleanup all clients."""
    await google.cleanup()
    await ollama.cleanup()
    await libretranslate.cleanup()
