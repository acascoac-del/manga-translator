"""
Translation Dispatcher - Routes translation requests to the selected provider.
"""

from __future__ import annotations

import logging
import os
from typing import List, Dict, Any

from . import google
from . import lmstudio
from . import baidu
from .quality import needs_google_fallback

log = logging.getLogger("translation.dispatcher")


async def translate_regions(
    textlines: List[Dict[str, Any]],
    source_lang: str = "auto",
    target_lang: str = "es",
    translator: str = "google",
    lmstudio_host: str | None = None,
    lmstudio_model: str | None = None,
    groq_api_key: str | None = None,
    groq_model: str | None = None,
) -> List[Dict[str, Any]]:
    """
    Route translation regions request to the appropriate translator engine.
    """
    translator = translator.lower().strip()
    # Keep the old identifier as a migration alias for existing API clients.
    if translator in {"lmstudio", "ollama"}:
        log.info(f"Dispatching translation to LM Studio (model={lmstudio_model or 'auto'})")
        translated_regions = await lmstudio.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
            host=lmstudio_host,
            model=lmstudio_model,
        )
    elif translator == "baidu":
        log.info("Dispatching translation to Baidu Translate (via translators)")
        translated_regions = await baidu.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    elif translator == "groq":
        from . import groq_translate
        log.info(f"Dispatching translation to Groq (model={groq_model})")
        translated_regions = await groq_translate.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
            api_key=groq_api_key,
            model=groq_model,
        )
    elif translator == "llamacpp":
        from . import llamacpp
        log.info(f"Dispatching translation to internal llama.cpp")
        translated_regions = await llamacpp.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
        )
    else:
        # Default fallback to Google Translate
        log.info("Dispatching translation to Google Translate")
        translated_regions = await google.translate_regions(
            textlines,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    # Repair only failed entries, once per provider and as a batch. The old
    # quality pass called Google for every bad bubble even when Google was
    # blocked, multiplying latency and sometimes leaving erased balloons.
    def failed_indices() -> list[int]:
        return [
            index
            for index, region in enumerate(translated_regions)
            if (region.get("text") or "").strip()
            and needs_google_fallback(
                region.get("text", ""),
                region.get("translated_text", ""),
                source_lang,
                target_lang,
            )
        ]

    async def apply_fallback(provider_name: str) -> None:
        indices = failed_indices()
        if not indices:
            return
        subset = [dict(translated_regions[index]) for index in indices]
        if provider_name == "groq":
            from . import groq_translate

            repaired = await groq_translate.translate_regions(
                subset,
                source_lang=source_lang,
                target_lang=target_lang,
                api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
                model=groq_model,
            )
        elif provider_name == "llamacpp":
            from . import llamacpp

            repaired = await llamacpp.translate_regions(
                subset,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        elif provider_name == "google":
            repaired = await google.translate_regions(
                subset,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        else:
            repaired = await lmstudio.translate_regions(
                subset,
                source_lang=source_lang,
                target_lang=target_lang,
                host=lmstudio_host,
                model=lmstudio_model,
            )
        for index, candidate in zip(indices, repaired):
            output = candidate.get("translated_text", "")
            source = translated_regions[index].get("text", "")
            if not needs_google_fallback(source, output, source_lang, target_lang):
                translated_regions[index]["translated_text"] = output

    # Use at most one alternate provider. The previous chain could call Groq,
    # LM Studio and llama.cpp consecutively for the same page, which made a
    # temporary provider failure look like a frozen translation job.
    fallback_provider: str | None = None
    groq_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    failed = failed_indices()
    if failed and translator in {"llamacpp", "lmstudio", "ollama", "groq"}:
        # A local/chat model may skip a short OCR fragment or copy Japanese.
        # Google handles those small repair batches quickly and prevents the
        # pipeline from restoring visibly untranslated source text.
        fallback_provider = "google"
    elif failed and translator != "groq" and groq_key:
        fallback_provider = "groq"
    elif failed:
        if (
            translator not in {"lmstudio", "ollama"}
            and await lmstudio.is_available(
                lmstudio_host, lmstudio_model, timeout=1.0
            )
        ):
            fallback_provider = "lmstudio"
        elif translator in {"google", "baidu"}:
            # Do not leave erased balloons when Google is unavailable and LM
            # Studio is not running. The old ``elif`` structure skipped this
            # branch precisely in that situation.
            fallback_provider = "llamacpp"

    if fallback_provider:
        log.info(
            "Repairing %d failed translations with %s",
            len(failed_indices()),
            fallback_provider,
        )
        await apply_fallback(fallback_provider)

    remaining = failed_indices()
    if remaining:
        log.warning(
            "%d translation(s) remain unresolved; preserving their source pixels",
            len(remaining),
        )
    return translated_regions


async def cleanup():
    """Cleanup all clients."""
    await google.cleanup()
    await lmstudio.cleanup()
    await baidu.cleanup()
    try:
        from . import llamacpp
        await llamacpp.cleanup()
    except Exception:
        pass
