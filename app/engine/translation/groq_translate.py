"""
Translation using Groq API.
Connects to https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Optional

import httpx

from .quality import (
    batch_prompt,
    clean_model_output,
    openai_chat_payload,
    parse_numbered_output,
    single_prompt,
)

log = logging.getLogger("translation.groq")

_client: Optional[httpx.AsyncClient] = None
_resolved_models: dict[str, str] = {}

_GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_GROQ_MODEL_PREFERENCE = (
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
)


def _model_cache_key(api_key: str, requested: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
    return f"{digest}:{requested}"


def _choose_available_model(model_ids: list[str], requested: str | None) -> str:
    """Select a chat model without relying on a stale UI model list."""
    requested = (requested or "auto").strip()
    if requested.lower() != "auto" and requested in model_ids:
        return requested
    for candidate in _GROQ_MODEL_PREFERENCE:
        if candidate in model_ids:
            return candidate
    excluded = ("whisper", "tts", "guard", "safety", "compound")
    compatible = [
        model_id for model_id in model_ids
        if not any(token in model_id.lower() for token in excluded)
    ]
    if compatible:
        return compatible[0]
    # This is only used if the model-list request itself is unavailable.
    return "llama-3.1-8b-instant"


async def resolve_model(api_key: str, requested: str | None = None) -> str:
    """Resolve and cache a model that the supplied Groq key can access."""
    normalized = (requested or "auto").strip()
    cache_key = _model_cache_key(api_key, normalized)
    if cache_key in _resolved_models:
        return _resolved_models[cache_key]

    client = await get_client()
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        response = await client.get(_GROQ_MODELS_URL, headers=headers)
        response.raise_for_status()
        model_ids = [
            item.get("id") for item in response.json().get("data", [])
            if item.get("id")
        ]
        selected = _choose_available_model(model_ids, normalized)
    except Exception as exc:
        selected = (
            normalized if normalized.lower() != "auto"
            else "llama-3.1-8b-instant"
        )
        log.warning("Could not query Groq models; trying %s: %s", selected, exc)

    if selected != normalized:
        log.warning("Groq model %r unavailable; automatically using %r", normalized, selected)
    _resolved_models[cache_key] = selected
    return selected

async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0, verify=False)
    return _client


async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
    api_key: str = None,
    model: str = None,
) -> str:
    """Translate a single text using Groq."""
    if not text or not text.strip():
        return ""
    if not api_key:
        log.warning("Groq API key is missing. Cannot translate.")
        return text

    api_model = await resolve_model(api_key, model)
    client = await get_client()

    url = _GROQ_CHAT_URL
    
    prompt = single_prompt(source_lang, target_lang, text)

    try:
        payload = openai_chat_payload(api_model, prompt)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            log.error(f"Groq API returned {resp.status_code}: {resp.text}")
            raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            
        data = resp.json()
        translated = clean_model_output(data["choices"][0]["message"]["content"])
        return translated if translated else text
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            err_msg += f" | Body: {e.response.text}"
        log.error(f"Groq translation failed: {err_msg}. Model={api_model}")
        return text


async def translate_batch(
    texts: list[str],
    source_lang: str = "auto",
    target_lang: str = "es",
    api_key: str = None,
    model: str = None,
) -> list[str]:
    """Translate all texts in a single Groq call."""
    indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    results = list(texts)
    
    if not indexed:
        return results
    if not api_key:
        log.warning("Groq API key missing; leaving regions for shared fallback.")
        return results
    
    if len(indexed) == 1:
        i, t = indexed[0]
        results[i] = await translate_text(t, source_lang, target_lang, api_key, model)
        return results
    
    api_model = await resolve_model(api_key, model)
    client = await get_client()
    url = _GROQ_CHAT_URL

    numbered = "\n".join(f"{n+1}. {t}" for n, (_, t) in enumerate(indexed))
    prompt = batch_prompt(source_lang, target_lang, numbered)

    try:
        payload = openai_chat_payload(api_model, prompt)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            log.error(f"Groq API returned {resp.status_code}: {resp.text}")
            raise Exception(f"HTTP {resp.status_code}: {resp.text}")
        
        data = resp.json()
        raw = data["choices"][0]["message"]["content"].strip()
        log.info(f"GROQ RAW RESPONSE:\n{raw}")

        parsed = parse_numbered_output(raw, len(indexed))
        
        for n, (i, _) in enumerate(indexed):
            if n in parsed:
                results[i] = parsed[n]

        # A malformed batch response must not shift translations between
        # bubbles. Retry only entries whose explicit number was missing.
        for n, (i, text) in enumerate(indexed):
            if n not in parsed:
                results[i] = await translate_text(text, source_lang, target_lang, api_key, model)

        log.info(f"Groq batch translated {len(indexed)} texts")
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            err_msg += f" | Body: {e.response.text}"
        log.error(f"Groq batch translation failed: {err_msg}. Model={api_model}")
        # Provider/auth/model failures affect the whole batch. Do not repeat
        # the same failed request once per balloon; the dispatcher performs
        # one shared fallback pass below this engine.
    
    return results


async def translate_regions(
    textlines: list[dict],
    source_lang: str = "auto",
    target_lang: str = "es",
    api_key: str = None,
    model: str = None,
) -> list[dict]:
    """Translate all text regions."""
    texts = [tl.get("text", "") for tl in textlines]
    translations = await translate_batch(texts, source_lang, target_lang, api_key, model)

    for tl, trans in zip(textlines, translations):
        tl["translated_text"] = trans

    log.info(f"Groq translated {len(textlines)} regions ({source_lang} → {target_lang}).")
    return textlines


async def cleanup():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
    _resolved_models.clear()
