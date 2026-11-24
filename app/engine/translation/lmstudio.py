"""Translation through LM Studio's OpenAI-compatible local API."""

from __future__ import annotations

import logging
import os
import asyncio
import time

import httpx

from .quality import (
    batch_prompt,
    clean_model_output,
    openai_chat_payload,
    parse_numbered_output,
    single_prompt,
)

log = logging.getLogger("translation.lmstudio")

_resolved_models: dict[tuple[str, str], str] = {}
_availability_cache: dict[tuple[str, str], tuple[bool, float]] = {}
_client: httpx.AsyncClient | None = None
_client_lock: asyncio.Lock | None = None

_MODEL_PREFERENCE = ("qwen", "gemma", "llama", "mistral", "phi")


def _normalize_host(host_str: str | None) -> str:
    """Make a host configured on Windows reachable from the Docker API."""
    if not host_str:
        return ""
    if os.path.exists("/.dockerenv"):
        host_str = host_str.replace("localhost", "host.docker.internal")
        host_str = host_str.replace("127.0.0.1", "host.docker.internal")
    return host_str


def _get_lmstudio_host() -> str:
    configured = os.environ.get("LMSTUDIO_HOST")
    if configured:
        return _normalize_host(configured)
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:1234/v1"
    return "http://localhost:1234/v1"


def _lmstudio_api_url(host: str | None, endpoint: str) -> str:
    """Build an OpenAI-compatible endpoint without duplicating ``/v1``."""
    base = (_normalize_host(host) or _get_lmstudio_host()).rstrip("/")
    endpoint = endpoint.strip("/")
    if base.lower().endswith("/v1"):
        return f"{base}/{endpoint}"
    return f"{base}/v1/{endpoint}"


def _get_lmstudio_model() -> str:
    return os.environ.get("LMSTUDIO_MODEL", "auto")


async def _resolve_model(
    client: httpx.AsyncClient, host: str, requested: str | None = None
) -> str:
    """Use the requested LM Studio model, or automatically select an available one."""
    normalized = (requested or _get_lmstudio_model()).strip()
    key = (host, normalized)
    if key in _resolved_models:
        return _resolved_models[key]

    response = await client.get(_lmstudio_api_url(host, "models"))
    response.raise_for_status()
    models = [item.get("id") for item in response.json().get("data", []) if item.get("id")]

    if normalized.lower() != "auto" and normalized in models:
        selected = normalized
    else:
        selected = next(
            (model for hint in _MODEL_PREFERENCE for model in models if hint in model.lower()),
            models[0] if models else "",
        )
    if not selected:
        raise RuntimeError("LM Studio no tiene ningún modelo disponible")
    if selected != normalized:
        log.info("LM Studio seleccionó automáticamente el modelo %r", selected)
    _resolved_models[key] = selected
    return selected


def _new_client() -> httpx.AsyncClient:
    timeout = float(os.getenv("LMSTUDIO_TIMEOUT_SECONDS", "75"))
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout, connect=3.0),
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
    )


async def get_client() -> httpx.AsyncClient:
    """Reuse one keep-alive client for model discovery and translations."""
    global _client, _client_lock
    if _client is not None and not _client.is_closed:
        return _client
    if _client_lock is None:
        _client_lock = asyncio.Lock()
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = _new_client()
        return _client


async def resolve_model(host: str, requested: str | None = None) -> str:
    return await _resolve_model(await get_client(), host, requested)


async def is_available(
    host: str | None = None,
    model: str | None = None,
    timeout: float = 2.0,
) -> bool:
    """Probe LM Studio without holding up another provider's fallback path."""
    api_host = _normalize_host(host) or _get_lmstudio_host()
    requested_model = (model or _get_lmstudio_model()).strip()
    cache_key = (api_host, requested_model)
    cached = _availability_cache.get(cache_key)
    now = time.monotonic()
    if cached and cached[1] > now:
        return cached[0]
    try:
        await asyncio.wait_for(resolve_model(api_host, model), timeout=timeout)
        _availability_cache[cache_key] = (True, now + 10.0)
        return True
    except Exception as exc:
        _availability_cache[cache_key] = (False, now + 45.0)
        log.info("LM Studio fallback unavailable at %s: %s", api_host, exc)
        return False


def _content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    return clean_model_output(choices[0].get("message", {}).get("content", ""))


async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str | None = None,
    model: str | None = None,
) -> str:
    if not text or not text.strip():
        return ""
    api_host = _normalize_host(host) or _get_lmstudio_host()
    try:
        client = await get_client()
        api_model = await resolve_model(api_host, model)
        response = await client.post(
            _lmstudio_api_url(api_host, "chat/completions"),
            json=openai_chat_payload(
                api_model,
                single_prompt(source_lang, target_lang, text),
                max_tokens=min(384, max(64, int(len(text) * 2.2) + 24)),
            ),
        )
        response.raise_for_status()
        translated = _content(response.json())
        return translated or text
    except Exception as exc:
        log.error("LM Studio translation failed: %s. Host=%s", exc, api_host)
        return text


async def translate_batch(
    texts: list[str],
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str | None = None,
    model: str | None = None,
) -> list[str]:
    indexed = [(i, text) for i, text in enumerate(texts) if text and text.strip()]
    results = list(texts)
    if not indexed:
        return results
    if len(indexed) == 1:
        i, text = indexed[0]
        results[i] = await translate_text(text, source_lang, target_lang, host, model)
        return results

    api_host = _normalize_host(host) or _get_lmstudio_host()
    try:
        numbered = "\n".join(f"{n + 1}. {text}" for n, (_, text) in enumerate(indexed))
        source_chars = sum(len(text) for _, text in indexed)
        client = await get_client()
        api_model = await resolve_model(api_host, model)
        response = await client.post(
            _lmstudio_api_url(api_host, "chat/completions"),
            json=openai_chat_payload(
                api_model,
                batch_prompt(source_lang, target_lang, numbered),
                max_tokens=min(
                    1024,
                    max(128, int(source_chars * 2.2) + len(indexed) * 10),
                ),
            ),
        )
        response.raise_for_status()
        raw = _content(response.json())
        parsed = parse_numbered_output(raw, len(indexed))
        for n, (i, _) in enumerate(indexed):
            if n in parsed:
                results[i] = parsed[n]
        missing = len(indexed) - len(parsed)
        if missing:
            log.warning(
                "LM Studio omitted %d numbered translation(s); leaving them for shared fallback",
                missing,
            )
        log.info("LM Studio translated %d texts", len(indexed))
    except Exception as exc:
        log.error("LM Studio batch translation failed: %s. Host=%s", exc, api_host)
        # A provider/model error affects the whole batch. Repeating the same
        # request once per balloon can freeze the PC for minutes; the common
        # dispatcher performs one alternate-provider batch instead.
    return results


async def translate_regions(
    textlines: list[dict],
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str | None = None,
    model: str | None = None,
) -> list[dict]:
    translations = await translate_batch(
        [line.get("text", "") for line in textlines], source_lang, target_lang, host, model
    )
    for line, translated in zip(textlines, translations):
        line["translated_text"] = translated
    return textlines


async def cleanup():
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
    _resolved_models.clear()
    _availability_cache.clear()
