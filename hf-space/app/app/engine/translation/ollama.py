"""
Translation using a local Ollama Docker instance.
Connects to localhost:8080 (external) or http://ollama:11434 (docker-internal).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import httpx

log = logging.getLogger("translation.ollama")

_client: Optional[httpx.AsyncClient] = None


def _normalize_host(host_str: str) -> str:
    """Normalize host string, translating localhost to host.docker.internal if running in Docker."""
    if not host_str:
        return ""
        
    # Check if we are running in Docker
    in_docker = os.path.exists("/.dockerenv")
    
    if in_docker:
        # Rewrite localhost/127.0.0.1 to host.docker.internal
        if "localhost" in host_str:
            host_str = host_str.replace("localhost", "host.docker.internal")
        elif "127.0.0.1" in host_str:
            host_str = host_str.replace("127.0.0.1", "host.docker.internal")
            
    return host_str


def _get_ollama_host() -> str:
    """Resolve Ollama API endpoint host."""
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        return _normalize_host(env_host)
    if os.path.exists("/.dockerenv"):
        return "http://host.docker.internal:8080"
    return "http://localhost:8080"


def _get_ollama_model() -> str:
    """Resolve Ollama model to use."""
    return os.environ.get("OLLAMA_MODEL", "llama3")


async def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        # Timeout: enough for batch generation but not so long it blocks the server
        _client = httpx.AsyncClient(timeout=90.0)
    return _client


async def translate_text(
    text: str,
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str = None,
    model: str = None,
) -> str:
    """Translate a single text using Ollama."""
    if not text or not text.strip():
        return ""

    api_host = _normalize_host(host) or _get_ollama_host()
    api_model = model or _get_ollama_model()
    client = await get_client()

    url = f"{api_host.rstrip('/')}/api/generate"
    
    # System prompt to ensure we only get translation output
    prompt = (
        f"You are a professional manga translator. Translate the following text from {source_lang} to {target_lang}. "
        "Keep the translation natural, colloquial, and matching the style of speech bubble dialogue. "
        "Output ONLY the translated text. DO NOT add any explanations, notes, greetings, or punctuation wrappers. "
        "Preserve line breaks if they exist. Do not enclose the output in quotes.\n\n"
        f"Text to translate:\n{text}"
    )

    try:
        payload = {
            "model": api_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
            }
        }
        resp = await client.post(url, json=payload)
        
        if resp.status_code == 404:
            # Model not found or endpoint incorrect
            log.warning(f"Ollama returned 404. Model '{api_model}' may not be pulled. Run 'ollama pull {api_model}'")
            return text
            
        resp.raise_for_status()
        data = resp.json()
        translated = data.get("response", "").strip()
        
        # Clean up any quotes that the model might have returned
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1].strip()
        elif translated.startswith("'") and translated.endswith("'"):
            translated = translated[1:-1].strip()
            
        return translated if translated else text
    except Exception as e:
        log.error(f"Ollama translation failed: {e}. Host={api_host}, Model={api_model}")
        # Return original text as fallback
        return text


async def translate_batch(
    texts: list[str],
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str = None,
    model: str = None,
) -> list[str]:
    """Translate all texts in a SINGLE Ollama call to avoid parallel timeouts.
    Ollama is a sequential LLM server - parallel requests cause 30s timeouts.
    We send all texts as a numbered list in one prompt and parse the response.
    """
    # Filter out empty texts, keeping track of their original indices
    indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    results = list(texts)  # start with originals as fallback
    
    if not indexed:
        return results
    
    # If only one text, use the simpler single-text path
    if len(indexed) == 1:
        i, t = indexed[0]
        results[i] = await translate_text(t, source_lang, target_lang, host, model)
        return results
    
    api_host = _normalize_host(host) or _get_ollama_host()
    api_model = model or _get_ollama_model()
    client = await get_client()
    url = f"{api_host.rstrip('/')}/api/generate"

    # Build a numbered list prompt with all texts
    numbered = "\n".join(f"{n+1}. {t}" for n, (_, t) in enumerate(indexed))
    prompt = (
        f"You are a professional manga translator. Translate each line below from {source_lang} to {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY the translations, one per line, numbered the same way.\n"
        "- Keep translations natural, short, and matching manga speech bubble style.\n"
        "- Do NOT add explanations, notes, or extra lines.\n\n"
        f"Lines to translate:\n{numbered}"
    )

    try:
        payload = {
            "model": api_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2, "top_p": 0.9},
        }
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("response", "").strip()

        # Parse numbered lines back
        import re
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        # Try to match "1. text" pattern
        parsed = {}
        for line in lines:
            m = re.match(r'^(\d+)\.?\s+(.*)', line)
            if m:
                num = int(m.group(1)) - 1
                parsed[num] = m.group(2).strip()
        
        # If we matched all, great; otherwise fall back to positional
        if len(parsed) >= len(indexed):
            for n, (i, _) in enumerate(indexed):
                if n in parsed:
                    results[i] = parsed[n]
        else:
            # Positional fallback: strip numbering if present
            clean_lines = []
            for line in lines:
                m = re.match(r'^\d+\.?\s+(.*)', line)
                clean_lines.append(m.group(1).strip() if m else line)
            for n, (i, _) in enumerate(indexed):
                if n < len(clean_lines):
                    results[i] = clean_lines[n]
                    
        log.info(f"Ollama batch translated {len(indexed)} texts in single call")
    except Exception as e:
        log.error(f"Ollama batch translation failed: {e}. Host={api_host}, Model={api_model}")
        # On failure, try each text individually as fallback
        for i, t in indexed:
            try:
                results[i] = await translate_text(t, source_lang, target_lang, host, model)
            except Exception:
                pass  # keep original
    
    return results


async def translate_regions(
    textlines: list[dict],
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str = None,
    model: str = None,
) -> list[dict]:
    """Translate all text regions."""
    texts = [tl.get("text", "") for tl in textlines]
    translations = await translate_batch(texts, source_lang, target_lang, host, model)

    for tl, trans in zip(textlines, translations):
        tl["translated_text"] = trans

    log.info(f"Ollama translated {len(textlines)} regions ({source_lang} → {target_lang}).")
    return textlines


async def cleanup():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
