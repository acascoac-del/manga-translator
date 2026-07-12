"""
Translation using Groq API.
Connects to https://api.groq.com/openai/v1/chat/completions
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

log = logging.getLogger("translation.groq")

_client: Optional[httpx.AsyncClient] = None

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

    api_model = model or "llama-3.1-8b-instant"
    client = await get_client()

    url = "https://api.groq.com/openai/v1/chat/completions"
    
    system_prompt = (
        f"You are a professional manga translator. Translate the following text from {source_lang} to {target_lang}. "
        "Keep the translation natural, colloquial, and matching the style of speech bubble dialogue. "
        "Output ONLY the translated text. DO NOT add any explanations, notes, greetings, or punctuation wrappers. "
        "Preserve line breaks if they exist. Do not enclose the output in quotes."
    )

    try:
        payload = {
            "model": api_model,
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\nText to translate:\n{text}"}
            ]
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code != 200:
            log.error(f"Groq API returned {resp.status_code}: {resp.text}")
            raise Exception(f"HTTP {resp.status_code}: {resp.text}")
            
        data = resp.json()
        translated = data["choices"][0]["message"]["content"].strip()
        
        # Clean up any quotes that the model might have returned
        if translated.startswith('"') and translated.endswith('"'):
            translated = translated[1:-1].strip()
        elif translated.startswith("'") and translated.endswith("'"):
            translated = translated[1:-1].strip()
            
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
        log.warning("Groq API key missing for batch translation.")
        return results
    
    if len(indexed) == 1:
        i, t = indexed[0]
        results[i] = await translate_text(t, source_lang, target_lang, api_key, model)
        return results
    
    api_model = model or "llama-3.1-8b-instant"
    client = await get_client()
    url = "https://api.groq.com/openai/v1/chat/completions"

    numbered = "\n".join(f"{n+1}. {t}" for n, (_, t) in enumerate(indexed))
    system_prompt = (
        f"You are a professional manga translator. Translate each line below from {source_lang} to {target_lang}.\n"
        "Rules:\n"
        "- Output ONLY the translations, one per line, numbered exactly as in the input.\n"
        "- Keep translations natural, short, and matching manga speech bubble style.\n"
        "- Do NOT add explanations, notes, or extra lines."
    )

    try:
        payload = {
            "model": api_model,
            "messages": [
                {"role": "user", "content": f"{system_prompt}\n\nLines to translate:\n{numbered}"}
            ]
        }
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

        # Parse numbered lines back
        import re
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        parsed = {}
        for line in lines:
            m = re.match(r'^(\d+)\.?\s+(.*)', line)
            if m:
                num = int(m.group(1)) - 1
                parsed[num] = m.group(2).strip()
        
        if len(parsed) >= len(indexed):
            for n, (i, _) in enumerate(indexed):
                if n in parsed:
                    results[i] = parsed[n]
        else:
            clean_lines = []
            for line in lines:
                m = re.match(r'^\d+\.?\s+(.*)', line)
                clean_lines.append(m.group(1).strip() if m else line)
            for n, (i, _) in enumerate(indexed):
                if n < len(clean_lines):
                    results[i] = clean_lines[n]
                    
        log.info(f"Groq batch translated {len(indexed)} texts")
    except Exception as e:
        err_msg = str(e)
        if hasattr(e, 'response') and e.response is not None:
            err_msg += f" | Body: {e.response.text}"
        log.error(f"Groq batch translation failed: {err_msg}. Model={api_model}")
        for i, t in indexed:
            try:
                results[i] = await translate_text(t, source_lang, target_lang, api_key, model)
            except Exception:
                pass
    
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
