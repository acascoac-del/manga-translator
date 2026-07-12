import logging
import httpx
from typing import List, Dict, Any

log = logging.getLogger("translation.libretranslate")
_client: httpx.AsyncClient | None = None

def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client

async def translate_regions(
    textlines: List[Dict[str, Any]],
    source_lang: str = "auto",
    target_lang: str = "es",
    host: str = "http://localhost:5000",
) -> List[Dict[str, Any]]:
    """
    Translate using a local LibreTranslate instance.
    """
    if not textlines:
        return textlines

    client = get_client()
    url = f"{host.rstrip('/')}/translate"

    if source_lang == "auto":
        source_lang = "auto" # LibreTranslate uses "auto" for detection
    
    texts = [t["text"] for t in textlines]
    
    try:
        response = await client.post(
            url,
            json={
                "q": texts,
                "source": source_lang,
                "target": target_lang,
                "format": "text",
            }
        )
        response.raise_for_status()
        data = response.json()
        
        translated_texts = data.get("translatedText", [])
        if isinstance(translated_texts, str):
            translated_texts = [translated_texts]
            
        for i, text in enumerate(translated_texts):
            if i < len(textlines):
                textlines[i]["text"] = text
                
    except Exception as e:
        log.error(f"LibreTranslate translation failed: {e}. Host={host}")
        pass

    return textlines

async def cleanup():
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
