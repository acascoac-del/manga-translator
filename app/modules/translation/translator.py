"""
Translation Module - Multi-provider async translation.
Supports: MyMemory (free), Google Translate (free), LibreTranslate (self-hosted).
All providers use async HTTP for non-blocking I/O.
"""

import asyncio
from typing import Optional

import httpx

from app.core.config import Settings


# Language code mapping for different providers
LANG_MAP = {
    "ja": "ja", "jpn": "ja",
    "ko": "ko", "kor": "ko",
    "zh": "zh", "chi": "zh", "chs": "zh",
    "en": "en", "eng": "en",
    "es": "es", "spa": "es",
    "pt": "pt", "por": "pt",
    "fr": "fr", "fra": "fr",
    "de": "de", "deu": "de",
    "it": "it", "ita": "it",
}


class Translator:
    """
    Async translation with multiple providers.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: Optional[httpx.AsyncClient] = None

    async def load(self):
        """Initialize HTTP client."""
        self.client = httpx.AsyncClient(timeout=30.0)
        print(f"[Translator] Using {self.settings.TRANSLATION_SERVICE}")

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate a single text."""
        if not text or not text.strip():
            return ""

        source = LANG_MAP.get(source_lang, source_lang)
        target = LANG_MAP.get(target_lang, target_lang)

        try:
            if self.settings.TRANSLATION_SERVICE == "google":
                return await self._translate_google(text, source, target)
            elif self.settings.TRANSLATION_SERVICE == "libretranslate":
                return await self._translate_libre(text, source, target)
            else:
                return await self._translate_mymemory(text, source, target)
        except Exception as e:
            print(f"[Translator] Error: {e}")
            return text

    async def translate_batch(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate multiple texts in parallel."""
        tasks = [self.translate(t, source_lang, target_lang) for t in texts]
        return await asyncio.gather(*tasks)

    async def _translate_mymemory(self, text: str, source: str, target: str) -> str:
        """MyMemory API (free, no key required)."""
        resp = await self.client.get(
            self.settings.MYMEMORY_URL,
            params={"q": text, "langpair": f"{source}|{target}"},
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("responseStatus") == 200:
            return data["responseData"]["translatedText"]
        elif data.get("matches"):
            return data["matches"][0]["translation"]
        return text

    async def _translate_google(self, text: str, source: str, target: str) -> str:
        """Google Translate (unofficial, free)."""
        resp = await self.client.get(
            self.settings.GOOGLE_TRANSLATE_URL,
            params={"client": "gtx", "sl": source, "tl": target, "dt": "t", "q": text},
        )
        resp.raise_for_status()
        data = resp.json()

        if data and data[0]:
            return "".join(item[0] for item in data[0] if item[0])
        return text

    async def _translate_libre(self, text: str, source: str, target: str) -> str:
        """LibreTranslate (self-hosted)."""
        resp = await self.client.post(
            f"{self.settings.LIBRETRANSLATE_URL}/translate",
            json={"q": text, "source": source, "target": target},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("translatedText", text)
