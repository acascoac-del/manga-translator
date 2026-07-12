"""
OCR Engine - GPU-optimized manga text recognition.
Uses manga-ocr (Hugging Face) on CUDA for fast inference.
Supports batch processing for multiple regions.
"""

import asyncio
import io
from typing import Optional

import numpy as np
import torch
from PIL import Image

from app.core.config import Settings


class OCREngine:
    """
    GPU-accelerated OCR using manga-ocr.
    Model persists in GPU memory for zero-copy inference.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.device = settings.CUDA_DEVICE
        self.processor = None

    async def load(self):
        """Load manga-ocr model into GPU memory."""
        try:
            from manga_ocr import MangaOcr

            self.model = MangaOcr()
            print(f"[OCR] Loaded manga-ocr on {self.device}")

        except Exception as e:
            print(f"[OCR] Failed to load manga-ocr: {e}")
            self.model = None

    async def recognize(self, image: Image.Image, region: Optional[dict] = None) -> dict:
        """
        Recognize text in an image region.
        Returns { text, confidence, bbox, orientation }.
        """
        if region:
            # Crop region from image
            x, y, w, h = region["x"], region["y"], region["width"], region["height"]
            cropped = image.crop((x, y, x + w, y + h))
        else:
            cropped = image

        if self.model is None:
            return {"text": "", "confidence": 0, "bbox": region, "orientation": "horizontal"}

        # Run OCR in thread pool (non-blocking)
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(None, lambda: self.model(cropped))

        return {
            "text": text.strip() if text else "",
            "confidence": 0.9,  # manga-ocr doesn't provide confidence
            "bbox": region,
            "orientation": region.get("orientation", "horizontal") if region else "horizontal",
        }

    async def recognize_batch(self, image: Image.Image, regions: list[dict]) -> list[dict]:
        """
        Recognize text in multiple regions (sequential but async).
        manga-ocr doesn't support true batching, but we use async for non-blocking.
        """
        tasks = [self.recognize(image, region) for region in regions]
        return await asyncio.gather(*tasks)

    async def unload(self):
        """Release GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
