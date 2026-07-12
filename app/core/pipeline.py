"""
Translation Pipeline - Orchestrates all modules.
GPU-optimized with persistent model loading.

Pipeline stages:
  1. Detection → Find text regions (YOLO/CTD)
  2. OCR → Read text (manga-ocr on GPU)
  3. Translation → Translate text (MyMemory/Google)
  4. Inpainting → Remove text (LaMa ONNX on GPU)
  5. Typesetting → Render translated text
"""

import asyncio
import base64
import io
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.modules.detection.detector import TextDetector
from app.modules.ocr.engine import OCREngine
from app.modules.translation.translator import Translator
from app.modules.inpainting.inpainter import Inpainter
from app.modules.typesetting.renderer import Renderer


class TranslationPipeline:
    """
    GPU-optimized translation pipeline.
    Models are loaded once and stay in GPU memory.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector: Optional[TextDetector] = None
        self.ocr: Optional[OCREngine] = None
        self.translator: Optional[Translator] = None
        self.inpainter: Optional[Inpainter] = None
        self.renderer: Optional[Renderer] = None
        self.models_loaded = False

    async def initialize(self):
        """Load all models into GPU memory (called once at startup)."""
        print("[Pipeline] Initializing modules...")

        # Load models in parallel for faster startup
        tasks = [
            self._init_detector(),
            self._init_ocr(),
            self._init_translator(),
            self._init_inpainter(),
            self._init_renderer(),
        ]
        await asyncio.gather(*tasks)

        self.models_loaded = True
        print("[Pipeline] All modules initialized.")

    async def _init_detector(self):
        print("[Pipeline] Loading text detector...")
        self.detector = TextDetector(self.settings)
        await self.detector.load()

    async def _init_ocr(self):
        print("[Pipeline] Loading OCR engine...")
        self.ocr = OCREngine(self.settings)
        await self.ocr.load()

    async def _init_translator(self):
        print("[Pipeline] Loading translator...")
        self.translator = Translator(self.settings)
        await self.translator.load()

    async def _init_inpainter(self):
        print("[Pipeline] Loading inpainter...")
        self.inpainter = Inpainter(self.settings)
        await self.inpainter.load()

    async def _init_renderer(self):
        print("[Pipeline] Loading renderer...")
        self.renderer = Renderer(self.settings)
        await self.renderer.load()

    async def process_image(
        self,
        image_bytes: bytes,
        source_lang: str = "ja",
        target_lang: str = "es",
        inpaint_method: str = "sample",
    ) -> dict[str, Any]:
        """
        Full pipeline: Detect → OCR → Translate → Inpaint → Render.
        Returns dict with image (base64), regions, and text pairs.
        """
        start = time.time()

        # Load image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Downscale if too large (saves VRAM)
        if max(image.size) > self.settings.MAX_IMAGE_SIZE:
            ratio = self.settings.MAX_IMAGE_SIZE / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Step 1: Detect text regions
        t1 = time.time()
        regions = await self.detect_regions_from_image(image)
        print(f"[Pipeline] Detection: {len(regions)} regions in {(time.time()-t1)*1000:.0f}ms")

        if not regions:
            return {
                "image": self._image_to_b64(image),
                "regions": [],
                "original_texts": [],
                "translated_texts": [],
            }

        # Step 2: OCR all regions (batched for GPU efficiency)
        t2 = time.time()
        ocr_results = await self.ocr_regions(image, regions)
        print(f"[Pipeline] OCR: {len(ocr_results)} texts in {(time.time()-t2)*1000:.0f}ms")

        # Filter low confidence
        valid = [r for r in ocr_results if r["confidence"] > self.settings.OCR_CONF_THRESHOLD]

        if not valid:
            return {
                "image": self._image_to_b64(image),
                "regions": [r["bbox"] for r in regions],
                "original_texts": [],
                "translated_texts": [],
            }

        # Step 3: Translate all texts (batched)
        t3 = time.time()
        translations = await self.translator.translate_batch(
            texts=[v["text"] for v in valid],
            source_lang=source_lang,
            target_lang=target_lang,
        )
        print(f"[Pipeline] Translation: {len(translations)} texts in {(time.time()-t3)*1000:.0f}ms")

        # Step 4: Inpaint all regions (batched GPU operations)
        t4 = time.time()
        canvas = image.copy()
        for v in valid:
            await self.inpainter.inpaint_region(canvas, v["bbox"], method=inpaint_method)
        print(f"[Pipeline] Inpainting: {len(valid)} regions in {(time.time()-t4)*1000:.0f}ms")

        # Step 5: Render translated text
        t5 = time.time()
        for i, v in enumerate(valid):
            self.renderer.render_text(
                canvas=canvas,
                text=translations[i],
                bbox=v["bbox"],
                orientation=v.get("orientation", "horizontal"),
            )
        print(f"[Pipeline] Rendering: {len(valid)} texts in {(time.time()-t5)*1000:.0f}ms")

        total = time.time() - start
        print(f"[Pipeline] Total: {total*1000:.0f}ms")

        return {
            "image": self._image_to_b64(canvas),
            "regions": [v["bbox"] for v in valid],
            "original_texts": [v["text"] for v in valid],
            "translated_texts": translations,
        }

    async def process_batch(
        self,
        image_bytes_list: list[bytes],
        source_lang: str = "ja",
        target_lang: str = "es",
    ) -> list[dict]:
        """Process multiple images in parallel."""
        tasks = [
            self.process_image(img, source_lang, target_lang)
            for img in image_bytes_list
        ]
        return await asyncio.gather(*tasks)

    async def detect_regions_from_image(self, image: Image.Image) -> list[dict]:
        """Detect text regions in a PIL Image."""
        return await self.detector.detect(image)

    async def detect_regions(self, image_bytes: bytes) -> list[dict]:
        """Detect text regions from raw bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return await self.detect_regions_from_image(image)

    async def ocr_regions(self, image: Image.Image, regions: list[dict]) -> list[dict]:
        """OCR all regions (batched for GPU efficiency)."""
        return await self.ocr.recognize_batch(image, regions)

    async def ocr_image(self, image_bytes: bytes) -> list[dict]:
        """OCR from raw bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        regions = await self.detector.detect(image)
        return await self.ocr.recognize_batch(image, regions)

    async def translate_texts(
        self, texts: list[str], source_lang: str, target_lang: str
    ) -> list[str]:
        """Translate a list of texts."""
        return await self.translator.translate_batch(texts, source_lang, target_lang)

    async def get_gpu_info(self) -> dict:
        """Get GPU memory info."""
        try:
            import torch
            if torch.cuda.is_available():
                return {
                    "device": torch.cuda.get_device_name(0),
                    "memory_total_mb": torch.cuda.get_device_properties(0).total_mem // (1024 * 1024),
                    "memory_used_mb": torch.cuda.memory_allocated(0) // (1024 * 1024),
                    "memory_free_mb": torch.cuda.memory_reserved(0) // (1024 * 1024),
                }
        except Exception:
            pass
        return {"device": "CPU"}

    async def cleanup(self):
        """Release GPU memory."""
        if self.ocr:
            await self.ocr.unload()
        if self.inpainter:
            await self.inpainter.unload()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _image_to_b64(image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
