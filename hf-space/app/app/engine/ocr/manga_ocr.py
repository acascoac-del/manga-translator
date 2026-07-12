"""
OCR using manga-ocr.
Specialized model for Japanese manga text recognition.
Handles transformers version compatibility.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("ocr.manga_ocr")

_model = None
_model_lock = None


async def load_model(device: str = "cpu"):
    """Load manga-ocr model with compatibility fix."""
    global _model, _model_lock
    if _model is not None:
        return

    import asyncio
    _model_lock = asyncio.Lock()

    async with _model_lock:
        if _model is not None:
            return

        log.info("Loading manga-ocr model...")
        try:
            # Try loading manga-ocr normally first
            from manga_ocr import MangaOcr
            _model = MangaOcr()
            log.info("manga-ocr loaded (standard).")
        except Exception as e:
            log.warning(f"Standard manga-ocr load failed: {e}")
            log.info("Trying manual model load...")
            try:
                _model = _load_manual(device)
                log.info("manga-ocr loaded (manual).")
            except Exception as e2:
                log.error(f"Manual manga-ocr load also failed: {e2}")
                _model = None
                raise


def _load_manual(device: str = "cpu"):
    """Load manga-ocr manually bypassing AutoFeatureExtractor."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoProcessor
    import torch

    model_name = "kha-white/manga-ocr-base"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    if device.startswith("cuda"):
        model = model.to(device)

    model.eval()

    # Try to load processor, fallback to manual image processing
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception:
        processor = None

    # Create a wrapper class that mimics MangaOcr interface
    class ManualMangaOcr:
        def __init__(self, model, tokenizer, processor, device):
            self.model = model
            self.tokenizer = tokenizer
            self.processor = processor
            self.device = device

        def __call__(self, image: Image.Image) -> str:
            return self._recognize(image)

        def _recognize(self, image: Image.Image) -> str:
            import torchvision.transforms as T

            # Preprocess image
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            img_tensor = transform(image.convert("RGB")).unsqueeze(0)

            if self.device.startswith("cuda"):
                img_tensor = img_tensor.to(self.device)

            # Generate text
            with torch.no_grad():
                if self.processor is not None:
                    inputs = self.processor(images=image.convert("RGB"), return_tensors="pt")
                    if self.device.startswith("cuda"):
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    outputs = self.model.generate(**inputs, max_length=128)
                else:
                    # Use pixel_values directly
                    outputs = self.model.generate(pixel_values=img_tensor, max_length=128)

            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return text.strip()

    return ManualMangaOcr(model, tokenizer, processor, device)


async def recognize(
    img_rgb: np.ndarray,
    textlines: list[dict],
    device: str = "cpu",
) -> list[dict]:
    """
    OCR each detected text region.
    Returns textlines with 'text' field added.
    """
    global _model
    if _model is None:
        raise RuntimeError("manga-ocr not loaded. Call load_model() first.")

    loop = asyncio.get_event_loop()

    for tl in textlines:
        bbox = tl["bbox"]
        x1, y1, x2, y2 = bbox

        # Crop region from image
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            tl["text"] = ""
            continue

        # Convert to PIL
        pil_crop = Image.fromarray(crop)

        # Run OCR in thread pool
        text = await loop.run_in_executor(None, lambda c=pil_crop: _model(c))
        tl["text"] = text.strip() if text else ""

    recognized = [tl for tl in textlines if tl.get("text", "").strip()]
    log.info(f"OCR recognized {len(recognized)}/{len(textlines)} regions.")
    return textlines


def unload():
    """Unload model to free memory."""
    global _model
    if _model is not None:
        del _model
        _model = None
        log.info("manga-ocr unloaded.")
