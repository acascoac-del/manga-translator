"""
OCR using manga-ocr.
Specialized model for Japanese manga text recognition.
Handles transformers version compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
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
    if _model_lock is None:
        _model_lock = asyncio.Lock()

    async with _model_lock:
        if _model is not None:
            return

        log.info("Loading manga-ocr model...")
        try:
            # Try loading manga-ocr normally first with CUDA/CPU support
            from manga_ocr import MangaOcr
            use_cpu = not device.startswith("cuda")
            _model = MangaOcr(force_cpu=use_cpu)
            log.info(f"manga-ocr loaded (standard, force_cpu={use_cpu}).")
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
                    outputs = self.model.generate(
                        **inputs, max_length=96, num_beams=1, do_sample=False
                    )
                else:
                    # Use pixel_values directly
                    outputs = self.model.generate(
                        pixel_values=img_tensor,
                        max_length=96,
                        num_beams=1,
                        do_sample=False,
                    )

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
    prepared: list[tuple[dict, Image.Image]] = []

    for tl in textlines:
        bbox = tl["bbox"]
        x1, y1, x2, y2 = bbox

        # CTD sometimes returns one narrow Japanese column rather than the
        # complete vertical text block. Dynamic padding gives manga-ocr enough
        # neighboring context to reconstruct the full sentence.
        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        pad_x = max(32, int(box_width * 0.22))
        pad_y = max(32, int(box_height * 0.12))
        h, w = img_rgb.shape[:2]
        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(w, x2 + pad_x)
        py2 = min(h, y2 + pad_y)

        # Crop region from image
        crop = img_rgb[py1:py2, px1:px2]
        if crop.size == 0:
            tl["text"] = ""
            continue

        # Check if it's inverted (white text on dark background)
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        if np.mean(gray_crop) < 110:
            # Invert colors: manga-ocr works much better with black text on white
            crop = cv2.bitwise_not(crop)

        prepared.append((tl, Image.fromarray(crop)))

    def _run_batch(
        batch_items: list[tuple[dict, Image.Image]] | None = None,
    ):
        """Use the native VisionEncoderDecoder model in small batches.

        The public MangaOcr callable invokes ``model.generate`` once per crop.
        Its feature extractor and model support batches, so page OCR can share
        encoder work and GPU launches without changing recognition semantics.
        """
        items = prepared if batch_items is None else batch_items
        if not items:
            return []

        if not all(
            hasattr(_model, name)
            for name in ("feature_extractor", "model", "tokenizer")
        ):
            return [_model(image) for _, image in items]

        import torch
        from manga_ocr.ocr import post_process

        outputs: list[str] = []
        # Four CPU crops amortize encoder/generation overhead substantially.
        # Native thread pools are capped to four in Docker, so this does not
        # restore the previous all-core contention that froze the host.
        uses_cuda = str(_model.model.device).startswith("cuda")
        default_batch_size = 12 if uses_cuda else 4
        batch_size = max(
            1,
            int(os.getenv("MANGA_OCR_BATCH_SIZE", str(default_batch_size))),
        )
        # Manga dialogue regions are far shorter than 128 tokens.  The old
        # 300-token ceiling made malformed/noisy crops occupy the GPU for much
        # longer and delayed every translator because OCR is shared by all of
        # them.
        max_length = max(32, int(os.getenv("MANGA_OCR_MAX_LENGTH", "96")))
        for start in range(0, len(items), batch_size):
            images = [
                image.convert("L").convert("RGB")
                for _, image in items[start:start + batch_size]
            ]
            pixels = _model.feature_extractor(
                images, return_tensors="pt"
            ).pixel_values.to(_model.model.device)
            with torch.inference_mode():
                generated = _model.model.generate(
                    pixels,
                    max_length=max_length,
                    use_cache=True,
                    num_beams=1,
                    do_sample=False,
                )
            decoded = _model.tokenizer.batch_decode(
                generated.cpu(), skip_special_tokens=True
            )
            outputs.extend(post_process(text) for text in decoded)
        return outputs

    texts = await loop.run_in_executor(None, _run_batch)
    for (tl, _), text in zip(prepared, texts):
        tl["text"] = text.strip() if text else ""

    # Manga OCR occasionally returns an empty string (or a few Latin noise
    # characters) for a narrow Japanese column even though the detector found
    # real ink. Retry only a handful of those regions with wider context. This
    # recovers otherwise blank balloons without doubling OCR time for the page.
    retry_limit = max(0, int(os.getenv("MANGA_OCR_RETRY_LIMIT", "6")))

    def _native_count(value: str) -> int:
        return sum(
            "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
            for char in value
        )

    retry_items: list[tuple[dict, Image.Image]] = []
    if retry_limit:
        h, w = img_rgb.shape[:2]
        for tl, _ in prepared:
            current = str(tl.get("text", "")).strip()
            compact = "".join(char for char in current if char.isalnum())
            if current and (_native_count(current) > 0 or len(compact) > 4):
                continue

            x1, y1, x2, y2 = [int(v) for v in tl["bbox"]]
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)
            pad_x = max(48, int(box_width * 0.55))
            pad_y = max(40, int(box_height * 0.25))
            rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
            crop = img_rgb[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            gray_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            ink_ratio = float(np.mean(gray_crop < 180))
            if not 0.003 <= ink_ratio <= 0.48:
                continue
            if float(np.mean(gray_crop)) < 110:
                crop = cv2.bitwise_not(crop)

            retry_items.append((tl, Image.fromarray(crop)))
            if len(retry_items) >= retry_limit:
                break

    if retry_items:
        retry_texts = await loop.run_in_executor(None, _run_batch, retry_items)
        for (tl, _), retry_text in zip(retry_items, retry_texts):
            retry_text = retry_text.strip() if retry_text else ""
            current = str(tl.get("text", "")).strip()
            if (
                _native_count(retry_text) > _native_count(current)
                or (not current and retry_text)
            ):
                tl["text"] = retry_text

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
