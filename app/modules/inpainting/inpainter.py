"""
Inpainting Module - GPU-accelerated text removal.
Supports: LaMa (ONNX on GPU), OpenCV (CPU fallback).
Removes text from speech bubbles and reconstructs background.
"""

import asyncio
import io
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.core.config import Settings


class Inpainter:
    """
    GPU-accelerated inpainting for text removal.
    Uses LaMa ONNX on CUDA for best quality.
    Falls back to OpenCV if ONNX unavailable.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session = None
        self.use_onnx = False

    async def load(self):
        """Load LaMa ONNX model on GPU."""
        try:
            import onnxruntime as ort

            # Try CUDA provider first
            providers = self.settings.ONNX_PROVIDERS
            model_path = self.settings.MODEL_DIR / "lama_onnx" / "model.onnx"

            if model_path.exists():
                self.session = ort.InferenceSession(
                    str(model_path),
                    providers=providers,
                )
                self.use_onnx = True
                print(f"[Inpainter] Loaded LaMa ONNX on {providers[0]}")
            else:
                print("[Inpainter] LaMa ONNX not found, using OpenCV fallback")

        except Exception as e:
            print(f"[Inpainter] ONNX failed: {e}")
            print("[Inpainter] Using OpenCV fallback")

    async def inpaint_region(
        self,
        canvas: Image.Image,
        bbox: dict,
        method: str = "sample",
    ):
        """
        Inpaint (remove text) from a region.
        Methods: sample (bg color), fill (solid white), blur, lama (GPU).
        """
        if method == "lama" and self.use_onnx:
            await self._inpaint_lama(canvas, bbox)
        elif method == "sample":
            self._inpaint_sample(canvas, bbox)
        elif method == "blur":
            self._inpaint_blur(canvas, bbox)
        else:
            self._inpaint_fill(canvas, bbox)

    def _inpaint_sample(self, canvas: Image.Image, bbox: dict):
        """Sample background color from edges and fill."""
        img_array = np.array(canvas)
        bg_color = self._sample_bg_color(img_array, bbox)

        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(img_array, (x, y), (x + w, y + h), bg_color.tolist(), -1)

        # Update canvas in-place
        canvas.paste(Image.fromarray(img_array))

    def _inpaint_fill(self, canvas: Image.Image, bbox: dict):
        """Solid white fill."""
        img_array = np.array(canvas)
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 255, 255), -1)
        canvas.paste(Image.fromarray(img_array))

    def _inpaint_blur(self, canvas: Image.Image, bbox: dict):
        """Blur the region."""
        img_array = np.array(canvas)
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

        # Expand region slightly
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_array.shape[1], x + w + pad)
        y2 = min(img_array.shape[0], y + h + pad)

        roi = img_array[y1:y2, x1:x2]
        blurred = cv2.GaussianBlur(roi, (15, 15), 0)
        img_array[y1:y2, x1:x2] = blurred

        canvas.paste(Image.fromarray(img_array))

    async def _inpaint_lama(self, canvas: Image.Image, bbox: dict):
        """LaMa ONNX inpainting (GPU-accelerated)."""
        if not self.use_onnx or self.session is None:
            return self._inpaint_sample(canvas, bbox)

        try:
            img_array = np.array(canvas)
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]

            # Create mask
            mask = np.zeros(img_array.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # Prepare input for LaMa
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_resized = cv2.resize(img_rgb, (256, 256))
            mask_resized = cv2.resize(mask, (256, 256))

            # Normalize
            img_norm = img_resized.astype(np.float32) / 255.0
            mask_norm = mask_resized.astype(np.float32) / 255.0

            # Run inference
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.session.run(
                    None,
                    {
                        "image": img_norm.transpose(2, 0, 1)[np.newaxis],
                        "mask": mask_norm[np.newaxis, np.newaxis],
                    },
                ),
            )

            # Post-process
            output = result[0][0].transpose(1, 2, 0)
            output = (output * 255).clip(0, 255).astype(np.uint8)
            output = cv2.resize(output, (img_array.shape[1], img_array.shape[0]))

            # Apply only in masked region
            mask_bool = mask > 128
            img_array[mask_bool] = output[mask_bool]

            canvas.paste(Image.fromarray(img_array))

        except Exception as e:
            print(f"[Inpainter] LaMa failed: {e}, falling back to sample")
            self._inpaint_sample(canvas, bbox)

    def _sample_bg_color(self, img_array: np.ndarray, bbox: dict, margin: int = 8) -> np.ndarray:
        """Sample background color from edges of the region."""
        h_img, w_img = img_array.shape[:2]
        x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
        pixels = []

        # Sample from edges
        for i in range(0, w, max(1, w // 8)):
            for dy in [-margin, h + margin]:
                sy = y + dy
                if 0 <= sy < h_img:
                    sx = min(max(x + i, 0), w_img - 1)
                    pixels.append(img_array[sy, sx])

        for i in range(0, h, max(1, h // 8)):
            for dx in [-margin, w + margin]:
                sx = x + dx
                if 0 <= sx < w_img:
                    sy = min(max(y + i, 0), h_img - 1)
                    pixels.append(img_array[sy, sx])

        if not pixels:
            return np.array([255, 255, 255], dtype=np.uint8)

        pixels = np.array(pixels)
        return np.median(pixels, axis=0).astype(np.uint8)

    async def unload(self):
        """Release resources."""
        if self.session is not None:
            del self.session
            self.session = None
