"""
Text Detection Module - GPU-optimized.
Uses YOLO (comic-text-detector) for fast manga text detection.
Model stays in GPU memory for zero-copy inference.
"""

import asyncio
from typing import Optional
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import Settings


class TextDetector:
    """
    GPU-accelerated text detection using YOLO.
    Model is loaded once and persists in GPU memory.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.device = settings.CUDA_DEVICE

    async def load(self):
        """Load YOLO model into GPU memory."""
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download

            # Download model if not cached
            cache_dir = self.settings.CACHE_DIR / "detection"
            cache_dir.mkdir(parents=True, exist_ok=True)

            model_path = hf_hub_download(
                repo_id=self.settings.DETECTION_MODEL,
                filename="comic-text-detector.pt",
                cache_dir=str(cache_dir),
            )

            self.model = YOLO(model_path)
            # Move to GPU
            self.model.to(self.device)
            print(f"[Detector] Loaded {self.settings.DETECTION_MODEL} on {self.device}")

        except Exception as e:
            print(f"[Detector] Failed to load YOLO: {e}")
            print("[Detector] Falling back to contour-based detection")
            self.model = None

    async def detect(self, image: Image.Image) -> list[dict]:
        """
        Detect text regions in image.
        Returns list of { x, y, width, height, confidence }.
        """
        if self.model is not None:
            return await self._detect_yolo(image)
        return await self._detect_contour(image)

    async def _detect_yolo(self, image: Image.Image) -> list[dict]:
        """YOLO-based detection (GPU-accelerated)."""
        img_array = np.array(image)

        # Run inference (non-blocking via thread)
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: self.model.predict(
                img_array,
                conf=self.settings.DETECTION_CONF_THRESHOLD,
                iou=self.settings.DETECTION_IOU_THRESHOLD,
                device=self.device,
                verbose=False,
            ),
        )

        regions = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    regions.append({
                        "x": int(x1),
                        "y": int(y1),
                        "width": int(x2 - x1),
                        "height": int(y2 - y1),
                        "confidence": conf,
                        "orientation": "vertical" if (y2 - y1) > (x2 - x1) * 2 else "horizontal",
                    })

        return regions

    async def _detect_contour(self, image: Image.Image) -> list[dict]:
        """Fallback: contour-based detection (CPU, no model needed)."""
        img_array = np.array(image.convert("L"))

        # Adaptive thresholding
        binary = (img_array < 128).astype(np.uint8)

        # Connected components
        from scipy import ndimage
        labeled, num_features = ndimage.label(binary)

        regions = []
        for i in range(1, num_features + 1):
            component = (labeled == i)
            y_coords, x_coords = np.where(component)

            if len(x_coords) < 30:
                continue

            x_min, x_max = int(x_coords.min()), int(x_coords.max())
            y_min, y_max = int(y_coords.min()), int(y_coords.max())
            w = x_max - x_min
            h = y_max - y_min

            if w >= 10 and h >= 10 and w * h >= 100:
                regions.append({
                    "x": x_min,
                    "y": y_min,
                    "width": w,
                    "height": h,
                    "confidence": 0.5,
                    "orientation": "vertical" if h > w * 2 else "horizontal",
                })

        return regions
