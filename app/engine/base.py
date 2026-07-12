"""
Base classes and context for the manga translation pipeline.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from typing import Optional, Any

import numpy as np
from PIL import Image

log = logging.getLogger("engine")


@dataclass
class TextRegion:
    """Detected text region with bounding box and polygon."""
    index: int
    bbox: list[int]  # [x1, y1, x2, y2]
    polygon: list[list[int]]  # [[x,y], ...]
    source_text: str = ""
    translated_text: str = ""
    confidence: float = 0.0
    orientation: str = "horizontal"  # horizontal | vertical
    fg_color: tuple = (0, 0, 0)
    bg_color: tuple = (255, 255, 255)
    font_size: int = 0


@dataclass
class PipelineContext:
    """Context object passed through the pipeline stages."""
    input_image: Image.Image = None
    img_rgb: np.ndarray = None
    img_alpha: np.ndarray = None
    text_regions: list[TextRegion] = field(default_factory=list)
    mask: np.ndarray = None
    img_inpainted: np.ndarray = None
    result: Image.Image = None
    verbose: bool = False
    device: str = "cpu"
    metadata: dict = field(default_factory=dict)


def pil_to_b64(img: Image.Image, fmt: str = "PNG", quality: int = 90) -> str:
    buf = io.BytesIO()
    if fmt.upper() in ("JPEG", "JPG"):
        # JPEG doesn't support alpha - convert RGBA to RGB
        if img.mode == "RGBA":
            img = img.convert("RGB")
        img.save(buf, format=fmt, quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def load_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Convert PIL Image to RGB numpy array and extract alpha channel."""
    img = np.array(image.convert("RGBA"))
    img_rgb = img[:, :, :3].copy()
    img_alpha = img[:, :, 3].copy() if img.shape[2] == 4 else None
    return img_rgb, img_alpha


def dump_image(input_image: Image.Image, img_rgb: np.ndarray, img_alpha: np.ndarray) -> Image.Image:
    """Convert numpy array back to PIL Image, preserving alpha if present."""
    if img_alpha is not None:
        rgba = np.dstack((img_rgb, img_alpha))
        return Image.fromarray(rgba, "RGBA")
    return Image.fromarray(img_rgb, "RGB")
