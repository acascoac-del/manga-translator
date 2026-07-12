"""
Inpainting module - LaMa ONNX and OpenCV fallback.
Removes text from manga images and reconstructs the background.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("inpainting")

_ort_session = None
_use_onnx = False


def _resolve_model_path() -> str | None:
    """Resolve the location of the LaMa ONNX model."""
    search_paths = [
        # Environment variable
        os.path.join(os.environ.get("MODEL_DIR", ""), "inpainting", "lama.onnx"),
        os.path.join(os.environ.get("MODEL_DIR", ""), "lama.onnx"),
        # Docker volume path
        "/app/models/weights/inpainting/lama.onnx",
        # Local paths relative to project root
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "inpainting", "lama.onnx"),
        os.path.join("models", "inpainting", "lama.onnx"),
    ]
    for path in search_paths:
        if path and os.path.exists(path):
            log.info(f"Resolved inpainting model path: {path}")
            return os.path.abspath(path)
    return None


async def load_model(model_path: str = None, device: str = "cpu"):
    """Load LaMa ONNX model using onnxruntime. Falls back to OpenCV on failure."""
    global _ort_session, _use_onnx

    if _ort_session is not None:
        return

    path = model_path or _resolve_model_path()
    if not path:
        log.warning("lama.onnx model file not found. Falling back to OpenCV inpainting.")
        _use_onnx = False
        return

    def _init_session():
        global _ort_session, _use_onnx
        try:
            import onnxruntime as ort
            
            # Select execution providers
            # We force CPUExecutionProvider to prevent Reshape/shape memory corruption crashes on CUDA when sharing a 6GB VRAM GPU with Ollama/PyTorch
            providers = ['CPUExecutionProvider']
            log.info(f"Initializing ONNX session for inpainting with providers: {providers}")
            _ort_session = ort.InferenceSession(path, providers=providers)
            _use_onnx = True
            log.info(f"Loaded LaMa ONNX model successfully on: {_ort_session.get_providers()}")
        except Exception as e:
            log.error(f"Failed to load ONNX inpainting model: {e}. Falling back to OpenCV.")
            _ort_session = None
            _use_onnx = False

    await asyncio.get_event_loop().run_in_executor(None, _init_session)


async def inpaint(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    inpainting_size: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """Inpaint text regions. Uses ONNX LaMa if loaded, otherwise OpenCV."""
    if _use_onnx and _ort_session is not None:
        try:
            return await _inpaint_lama(img_rgb, mask)
        except Exception as e:
            log.error(f"LaMa inpainting failed: {e}. Falling back to OpenCV.")
            
    return inpaint_opencv(img_rgb, mask)


async def _inpaint_lama(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Run LaMa ONNX inpainting model."""
    global _ort_session
    loop = asyncio.get_event_loop()

    def _run():
        h, w = img_rgb.shape[:2]

        # 1. Preprocessing: resize image & mask to 512x512
        img_resized = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_AREA)
        img_float = img_resized.astype(np.float32) / 255.0
        # HWC to CHW
        img_chw = np.transpose(img_float, (2, 0, 1))
        # Add batch dim -> [1, 3, 512, 512]
        img_tensor = np.expand_dims(img_chw, axis=0)

        # Preprocess mask: resize, normalize, threshold to binary [0.0, 1.0]
        mask_resized = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        mask_float = (mask_resized > 128).astype(np.float32)
        # Add dimensions to make it [1, 1, 512, 512]
        mask_tensor = np.expand_dims(mask_float, axis=0)
        mask_tensor = np.expand_dims(mask_tensor, axis=0)

        # 2. ONNX Inference
        # Input names: 'image' and 'mask'
        outputs = _ort_session.run(None, {
            "image": img_tensor,
            "mask": mask_tensor
        })
        
        output = outputs[0]  # Shape [1, 3, 512, 512]
        
        # 3. Postprocessing
        # Squeeze batch -> [3, 512, 512]
        out_chw = output[0]
        # CHW to HWC -> [512, 512, 3]
        out_hwc = np.transpose(out_chw, (1, 2, 0))
        # Denormalize
        out_rgb_512 = (out_hwc * 255.0).clip(0, 255).astype(np.uint8)

        # Resize back to original size
        out_rgb_orig = cv2.resize(out_rgb_512, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply mask blending (only replace where mask was active)
        result = img_rgb.copy()
        mask_bool = mask > 128
        result[mask_bool] = out_rgb_orig[mask_bool]

        return result

    return await loop.run_in_executor(None, _run)


def inpaint_opencv(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Improved OpenCV inpainting (blends TELEA + NS for best quality)."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Dilate mask to cover text edges completely
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask, kernel, iterations=3)

    # Use both methods and blend
    r1 = cv2.inpaint(img_bgr, mask_dilated, 9, cv2.INPAINT_TELEA)
    r2 = cv2.inpaint(img_bgr, mask_dilated, 9, cv2.INPAINT_NS)

    # Weighted blend
    result = cv2.addWeighted(r1, 0.6, r2, 0.4, 0)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def inpaint_solid(img_rgb: np.ndarray, mask: np.ndarray, color: tuple = (255, 255, 255)) -> np.ndarray:
    """Simple solid color fill."""
    result = img_rgb.copy()
    result[mask > 128] = color
    return result


def unload():
    global _ort_session, _use_onnx
    _ort_session = None
    _use_onnx = False
    log.info("LaMa ONNX model unloaded.")
