"""
Pipeline Orchestrator - Ties all modules together.
Detect → OCR → Translate → Inpaint → Render
"""

from __future__ import annotations

import asyncio
import base64
import gc
import io
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .base import (
    PipelineContext, TextRegion,
    pil_to_b64, b64_to_pil, load_image, dump_image,
)
from .detection import ctd as detection
from .ocr import manga_ocr as ocr
from .translation import dispatcher as translation
from .inpainting import lama as inpainting
from .rendering import manga2eng as rendering

log = logging.getLogger("pipeline")

# =============================================================================
# Configuration
# =============================================================================

MODELS_DIR = os.environ.get("MODEL_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "models"))
FONTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "fonts")

DEVICE = "cpu"
try:
    import torch
    if torch.cuda.is_available():
        DEVICE = "cuda"
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception:
    pass


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    detector: str = "ctd"
    ocr: str = "manga_ocr"
    translator: str = "google"
    inpainter: str = "lama"
    renderer: str = "manga2eng"
    source_lang: str = "auto"
    target_lang: str = "es"
    font_family: str = "DejaVuSans"  # Clear readable font with full Spanish support
    font_size: int = 0
    font_color: tuple = (0, 0, 0)
    text_shadow: bool = False
    detection_size: int = 1024
    inpainting_size: int = 640
    text_threshold: float = 0.5
    box_threshold: float = 0.5
    device: str = DEVICE
    ollama_host: str | None = None
    ollama_model: str | None = None
    libretranslate_host: str | None = None


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    success: bool
    original_b64: str = ""
    translated_b64: str = ""
    regions: list[dict] = field(default_factory=list)
    processing_time_ms: int = 0
    error: Optional[str] = None
    backend_used: dict = field(default_factory=dict)


# =============================================================================
# Model Loading
# =============================================================================

_models_loaded = False
_load_lock = asyncio.Lock()


async def ensure_models(config: PipelineConfig):
    """Load all models if not already loaded."""
    global _models_loaded
    if _models_loaded:
        return

    async with _load_lock:
        if _models_loaded:
            return

        log.info("Loading all models...")

        # Detection model (uses OpenCV if CTD not available)
        await detection.load_model(device=config.device)

        # OCR model
        await ocr.load_model(config.device)

        # Inpainting model (OpenCV)
        await inpainting.load_model(device=config.device)

        _models_loaded = True
        log.info("All models loaded.")


# =============================================================================
# Pipeline Execution
# =============================================================================

async def run_pipeline(
    image_b64: str,
    config: PipelineConfig = None,
) -> PipelineResult:
    """
    Execute the full translation pipeline.
    Detect → OCR → Translate → Inpaint → Render
    """
    if config is None:
        config = PipelineConfig()

    start = time.time()
    backend_used = {
        "detector": config.detector,
        "ocr": config.ocr,
        "translator": config.translator,
        "inpainter": config.inpainter,
        "renderer": config.renderer,
        "device": config.device,
    }

    try:
        # Ensure models are loaded
        await ensure_models(config)

        # Decode image
        pil_img = b64_to_pil(image_b64).convert("RGB")
        original_b64 = pil_to_b64(pil_img, "JPEG", quality=85)
        img_rgb, img_alpha = load_image(pil_img)

        ctx = PipelineContext(
            input_image=pil_img,
            img_rgb=img_rgb,
            img_alpha=img_alpha,
            device=config.device,
        )

        # ── Step 1: Detection ──────────────────────────────────────────
        t1 = time.time()
        log.info("Step 1: Detecting text regions...")
        textlines, mask_raw, mask = await detection.detect(
            img_rgb,
            detection_size=config.detection_size,
            text_threshold=config.text_threshold,
            box_threshold=config.box_threshold,
            device=config.device,
        )
        ctx.mask = mask
        log.info(f"  Detection: {len(textlines)} regions in {(time.time()-t1)*1000:.0f}ms")

        if not textlines:
            return PipelineResult(
                success=True,
                original_b64=original_b64,
                translated_b64=original_b64,
                regions=[],
                processing_time_ms=int((time.time() - start) * 1000),
                backend_used=backend_used,
            )

        # ── Step 2: OCR ────────────────────────────────────────────────
        t2 = time.time()
        log.info("Step 2: Running OCR...")
        textlines = await ocr.recognize(img_rgb, textlines, config.device)
        log.info(f"  OCR: done in {(time.time()-t2)*1000:.0f}ms")

        # Filter out empty text
        textlines = [tl for tl in textlines if tl.get("text", "").strip()]
        if not textlines:
            return PipelineResult(
                success=True,
                original_b64=original_b64,
                translated_b64=original_b64,
                regions=[],
                processing_time_ms=int((time.time() - start) * 1000),
                backend_used=backend_used,
            )

        # ── Step 3: Translation ────────────────────────────────────────
        t3 = time.time()
        log.info("Step 3: Translating...")
        textlines = await translation.translate_regions(
            textlines,
            source_lang=config.source_lang,
            target_lang=config.target_lang,
            translator=config.translator,
            ollama_host=config.ollama_host,
            ollama_model=config.ollama_model,
            libretranslate_host=config.libretranslate_host,
        )
        log.info(f"  Translation: done in {(time.time()-t3)*1000:.0f}ms")

        # ── Step 4: Inpainting ─────────────────────────────────────────
        t4 = time.time()
        log.info("Step 4: Inpainting...")
        if config.inpainter == "solid":
            img_inpainted = await inpainting.inpaint_solid(img_rgb, mask)
        else:
            img_inpainted = await inpainting.inpaint(img_rgb, mask, config.inpainting_size, config.device)
        ctx.img_inpainted = img_inpainted
        log.info(f"  Inpainting: done in {(time.time()-t4)*1000:.0f}ms")

        # ── Step 5: Rendering ──────────────────────────────────────────
        t5 = time.time()
        log.info("Step 5: Rendering text...")
        font_path = f"{config.font_family}.ttf"
        img_rendered = rendering.render_text_regions(
            img_inpainted,
            textlines,
            font_path=font_path,
            font_size=config.font_size,
            font_color=config.font_color,
            text_shadow=config.text_shadow,
        )
        log.info(f"  Rendering: done in {(time.time()-t5)*1000:.0f}ms")

        # Convert result to base64
        result_img = dump_image(pil_img, img_rendered, img_alpha)
        translated_b64 = pil_to_b64(result_img, "JPEG", quality=90)

        # Build regions output
        regions_out = []
        for i, tl in enumerate(textlines):
            regions_out.append({
                "index": i,
                "bbox": tl.get("bbox", []),
                "polygon": tl.get("polygon", []),
                "source_text": tl.get("text", ""),
                "translated_text": tl.get("translated_text", ""),
                "confidence": tl.get("confidence", 0.0),
            })

        total_ms = int((time.time() - start) * 1000)
        log.info(f"Pipeline complete: {total_ms}ms ({len(regions_out)} regions)")

        return PipelineResult(
            success=True,
            original_b64=original_b64,
            translated_b64=translated_b64,
            regions=regions_out,
            processing_time_ms=total_ms,
            backend_used=backend_used,
        )

    except Exception as exc:
        log.error(f"Pipeline failed: {exc}\n{traceback.format_exc()}")
        return PipelineResult(
            success=False,
            processing_time_ms=int((time.time() - start) * 1000),
            error=f"{type(exc).__name__}: {exc}",
            backend_used=backend_used,
        )


# =============================================================================
# Lifecycle
# =============================================================================

async def initialize(config: PipelineConfig = None):
    """Pre-load all models."""
    if config is None:
        config = PipelineConfig()
    await ensure_models(config)


async def cleanup():
    """Release all resources."""
    detection.unload()
    ocr.unload()
    inpainting.unload()
    rendering.clear_font_cache()
    await translation.cleanup()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    log.info("Pipeline cleaned up.")
