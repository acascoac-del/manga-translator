"""
Manga Translation Pipeline - Uses manga-image-translator engine.
Same core as MangaLingo: Detect → OCR → Translate → Inpaint → Render.

Components:
  - Detection: CTD (Comic Text Detector) - best for manga
  - OCR: manga-ocr - specialized for Japanese manga text
  - Translation: Google Translate (free, online)
  - Inpainting: LaMa (large) - GPU-accelerated
  - Renderer: manga2eng - English/typesetting renderer
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

from PIL import Image

# manga-image-translator imports
from manga_translator import Config
from manga_translator.config import (
    Detector, Ocr, Translator, Inpainter, Renderer as MTRenderer,
)
from manga_translator.manga_translator import MangaTranslator
from manga_translator.utils import Context

log = logging.getLogger("manga-pipeline")
log.setLevel(logging.INFO)

# =============================================================================
# Component Maps
# =============================================================================

DETECTOR_MAP = {
    "default": Detector.default,
    "dbconvnext": Detector.dbconvnext,
    "ctd": Detector.ctd,
    "craft": Detector.craft,
    "none": Detector.none,
}

OCR_MAP = {
    "48px": Ocr.ocr48px,
    "32px": Ocr.ocr32px,
    "48px_ctc": Ocr.ocr48px_ctc,
    "manga_ocr": Ocr.mocr,
    "mocr": Ocr.mocr,
}

TRANSLATOR_MAP = {
    "google": Translator.google,
    "none": Translator.none,
}

INPAINTER_MAP = {
    "default": Inpainter.default,
    "lama_large": Inpainter.lama_large,
    "lama_mpe": Inpainter.lama_mpe,
    "lama": Inpainter.lama_large,
    "sd": Inpainter.sd,
    "none": Inpainter.none,
    "original": Inpainter.original,
    "solid": Inpainter.solid,
}

RENDERER_MAP = {
    "default": MTRenderer.default,
    "manga2eng": MTRenderer.manga2Eng,
    "manga2eng_pillow": MTRenderer.manga2EngPillow,
    "pillow": MTRenderer.manga2EngPillow,
    "none": MTRenderer.none,
}

LANG_MAP = {
    "auto": "auto",
    "es": "ESP", "en": "ENG", "fr": "FRA", "de": "DEU",
    "it": "ITA", "pt": "PTB", "pt-BR": "PTB", "ru": "RUS",
    "ja": "JPN", "ko": "KOR", "zh": "CHS", "zh-CN": "CHS", "zh-TW": "CHT",
    "ar": "ARA", "nl": "NLD", "pl": "POL", "tr": "TRK",
    "id": "IND", "vi": "VIN", "th": "THA", "hi": "HIN",
}

FONT_FILES = {
    "comic": "comic shanns 2.ttf",
    "anime_ace": "anime_ace.ttf",
    "anime_ace_3": "anime_ace_3.ttf",
    "msyh": "msyh.ttc",
    "msgothic": "msgothic.ttc",
    "arial": "Arial-Unicode-Regular.ttf",
}

# =============================================================================
# Device Detection
# =============================================================================

def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            log.info(f"GPU detectada: {gpu_name} ({vram:.1f} GB VRAM)")
            return "cuda"
        log.warning("CUDA no disponible. Usando CPU.")
        return "cpu"
    except Exception as e:
        log.warning(f"GPU no detectada: {e}. CPU.")
        return "cpu"


DEVICE = _detect_device()

# =============================================================================
# Pipeline Configuration
# =============================================================================

MODELS_TTL = int(os.getenv("MODELS_TTL", "600"))
VRAM_CLEANUP_EVERY = int(os.getenv("VRAM_CLEANUP_EVERY", "8"))
PIPELINE_CONCURRENCY = int(os.getenv("PIPELINE_CONCURRENCY", "20"))

_pipeline_sem = asyncio.Semaphore(PIPELINE_CONCURRENCY)
_cleanup_counter = 0
_cleanup_lock = asyncio.Lock()

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Region:
    index: int
    bbox: list[int]
    polygon: list[list[int]]
    source_text: str = ""
    translated_text: str = ""
    confidence: float = 0.0
    inpainted: bool = False
    rendered: bool = False


@dataclass
class PipelineResult:
    success: bool
    original_b64: str = ""
    translated_b64: str = ""
    regions: list[dict] = field(default_factory=list)
    processing_time_ms: int = 0
    stages: dict = field(default_factory=dict)
    error: Optional[str] = None
    backend_used: dict = field(default_factory=dict)
    cache_hit: bool = False


# =============================================================================
# Utilities
# =============================================================================

def _pil_to_b64(img, fmt="PNG", quality=90):
    buf = io.BytesIO()
    save_kwargs = {"format": fmt}
    if fmt.upper() in ("JPEG", "JPG"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
        if img.mode == "RGBA":
            img = img.convert("RGB")
    img.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _b64_to_pil(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _resolve_font_path(family):
    fname = FONT_FILES.get(family) or FONT_FILES["anime_ace_3"]
    for p in [
        os.path.join(os.path.dirname(__file__), "fonts", fname),
        os.path.join(os.path.dirname(__file__), "manga_translator", "fonts", fname),
    ]:
        if os.path.exists(p):
            return p
    return None


# =============================================================================
# MangaTranslator Singleton
# =============================================================================

_mt_instance = None
_mt_lock = asyncio.Lock()


async def _get_mt():
    global _mt_instance
    if _mt_instance is None:
        async with _mt_lock:
            if _mt_instance is None:
                log.info(f"Inicializando MangaTranslator (device={DEVICE}, ttl={MODELS_TTL}s)...")
                _mt_instance = MangaTranslator({
                    "device": DEVICE,
                    "verbose": False,
                    "ignore_errors": False,
                    "models_ttl": MODELS_TTL,
                    "batch_size": 4,
                    "kernel_size": 3,
                    "mask_dilation_offset": 20,
                    "font_path": None,
                })
                log.info("MangaTranslator listo.")
    return _mt_instance


# =============================================================================
# Config Builder
# =============================================================================

def _build_config(
    target_lang, source_lang, detector, ocr, translator,
    inpainter, renderer, font_family, font_size,
):
    tgt = LANG_MAP.get(target_lang, "ESP")
    tra_enum = TRANSLATOR_MAP.get(translator, Translator.google)

    cfg = Config()
    cfg.detector.detector = DETECTOR_MAP.get(detector, Detector.ctd)
    cfg.ocr.ocr = OCR_MAP.get(ocr, Ocr.mocr)
    cfg.translator.translator = tra_enum
    cfg.translator.target_lang = tgt
    cfg.inpainter.inpainter = INPAINTER_MAP.get(inpainter, Inpainter.lama_large)
    cfg.render.renderer = RENDERER_MAP.get(renderer, MTRenderer.manga2EngPillow)

    # Optimizations
    cfg.detector.detection_size = 768
    cfg.inpainter.inpainting_size = 640

    if DEVICE == "cuda":
        try:
            from manga_translator.config import InpaintPrecision
            cfg.inpainter.inpainting_precision = InpaintPrecision.fp16
        except Exception:
            pass

    cfg.translator.enable_post_translation_check = False
    cfg.translator.no_text_lang_skip = True

    font_path = _resolve_font_path(font_family)
    if font_path:
        cfg._font_path = font_path
    if font_size and font_size > 0:
        cfg.render.font_size = font_size

    return cfg


# =============================================================================
# VRAM Cleanup
# =============================================================================

async def _maybe_cleanup_vram(force: bool = False):
    global _cleanup_counter
    if DEVICE != "cuda":
        return
    async with _cleanup_lock:
        _cleanup_counter += 1
        if force or _cleanup_counter >= VRAM_CLEANUP_EVERY:
            _cleanup_counter = 0
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass


# =============================================================================
# Pipeline Execution
# =============================================================================

async def _run_pipeline_inner(
    image_b64, target_lang="es", source_lang="auto",
    detector="ctd", ocr="manga_ocr", translator="google",
    inpainter="lama", renderer="manga2eng",
    font_family="anime_ace_3", font_size=0,
):
    start = time.time()
    backend_used = {
        "detector": detector, "ocr": ocr, "translator": translator,
        "inpainter": inpainter, "renderer": renderer,
        "target_lang": target_lang, "source_lang": source_lang,
        "device": DEVICE, "pipeline_version": "2.0.0",
    }

    try:
        pil_img = _b64_to_pil(image_b64).convert("RGB")
        original_b64 = _pil_to_b64(pil_img, "JPEG", quality=85)

        cfg = _build_config(
            target_lang, source_lang, detector, ocr, translator,
            inpainter, renderer, font_family, font_size,
        )
        mt = await _get_mt()

        log.info(
            f"Pipeline: det={detector} ocr={ocr} tra={translator}"
            f" inp={inpainter} tgt={target_lang} dev={DEVICE}"
        )

        ctx: Context = await mt.translate(pil_img, cfg)
        if ctx.result is None:
            raise RuntimeError("Pipeline produjo una imagen de resultado nula")

        translated_b64 = _pil_to_b64(ctx.result, "JPEG", quality=90)

        regions_out = []
        if hasattr(ctx, "text_regions") and ctx.text_regions:
            for i, tb in enumerate(ctx.text_regions):
                xyxy = tb.xyxy if hasattr(tb, "xyxy") else [0, 0, 0, 0]
                bbox = [int(v) for v in xyxy]
                poly = []
                if hasattr(tb, "polygon") and tb.polygon is not None:
                    try:
                        poly = [[int(p[0]), int(p[1])] for p in tb.polygon]
                    except Exception:
                        pass
                src = (tb.text or "").strip() if hasattr(tb, "text") else ""
                tgt = (tb.translation or "").strip() if hasattr(tb, "translation") else ""
                regions_out.append({
                    "index": i, "bbox": bbox, "polygon": poly,
                    "source_text": src, "translated_text": tgt,
                    "confidence": float(getattr(tb, "prob", 0.0) or 0.0),
                    "inpainted": True, "rendered": True,
                })

        await _maybe_cleanup_vram()

        total = int((time.time() - start) * 1000)
        log.info(f"Pipeline OK en {total}ms (device={DEVICE}, translator={translator})")

        return PipelineResult(
            success=True, original_b64=original_b64,
            translated_b64=translated_b64, regions=regions_out,
            processing_time_ms=total, backend_used=backend_used,
        )

    except Exception as exc:
        log.error(f"Pipeline falló: {exc}\n{traceback.format_exc()}")
        total = int((time.time() - start) * 1000)
        return PipelineResult(
            success=False, processing_time_ms=total,
            error=f"{type(exc).__name__}: {exc}", backend_used=backend_used,
        )


async def run_pipeline(
    image_b64, target_lang="es", source_lang="auto",
    detector="ctd", ocr="manga_ocr", translator="google",
    inpainter="lama", renderer="manga2eng",
    font_family="anime_ace_3", font_size=0,
):
    async with _pipeline_sem:
        return await _run_pipeline_inner(
            image_b64, target_lang, source_lang, detector, ocr, translator,
            inpainter, renderer, font_family, font_size,
        )


# =============================================================================
# Public API
# =============================================================================

async def initialize():
    """Pre-load models on startup."""
    log.info("Pre-cargando modelos...")
    await _get_mt()
    log.info("Modelos cargados.")


async def cleanup():
    """Release resources."""
    global _mt_instance
    if _mt_instance is not None:
        del _mt_instance
        _mt_instance = None
    await _maybe_cleanup_vram(force=True)
    gc.collect()
