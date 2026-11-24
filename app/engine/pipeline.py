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
import re
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
from .ocr import dispatcher as ocr
from .ocr.easy_ocr import _normalize_english_ocr
from .translation import dispatcher as translation
from .translation.quality import needs_google_fallback
from .inpainting import lama as inpainting
from .rendering import manga2eng as rendering

log = logging.getLogger("pipeline")


def _contains_cjk(text: str) -> bool:
    """Return True when OCR output contains a Japanese/CJK character."""
    return any(
        "\u3040" <= char <= "\u30ff"  # hiragana/katakana
        or "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uac00" <= char <= "\ud7af"
        for char in text
    )


def _is_valid_ocr_text(text: str, source_lang: str) -> bool:
    """Reject common manga-OCR hallucinations from blank detector boxes.

    manga-ocr is generative and may return a plausible Latin word for an empty
    crop.  For the Japanese/automatic route, a result without any CJK
    character cannot be Japanese dialogue and must not reach translation.
    """
    text = " ".join((text or "").split())
    if not text or not any(char.isalnum() for char in text):
        return False

    source = (source_lang or "auto").lower().strip()
    if source in {"ja", "jpn", "jpn_jpan", "japanese"}:
        native = [
            char for char in text
            if "\u3040" <= char <= "\u30ff" or "\u4e00" <= char <= "\u9fff"
        ]
        if not native:
            return False
        # Manga detectors often split a vertical Japanese sentence into
        # several one-character crops. Dropping lone kana here erased pieces
        # such as "は…？", "ル" or "ん" before translation and left visibly
        # blank balloons. Keep them so the renderer can group the fragments.
        alnum_count = sum(char.isalnum() for char in text)
        if alnum_count >= 5 and len(native) / alnum_count < 0.35:
            return False
        return True
    if source in {"ko", "kor", "korean"}:
        return any("\uac00" <= char <= "\ud7af" for char in text)
    if source in {"zh", "zh-cn", "ch_sim", "chinese"}:
        return any("\u4e00" <= char <= "\u9fff" for char in text)
    if source in {"en", "eng", "english"}:
        words = re.findall(r"[A-Za-z]+", text)
        letters = sum(len(word) for word in words)
        if letters >= 3:
            return True
        # Keep valid one-letter English dialogue, but reject detector noise
        # such as isolated digits, symbols, and arbitrary single capitals.
        short_words = {
            "a", "i", "am", "as", "at", "be", "by", "do", "go", "he",
            "if", "in", "is", "it", "me", "my", "no", "of", "ok", "on",
            "or", "so", "to", "up", "we",
        }
        return len(words) == 1 and words[0].lower() in short_words
    return True


def _flatten_uniform_english_regions(
    img_rgb: np.ndarray, textlines: list[dict]
) -> np.ndarray:
    """Remove residual source glyphs inside exact English OCR line boxes.

    Cleaning the whole CTD rectangle can cover artwork, while LaMa may restore
    letters when only its generated mask is used.  EasyOCR's quadrilaterals
    are tight around each actual line, so flatten only those small areas using
    the dominant light/dark color immediately around the line.
    """
    result = img_rgb.copy()
    height, width = result.shape[:2]
    for tl in textlines:
        polygons = tl.get("ocr_polygons", [])
        if not polygons and tl.get("ocr_recovered"):
            polygons = [tl.get("polygon", [])]
        for polygon in polygons:
            points = np.asarray(polygon, dtype=np.int32)
            if points.size < 8:
                continue
            x, y, bw, bh = cv2.boundingRect(points)
            if bw <= 1 or bh <= 1:
                continue
            outer = max(5, int(min(bw, bh) * 0.22))
            ox1, oy1 = max(0, x - outer), max(0, y - outer)
            ox2, oy2 = min(width, x + bw + outer), min(height, y + bh + outer)
            neighborhood = result[oy1:oy2, ox1:ox2]
            if neighborhood.size == 0:
                continue
            ring_mask = np.ones(neighborhood.shape[:2], dtype=bool)
            ring_mask[y - oy1:y + bh - oy1, x - ox1:x + bw - ox1] = False
            ring_pixels = neighborhood[ring_mask]
            if ring_pixels.size == 0:
                continue
            gray = cv2.cvtColor(
                ring_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY
            ).reshape(-1)
            light_fraction = float(np.mean(gray >= 175))
            dark_fraction = float(np.mean(gray <= 80))
            if max(light_fraction, dark_fraction) < 0.42:
                continue
            candidates = (
                ring_pixels[gray >= 175]
                if light_fraction >= dark_fraction
                else ring_pixels[gray <= 80]
            )
            if candidates.size == 0:
                continue
            candidate_gray = cv2.cvtColor(
                candidates.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY
            ).reshape(-1)
            # A gradient/illustration may contain many light or dark pixels,
            # but unlike a bubble background it does not have one stable tone.
            if float(np.percentile(candidate_gray, 90) - np.percentile(candidate_gray, 10)) > 28:
                continue
            color = tuple(int(value) for value in np.median(candidates, axis=0))
            pad = max(2, int(min(bw, bh) * 0.08))
            cv2.rectangle(
                result,
                (max(0, x - pad), max(0, y - pad)),
                (min(width - 1, x + bw + pad), min(height - 1, y + bh + pad)),
                color,
                -1,
            )
    return result


def _detect_source_language(textlines: list[dict], requested: str) -> str:
    """Infer one source language for a page when the user selects auto."""
    requested = (requested or "auto").lower().strip()
    if requested != "auto":
        return requested

    detected = [
        line.get("detected_source_lang") for line in textlines
        if line.get("detected_source_lang")
    ]
    if detected:
        return max(set(detected), key=detected.count)

    text = " ".join(t.get("text", "") for t in textlines)
    if any("\uac00" <= c <= "\ud7af" for c in text):
        return "ko"
    if any("\u3040" <= c <= "\u30ff" for c in text):
        return "ja"
    if any("\u4e00" <= c <= "\u9fff" for c in text):
        # Without kana, Han-only text is more likely to be Chinese.
        return "zh"
    if any(c.isalpha() for c in text):
        return "en"
    return "en"

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
    ocr: str = "auto"
    translator: str = "llamacpp"
    inpainter: str = "lama"
    renderer: str = "manga2eng"
    source_lang: str = "auto"
    target_lang: str = "es"
    font_family: str = "BarlowSemiCondensed-Medium"
    font_size: int = 0
    font_color: tuple = (0, 0, 0)
    text_shadow: bool = False
    detection_size: int = 1024
    inpainting_size: int = 640
    text_threshold: float = 0.25
    box_threshold: float = 0.25
    device: str = DEVICE
    lmstudio_host: str | None = None
    lmstudio_model: str | None = None
    groq_api_key: str | None = None
    groq_model: str | None = None


@dataclass
class PipelineResult:
    """Pipeline execution result."""
    success: bool
    original_b64: str = ""
    translated_b64: str = ""
    regions: list[dict] = field(default_factory=list)
    processing_time_ms: int = 0
    detected_source_lang: str = "auto"
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
        await ocr.load_models(config.device, config.source_lang)

        # Inpainting model (OpenCV)
        await inpainting.load_model(device=config.device)

        _models_loaded = True
        log.info("Core image models loaded (local translators load on demand).")


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
        # A previous local-translation session may have left llama.cpp on the
        # GPU. Release it before detection/OCR when another engine is selected;
        # doing this later in the translation dispatcher is too late to help
        # the expensive image stages of the first page.
        translator_name = config.translator.lower().strip()
        # A fallback can load llama.cpp on demand. Keeping it resident while a
        # network translator is selected steals RAM/VRAM from OCR and makes
        # every page slower even when the fallback is never needed.
        if translator_name != "llamacpp":
            from .translation import llamacpp
            await llamacpp.release_if_loaded()

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
        
        # CTD can miss narrow vertical Japanese columns. The OpenCV pass adds
        # recovery candidates; script-aware OCR validation removes artwork and
        # other false positives before inpainting/rendering.
        merge_opencv = True
        
        textlines, mask_raw_onnx, mask_onnx = await detection.detect(
            img_rgb,
            detection_size=config.detection_size,
            text_threshold=config.text_threshold,
            box_threshold=config.box_threshold,
            device=config.device,
            merge_opencv=merge_opencv
        )
        if not textlines:
            # Some pages use faint or unusually small lettering. A second,
            # more sensitive pass is cheaper than returning the whole page
            # untouched, and only runs when the normal detector found nothing.
            retry_text_threshold = min(config.text_threshold, 0.12)
            retry_box_threshold = min(config.box_threshold, 0.12)
            log.warning(
                "No text regions found; retrying detection at thresholds %.2f/%.2f",
                retry_text_threshold,
                retry_box_threshold,
            )
            textlines, mask_raw_onnx, mask_onnx = await detection.detect(
                img_rgb,
                detection_size=config.detection_size,
                text_threshold=retry_text_threshold,
                box_threshold=retry_box_threshold,
                device=config.device,
                merge_opencv=merge_opencv,
            )
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
        log.info("Step 2: OCR...")
        textlines = await ocr.recognize_regions(img_rgb, textlines, config.source_lang, config.device)
        # manga-ocr can hallucinate short Latin words (e.g. "que") on blank
        # crops.  Never translate or erase a region rejected by source-aware
        # OCR validation.
        before_ocr_filter = len(textlines)
        rejected_textlines = [
            t for t in textlines
            if not _is_valid_ocr_text(
                t.get("text", ""),
                t.get("detected_source_lang", config.source_lang),
            )
        ]
        textlines = [
            t for t in textlines
            if _is_valid_ocr_text(
                t.get("text", ""),
                t.get("detected_source_lang", config.source_lang),
            )
        ]
        detected_source_lang = _detect_source_language(textlines, config.source_lang)
        # Normalize English once more at the pipeline boundary. Automatic OCR
        # copies staged candidates back into detector regions, and this final
        # pass guarantees recovered/tiny lines receive the same repairs before
        # any translator sees them. Other language routes are untouched.
        if detected_source_lang == "en":
            for tl in textlines:
                tl["text"] = _normalize_english_ocr(tl.get("text", ""))
        log.info(
            f"  Source language: requested={config.source_lang}, "
            f"detected={detected_source_lang}"
        )
        log.info(
            f"  OCR validation: kept {len(textlines)}/{before_ocr_filter} "
            "regions with plausible source text"
        )
        if rejected_textlines:
            log.info(
                "  OCR rejected samples: %s",
                [
                    {
                        "bbox": region.get("bbox"),
                        "text": region.get("text", "")[:80],
                        "lang": region.get("detected_source_lang"),
                    }
                    for region in rejected_textlines[:8]
                ],
            )
        log.info(f"  OCR: {len(textlines)} valid regions found in {(time.time()-t2)*1000:.0f}ms")
        
        if not textlines:
            return PipelineResult(
                success=True,
                original_b64=original_b64,
                translated_b64=original_b64,
                regions=[],
                processing_time_ms=int((time.time() - start) * 1000),
                backend_used=backend_used,
            )

        # Build precise mask from validated textlines ONLY
        mask_raw = np.zeros_like(mask_onnx)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        region_masks: dict[tuple[int, int, int, int], np.ndarray] = {}
        for tl in textlines:
            region_mask_raw = np.zeros_like(mask_onnx)
            has_ocr_polygon = False
            # Mask the exact EasyOCR lines too. CTD masks can be narrower than
            # outlined scanlation glyphs, leaving English under the Spanish.
            for ocr_polygon in tl.get("ocr_polygons", []):
                pts = np.array(ocr_polygon, np.int32)
                if pts.size >= 8:
                    cv2.fillPoly(region_mask_raw, [pts], 255)
                    has_ocr_polygon = True
            if detected_source_lang == "en":
                # Scanlation fonts frequently extend beyond CTD's glyph mask.
                # Every region reaching this point already passed strict
                # English validation, so clearing its tight OCR/CTD rectangle
                # is safe and guarantees that untranslated letters cannot
                # remain underneath the Spanish rendering. Other languages
                # deliberately keep their existing precise-mask behavior.
                x1, y1, x2, y2 = tl["bbox"]
                h_mask, w_mask = region_mask_raw.shape[:2]
                pad = max(3, int(min(x2 - x1, y2 - y1) * 0.04))
                ex1, ey1 = max(0, x1 - pad), max(0, y1 - pad)
                ex2, ey2 = min(w_mask, x2 + pad), min(h_mask, y2 + pad)
                cv2.rectangle(region_mask_raw, (ex1, ey1), (ex2, ey2), 255, -1)
            # If the box comes from ONNX, we can copy its precise pixels from mask_raw_onnx
            # If it comes from OpenCV, we fill its polygon
            if tl.get("from_opencv", False) and not has_ocr_polygon:
                pts = np.array(tl["polygon"], np.int32)
                cv2.fillPoly(region_mask_raw, [pts], 255)
            elif not tl.get("from_opencv", False):
                x1, y1, x2, y2 = tl["bbox"]
                region_mask_raw[y1:y2, x1:x2] = cv2.bitwise_or(
                    region_mask_raw[y1:y2, x1:x2],
                    mask_raw_onnx[y1:y2, x1:x2],
                )
            mask_raw = cv2.bitwise_or(mask_raw, region_mask_raw)
            region_key = tuple(int(v) for v in tl["bbox"][:4])
            region_masks[region_key] = cv2.dilate(
                cv2.GaussianBlur(region_mask_raw, (5, 5), 0),
                kernel,
                iterations=1,
            )
                
        # Dilate the validated mask
        mask_blurred = cv2.GaussianBlur(mask_raw, (5, 5), 0)
        mask = cv2.dilate(mask_blurred, kernel, iterations=1)
        ctx.mask = mask

        # ── Step 3-4: Translation + Inpainting (parallel) ──────────────
        t3 = time.time()
        log.info("Steps 3-4: Translation + Inpainting (parallel)...")

        # Clean text for translation engines: removing newlines prevents fragmented translations
        for tl in textlines:
            if "text" in tl:
                # Japanese/Chinese don't use spaces between words, others do
                if detected_source_lang in ["ja", "jpn_Jpan", "zh", "zho_Hans"]:
                    tl["text"] = tl["text"].replace("\n", "")
                else:
                    tl["text"] = tl["text"].replace("\n", " ")

        async def _run_translate():
            return await translation.translate_regions(
                textlines,
                source_lang=detected_source_lang,
                target_lang=config.target_lang,
                translator=config.translator,
                lmstudio_host=config.lmstudio_host,
                lmstudio_model=config.lmstudio_model,
                groq_api_key=config.groq_api_key,
                groq_model=config.groq_model,
            )

        async def _run_inpaint():
            return await inpainting.inpaint(
                img_rgb, mask,
                inpainting_size=config.inpainting_size,
                device=config.device,
                method=config.inpainter,
            )

        def _translation_is_valid(tl: dict) -> bool:
            source_text = tl.get("text", "")
            translated_text = tl.get("translated_text", "")
            if needs_google_fallback(
                source_text,
                translated_text,
                detected_source_lang,
                config.target_lang,
            ):
                return False
            return bool(rendering.sanitize_text(translated_text))

        # Remote providers benefit from overlap. Local LLMs and inpainting
        # compete for the same CPU/GPU/RAM, so running them together freezes
        # lower-memory PCs and is usually slower than executing them serially.
        local_provider = config.translator.lower() in {
            "llamacpp", "lmstudio", "ollama"
        }
        if local_provider:
            textlines = await _run_translate()
            # Local models are the slowest and can fail individual regions.
            # Translate first, then inpaint only regions that have a renderable
            # result. This prevents blank balloons and avoids unnecessary CPU
            # work while the local model is occupying system memory.
            valid_mask = np.zeros_like(mask)
            for tl in textlines:
                if not _translation_is_valid(tl):
                    continue
                region_key = tuple(int(v) for v in tl.get("bbox", [])[:4])
                region_mask = region_masks.get(region_key)
                if region_mask is not None:
                    valid_mask = cv2.bitwise_or(valid_mask, region_mask)
            mask = valid_mask
            ctx.mask = mask
            img_inpainted = (
                await _run_inpaint() if np.any(mask) else img_rgb.copy()
            )
        else:
            textlines, img_inpainted = await asyncio.gather(
                _run_translate(), _run_inpaint()
            )

        # Never leave an inpainted balloon empty when every translation
        # attempt failed. Restore that region from the source page and skip
        # rendering it. A visible original is preferable to erased dialogue.
        unresolved = []
        image_h, image_w = img_rgb.shape[:2]
        for tl in textlines:
            if _translation_is_valid(tl):
                continue
            unresolved.append(tl)
            bbox = tl.get("bbox", [])
            if len(bbox) < 4:
                tl["translated_text"] = ""
                tl["translation_unresolved"] = True
                continue
            region_key = tuple(int(v) for v in bbox[:4])
            region_mask = region_masks.get(region_key)
            if region_mask is not None and np.any(region_mask):
                restore_pixels = region_mask > 0
                img_inpainted[restore_pixels] = img_rgb[restore_pixels]
            else:
                x1, y1, x2, y2 = map(int, bbox[:4])
                pad = max(6, int(min(max(1, x2 - x1), max(1, y2 - y1)) * 0.08))
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(image_w, x2 + pad), min(image_h, y2 + pad)
                if x2 > x1 and y2 > y1:
                    img_inpainted[y1:y2, x1:x2] = img_rgb[y1:y2, x1:x2]
            tl["translated_text"] = ""
            tl["translation_unresolved"] = True
        if unresolved:
            log.warning(
                "Preserved source artwork for %s unresolved translation regions",
                len(unresolved),
            )

        # English scanlation fonts often leave outlined source glyphs behind.
        # Flatten only successfully translated regions; failed translations
        # must preserve their original text instead of becoming empty boxes.
        if detected_source_lang == "en":
            resolved_textlines = [
                tl for tl in textlines if not tl.get("translation_unresolved")
            ]
            img_inpainted = _flatten_uniform_english_regions(
                img_inpainted, resolved_textlines
            )

        ctx.img_inpainted = img_inpainted
        log.info(f"  Translate+Inpaint: done in {(time.time()-t3)*1000:.0f}ms")

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
            detected_source_lang=detected_source_lang,
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
    ocr.unload_all()
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
