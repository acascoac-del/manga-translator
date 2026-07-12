"""
Manga Translator API - Our own engine.
Pipeline: Detect → OCR → Translate → Inpaint → Render
"""

import base64
import io
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import httpx

from app.engine.pipeline import (
    PipelineConfig, PipelineResult,
    run_pipeline, initialize, cleanup,
    DEVICE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("manga-api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[API] Starting MangaTranslatorAR API...")
    log.info(f"[API] Device: {DEVICE}")
    await initialize()
    log.info("[API] Models loaded. Server ready.")
    yield
    log.info("[API] Shutting down...")
    await cleanup()


app = FastAPI(
    title="MangaTranslatorAR",
    description="Detect → OCR → Translate → Inpaint → Render",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.middleware("http")
async def add_headers(request: Request, call_next):
    t0 = time.perf_counter()
    rid = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    response = await call_next(request)
    elapsed = (time.perf_counter() - t0) * 1000
    response.headers["x-process-time"] = f"{elapsed:.1f}"
    response.headers["x-request-id"] = rid
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


# =============================================================================
# Health
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": "manga-translator-ar",
        "version": "2.0.0",
        "device": DEVICE,
        "pipeline": ["detection", "ocr", "translation", "inpainting", "rendering"],
        "defaults": {
            "detector": "ctd",
            "ocr": "manga_ocr",
            "translator": "google",
            "inpainter": "lama",
            "renderer": "manga2eng",
            "target_lang": "es",
        },
    }


# =============================================================================
# Ollama Models
# =============================================================================

@app.get("/ollama/models")
async def get_ollama_models():
    """Retrieve the list of models pulled in the local Ollama instance."""
    from app.engine.translation.ollama import _get_ollama_host
    api_host = _get_ollama_host()
    url = f"{api_host.rstrip('/')}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            # Extract names
            models = [m.get("name") for m in data.get("models", [])]
            return {"models": models}
    except Exception as e:
        log.error(f"Failed to query Ollama models at {url}: {e}")
        # Return fallback models to choose from if Ollama is not reachable or fails
        return {"models": ["gemma3:4b", "llama3", "gemma:2b", "phi3"]}


# =============================================================================
# Translate
# =============================================================================

@app.post("/translate")
async def translate_multipart(
    image: UploadFile = File(...),
    target_lang: str = Form("es"),
    source_lang: str = Form("auto"),
    detector: str = Form("ctd"),
    ocr: str = Form("manga_ocr"),
    translator: str = Form("google"),
    inpainter: str = Form("lama"),
    renderer: str = Form("manga2eng"),
    font_family: str = Form("anime_ace_3"),
    font_size: int = Form(0),
    ollama_model: str = Form("llama3"),
    ollama_host: str = Form(""),
    libretranslate_host: str = Form("http://localhost:5000"),
):
    """Translate a manga image (FormData)."""
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty image")
    b64 = base64.b64encode(raw).decode("ascii")

    config = PipelineConfig(
        detector=detector, ocr=ocr, translator=translator,
        inpainter=inpainter, renderer=renderer,
        source_lang=source_lang, target_lang=target_lang,
        font_family=font_family, font_size=font_size,
        ollama_model=ollama_model, ollama_host=ollama_host if ollama_host else None,
        libretranslate_host=libretranslate_host,
    )

    result = await run_pipeline(b64, config)
    return _format_result(result)


@app.post("/translate/json")
async def translate_json(req: Request):
    """Translate from JSON body with base64 image."""
    body = await req.json()
    b64 = body.get("image")
    if not b64:
        raise HTTPException(400, "Field 'image' (base64) required")

    if isinstance(b64, str) and "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    config = PipelineConfig(
        detector=body.get("detector", "ctd"),
        ocr=body.get("ocr", "manga_ocr"),
        translator=body.get("translator", "google"),
        inpainter=body.get("inpainter", "lama"),
        renderer=body.get("renderer", "manga2eng"),
        source_lang=body.get("source_lang", "auto"),
        target_lang=body.get("target_lang", "es"),
        font_family=body.get("font_family", "anime_ace_3"),
        font_size=int(body.get("font_size", 0)),
        ollama_model=body.get("ollama_model", "llama3"),
        ollama_host=None,  # Always resolved server-side from OLLAMA_HOST env var
        libretranslate_host=body.get("libretranslate_host", "http://localhost:5000"),
    )

    result = await run_pipeline(b64, config)
    return _format_result(result)


@app.post("/translate/batch")
async def translate_batch(req: Request):
    """Translate multiple images in parallel."""
    import asyncio

    body = await req.json()
    images = body.get("images")
    if not images or not isinstance(images, list):
        raise HTTPException(400, "Field 'images' required (array of base64)")
    if len(images) > 50:
        raise HTTPException(400, f"Max 50 images per batch")

    config = PipelineConfig(
        detector=body.get("detector", "ctd"),
        ocr=body.get("ocr", "manga_ocr"),
        translator=body.get("translator", "google"),
        inpainter=body.get("inpainter", "lama"),
        renderer=body.get("renderer", "manga2eng"),
        source_lang=body.get("source_lang", "auto"),
        target_lang=body.get("target_lang", "es"),
        font_family=body.get("font_family", "anime_ace_3"),
        font_size=int(body.get("font_size", 0)),
        ollama_model=body.get("ollama_model", "llama3"),
        ollama_host=body.get("ollama_host", None),
        libretranslate_host=body.get("libretranslate_host", "http://localhost:5000"),
    )

    batch_start = time.time()

    async def process_one(index, b64):
        if isinstance(b64, str) and "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        result = await run_pipeline(b64, config)
        item = {"index": index, "success": result.success, "processing_time_ms": result.processing_time_ms}
        if result.success:
            item["translated_image"] = result.translated_b64
            item["region_count"] = len(result.regions)
        else:
            item["error"] = result.error
        return item

    results = await asyncio.gather(*[process_one(i, b) for i, b in enumerate(images)])
    total_ms = int((time.time() - batch_start) * 1000)
    ok = sum(1 for r in results if r["success"])

    return JSONResponse(content={
        "success": ok > 0,
        "total": len(images),
        "succeeded": ok,
        "failed": len(images) - ok,
        "processing_time_ms": total_ms,
        "results": results,
    })


def _format_result(result: PipelineResult):
    """Format pipeline result as JSON response."""
    if not result.success:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": result.error,
            "processing_time_ms": result.processing_time_ms,
        })

    return JSONResponse(content={
        "success": True,
        "translated_image": result.translated_b64,
        "processing_time_ms": result.processing_time_ms,
        "region_count": len(result.regions),
        "regions": result.regions,
        "original_image": result.original_b64,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
