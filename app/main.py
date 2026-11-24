"""
Manga Translator API - Our own engine.
Pipeline: Detect → OCR → Translate → Inpaint → Render
"""

import asyncio
import base64
import io
import logging
import os
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

# Detection, OCR, inpainting and local LLM inference share the same GPU.  Two
# complete pipelines in parallel are slower on consumer GPUs and can leave the
# browser/desktop unresponsive.  Serialize page pipelines while still allowing
# health/model-list requests to respond immediately.
_pipeline_lock = asyncio.Lock()


async def _run_pipeline_serialized(b64: str, config: PipelineConfig):
    async with _pipeline_lock:
        return await run_pipeline(b64, config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[API] Starting MangaTranslatorAR API...")
    log.info(f"[API] Device: {DEVICE}")
    await initialize()
    # Loading the GGUF from the Windows bind mount takes around 20-25 seconds.
    # Pay that cost while the server starts instead of making the first manga
    # page appear stuck on "thinking". Other providers still release the model
    # at the beginning of their first request.
    if os.getenv("PRELOAD_LLAMACPP", "1").strip().lower() not in {"0", "false", "no"}:
        try:
            from app.engine.translation import llamacpp

            log.info("[API] Preloading llama.cpp into VRAM...")
            await llamacpp.preload()
        except Exception as exc:
            log.warning("[API] llama.cpp preload failed: %s", exc)
    log.info("[API] Core image models loaded. Server ready.")
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
            "ocr": "auto",
            "translator": "llamacpp",
            "inpainter": "lama",
            "renderer": "manga2eng",
            "target_lang": "es",
        },
    }


@app.get("/lmstudio/models")
async def get_lmstudio_models():
    from app.engine.translation.lmstudio import _get_lmstudio_host, _lmstudio_api_url
    api_host = _get_lmstudio_host()
    url = _lmstudio_api_url(api_host, "models")
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            models = [item.get("id") for item in data.get("data", []) if item.get("id")]
            return {"models": models, "host": api_host}
    except Exception as e:
        log.error(f"Failed to query LM Studio models at {url}: {e}")
        raise HTTPException(503, "LM Studio service is unavailable")


@app.post("/translate")
async def translate_multipart(
    image: UploadFile = File(...),
    target_lang: str = Form("es"),
    source_lang: str = Form("auto"),
    detector: str = Form("ctd"),
    ocr: str = Form("auto"),
    translator: str = Form("llamacpp"),
    inpainter: str = Form("lama"),
    renderer: str = Form("manga2eng"),
    font_family: str = Form("BarlowSemiCondensed-Medium"),
    font_size: int = Form(0),
    lmstudio_model: str = Form("auto"),
    lmstudio_host: str = Form(""),
    groq_api_key: str = Form(""),
    groq_model: str = Form("auto"),
):
    raw = await image.read()
    if not raw:
        raise HTTPException(400, "Empty image")
    b64 = base64.b64encode(raw).decode("ascii")

    config = PipelineConfig(
        detector=detector, ocr=ocr, translator=translator,
        inpainter=inpainter, renderer=renderer,
        source_lang=source_lang or "auto", target_lang=target_lang,
        font_family=font_family, font_size=font_size,
        lmstudio_model=lmstudio_model,
        lmstudio_host=lmstudio_host if lmstudio_host else None,
        groq_api_key=groq_api_key if groq_api_key else None,
        groq_model=groq_model if groq_model else None,
    )

    result = await _run_pipeline_serialized(b64, config)
    return _format_result(result)


@app.post("/translate/json")
async def translate_json(req: Request):
    body = await req.json()
    b64 = body.get("image")
    if not b64:
        raise HTTPException(400, "Field 'image' (base64) required")

    if isinstance(b64, str) and "," in b64 and b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    config = PipelineConfig(
        detector=body.get("detector", "ctd"),
        ocr=body.get("ocr", "auto"),
        translator=body.get("translator", "llamacpp"),
        inpainter=body.get("inpainter", "lama"),
        renderer=body.get("renderer", "manga2eng"),
        source_lang=body.get("source_lang") or "auto",
        target_lang=body.get("target_lang", "es"),
        font_family=body.get("font_family", "BarlowSemiCondensed-Medium"),
        font_size=int(body.get("font_size", 0)),
        lmstudio_model=body.get("lmstudio_model", body.get("ollama_model", "auto")),
        lmstudio_host=body.get("lmstudio_host", body.get("ollama_host")),
        groq_api_key=body.get("groq_api_key"),
        groq_model=body.get("groq_model", "auto"),
    )

    result = await _run_pipeline_serialized(b64, config)
    return _format_result(result)


@app.post("/translate/batch")
async def translate_batch(req: Request):
    body = await req.json()
    images = body.get("images")
    if not images or not isinstance(images, list):
        raise HTTPException(400, "Field 'images' required (array of base64)")
    if len(images) > 50:
        raise HTTPException(400, "Max 50 images per batch")

    config = PipelineConfig(
        detector=body.get("detector", "ctd"),
        ocr=body.get("ocr", "auto"),
        translator=body.get("translator", "google"),
        inpainter=body.get("inpainter", "lama"),
        renderer=body.get("renderer", "manga2eng"),
        source_lang=body.get("source_lang") or "auto",
        target_lang=body.get("target_lang", "es"),
        font_family=body.get("font_family", "BarlowSemiCondensed-Medium"),
        font_size=int(body.get("font_size", 0)),
        lmstudio_model=body.get("lmstudio_model", body.get("ollama_model", "auto")),
        lmstudio_host=body.get("lmstudio_host", body.get("ollama_host")),
        groq_api_key=body.get("groq_api_key"),
        groq_model=body.get("groq_model", "auto"),
    )

    batch_start = time.time()

    async def process_one(index, b64_item):
        if isinstance(b64_item, str) and "," in b64_item and b64_item.startswith("data:"):
            b64_item = b64_item.split(",", 1)[1]
        result = await _run_pipeline_serialized(b64_item, config)
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
        "detected_source_lang": result.detected_source_lang,
        "regions": result.regions,
        "original_image": result.original_b64,
    })


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8002))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
