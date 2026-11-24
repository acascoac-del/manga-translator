"""
Configuration module - GPU-optimized settings.
All settings are environment-overridable.
"""

import os
from pathlib import Path


class Settings:
    """Application settings with CUDA optimization defaults."""

    # --- Paths ---
    MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models/weights"))
    CACHE_DIR = Path(os.environ.get("CACHE_DIR", "./models/cache"))
    CONFIG_DIR = Path("./configs")

    # --- GPU / CUDA ---
    CUDA_DEVICE = "cuda:0"
    MAX_VRAM_MB = int(os.environ.get("MAX_VRAM_MB", "4096"))
    USE_FP16 = True
    USE_TENSORRT = False
    ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # --- Pipeline ---
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
    WORKERS = int(os.environ.get("WORKERS", "2"))
    MAX_IMAGE_SIZE = 2048

    # --- Detection ---
    DETECTION_MODEL = "ogkalu/comic-text-detector"
    DETECTION_CONF_THRESHOLD = 0.3
    DETECTION_IOU_THRESHOLD = 0.45

    # --- OCR ---
    OCR_MODEL = "kha-white/manga-ocr-base"
    OCR_DEVICE = "cuda"
    OCR_CONF_THRESHOLD = 0.2

    # --- Translation ---
    TRANSLATION_SERVICE = os.environ.get("TRANSLATION_SERVICE", "google")
    LIBRETRANSLATE_URL = os.environ.get("LIBRETRANSLATE_URL", "http://localhost:5003")
    GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
    MYMEMORY_URL = "https://api.mymemory.translated.net/get"
    LMSTUDIO_HOST = os.environ.get("LMSTUDIO_HOST", "http://localhost:1234/v1")
    LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "auto")

    # --- Inpainting ---
    INPAINT_MODEL = "lama"
    INPAINT_ONNX_PATH = "models/weights/lama_onnx/model.onnx"

    # --- Typesetting ---
    DEFAULT_FONT = "fonts/anime_ace_3.ttf"
    FONT_SIZE_MIN = 10
    FONT_SIZE_MAX = 28

    # --- Server ---
    HOST = "0.0.0.0"
    PORT = 8000
    LOG_LEVEL = "info"


settings = Settings()
