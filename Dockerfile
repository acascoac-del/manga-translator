# =============================================================================
# Manga Translator API - Docker Build
# Uses manga-image-translator as core engine (same as MangaLingo)
# =============================================================================

# Stage 1: Builder
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS builder

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python3-pip \
    curl wget git \
    && rm -rf /var/lib/apt/lists/*

RUN python${PYTHON_VERSION} -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
# Install llama.cpp exclusively from the CUDA wheel index. Using it as an
# extra index lets pip prefer a newer CPU-only wheel from public PyPI.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir onnxruntime-gpu==1.19.2 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir llama-cpp-python \
        --index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Stage 2: Runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime

ARG PYTHON_VERSION=3.11
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-venv \
    curl libgl1-mesa-glx libglib2.0-0 \
    libcublas-12-1 \
    fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN groupadd -r manga && useradd -r -g manga -d /app manga

WORKDIR /app

COPY app/ /app/app/
COPY configs/ /app/configs/

RUN mkdir -p /app/models/weights /app/models/cache && \
    chown -R manga:manga /app

USER manga

HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')" || exit 1

EXPOSE 8002

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8002", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools"]
