FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Use non-root user as required by Hugging Face Spaces
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    curl libgl1-mesa-glx libglib2.0-0 \
    fonts-noto-cjk fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER user

# Set up Python virtual environment
RUN python3.11 -m venv $HOME/venv
ENV PATH="$HOME/venv/bin:$PATH"

# Install Python dependencies (GPU and CPU compatible)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir onnxruntime-gpu==1.19.2 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files and extension zip
COPY --chown=user app/ ./app/
COPY --chown=user MangaTranslatorAR_Extension.zip ./

# Make sure models dir is accessible
RUN mkdir -p $HOME/app/models/weights $HOME/app/models/cache

# Download models ahead of time during build (so HF Space doesn't download on startup)
# Create a small script to download models via python if needed, or rely on lazy-loading 
# (the pipeline will auto-download using huggingface_hub on first use anyway).

EXPOSE 7860

# Run uvicorn on port 7860 as required by HuggingFace Spaces
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
