# Build the existing engine image under a NEW tag; never replace/restart local.
ARG ENGINE_IMAGE=mangatranslator-engine:commercial
FROM ${ENGINE_IMAGE}
USER root
RUN pip install --no-cache-dir 'runpod>=1.7,<2'
COPY commercial/runpod/ /app/commercial/runpod/
COPY app/ /app/app/
COPY fonts/ /app/fonts/
ENV PYTHONPATH=/app
ENV PRELOAD_LLAMACPP=0
ENV HF_HOME=/runpod-volume/huggingface
ENV MODEL_DIR=/app/models/weights
ENV OMP_NUM_THREADS=2
ENV MANGA_OCR_BATCH_SIZE=8
HEALTHCHECK NONE
# Mount prepared model weights at /app/models/weights or bake them in a private
# image after checking distribution licenses. HF cache alone is not all models.
CMD ["python", "-u", "/app/commercial/runpod/handler.py"]
