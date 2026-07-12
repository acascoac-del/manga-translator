#!/bin/bash
# =============================================================================
# Manga Translator API - Startup Script
# =============================================================================

echo "============================================="
echo "  Manga Translator API"
echo "  GPU-accelerated manga translation"
echo "============================================="

# Check NVIDIA Container Toolkit
if ! command -v nvidia-smi &> /dev/null; then
    echo "[WARN] nvidia-smi not found. GPU may not be available."
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "[INFO] Using 'docker compose' (v2 syntax)"
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Start services
echo "[INFO] Starting services..."
$COMPOSE_CMD up --build -d

echo "[INFO] Waiting for API to be ready..."
sleep 10

# Health check
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "[OK] API is ready at http://localhost:8000"
    echo "[OK] API docs at http://localhost:8000/docs"
else
    echo "[WARN] API may still be loading models..."
    echo "[INFO] Check logs: $COMPOSE_CMD logs -f manga-translator"
fi

echo ""
echo "Usage:"
echo "  Translate image:  curl -X POST http://localhost:8000/translate -F 'image=@manga.png'"
echo "  API docs:         http://localhost:8000/docs"
echo "  Health check:     http://localhost:8000/health"
echo "  Stop:             $COMPOSE_CMD down"
