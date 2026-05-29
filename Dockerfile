# syntax=docker/dockerfile:1.7
# ──────────────────────────────────────────────────────────────────────────────
#  Backend image — FastAPI + RAG pipeline + all heavy ML deps
# ──────────────────────────────────────────────────────────────────────────────
# Notes:
#   • python:3.12-slim base. Python 3.13/3.14 wheels for torch + chromadb don't
#     exist yet on every platform — slim/3.12 has the best wheel coverage.
#   • ffmpeg is needed at runtime by VideoProcessor; we install it via apt.
#   • This image is genuinely large (~3-4 GB) because of torch + whisper +
#     docling + chromadb + sentence-transformers. There is no shortcut.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System dependencies. ffmpeg = audio/video; git, build-essential = some pip
# packages (whisper, sentence-transformers) compile small extensions at install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ONLY requirements first so layer cache survives source edits.
COPY requirements.txt ./

# Install Python deps. Pin torch to the CPU-only index — saves multi-GB by
# skipping the CUDA wheels we can't use without GPU passthrough.
#
# The build is RAM/disk-intensive: ~50 packages × multi-MB wheels can fill
# /tmp during the install phase, so we run the install in one shot and
# scrub pip's cache + temp files in the same RUN layer to keep the final
# layer manageable.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefer-binary --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
 && pip install --prefer-binary -r requirements.txt \
 && find /usr/local/lib/python3.12/site-packages -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true \
 && find /usr/local/lib/python3.12/site-packages -name "*.pyc" -delete \
 && rm -rf /tmp/pip-* /root/.cache/huggingface 2>/dev/null || true

# Now copy the application code.
COPY api.py memory_store.py rag_connector.py multimodal_rag_pipeline.py ./

# All persistent state lives here; mount a named volume on this path.
ENV RAG_DATA_DIR=/data
RUN mkdir -p /data
VOLUME ["/data"]

# Backend port
EXPOSE 8000

# Healthcheck so docker-compose can wait for us before starting the frontend.
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --retries=5 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# uvicorn directly — no --reload in production.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
