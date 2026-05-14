# SCM hosted demo container.
#
# Builds the SCM REST API on Python 3.11 slim. Defaults to Profile A
# (offline-only — heuristic encoder + sentence-transformers MiniLM) so
# the image is self-contained and runs without LLM provider credentials.
#
# To enable a stronger profile at deploy time:
#   docker run -e LLM_PROVIDER=deepseek -e DEEPSEEK_API_KEY=... ...
#   docker run -e LLM_PROVIDER=openai -e OPENAI_API_KEY=... ...
#
# Image is single-stage and intentionally minimal — no Ollama bundled
# (Ollama would balloon the image to ~5 GB; users pair with Ollama
# externally if they want Profile B).

FROM python:3.11-slim AS base

# System deps for sentence-transformers + scientific Python
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer caching: copy only what pip needs first
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Pre-create the data dir SCM persists to
RUN mkdir -p /data && chmod 777 /data

# Install the package + its dependencies
RUN pip install --no-cache-dir -e .

# Pre-fetch the default sentence-transformer model so first request is fast.
# (Skips the cold-start download; ~80 MB added to the image.)
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
    || echo "sentence-transformer pre-fetch failed; will fall back to hash"

# Reasonable production defaults
ENV SCM_DATA_DIR=/data \
    SCM_EMBEDDING_BACKEND=sentence_transformers \
    SCM_AUTO_SLEEP_DISABLE=0 \
    SCM_IDLE_THRESHOLD_SEC=300 \
    SCM_MCP_SWEEP_INTERVAL_SEC=30 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Health check uses the SCM /v1/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fs http://localhost:${PORT}/v1/health || exit 1

# Run the API server. Uvicorn binds to 0.0.0.0 so the container is reachable.
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
