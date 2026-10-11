# Universal Document RAG v5 — hybrid and lexical deployment profiles.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf-cache \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt requirements-server-lexical.txt requirements-torch-cpu.txt ./
ARG SEARCH_PROFILE=hybrid
RUN if [ "$SEARCH_PROFILE" = "lexical" ]; then \
        pip install -r requirements-server-lexical.txt; \
    elif [ "$SEARCH_PROFILE" = "hybrid" ]; then \
        pip install -r requirements-torch-cpu.txt && pip install -r requirements-server.txt; \
    else \
        echo "Invalid SEARCH_PROFILE=$SEARCH_PROFILE" && exit 2; \
    fi

COPY . .

# Optional Hugging Face model baking; name must match index build configuration.
ARG BAKE_MODEL=0
ARG EMBEDDING_MODEL=intfloat/multilingual-e5-small
RUN if [ "$SEARCH_PROFILE" = "hybrid" ] && [ "$BAKE_MODEL" = "1" ]; then \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('${EMBEDDING_MODEL}')"; \
    fi

RUN useradd -r -u 10001 rag && chown -R rag:rag /app
USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
