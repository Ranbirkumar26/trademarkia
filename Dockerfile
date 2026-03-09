# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Avoid interactive prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ──────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer-cached) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project source ────────────────────────────────────────────────────────
COPY . .

# ── Environment variables (override at runtime) ───────────────────────────────
ENV VECTOR_STORE_DIR=/app/vector_db
ENV CLUSTER_DIR=/app/cache_store
ENV CACHE_THRESHOLD=0.85

# ── Expose API port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# The index must be built before starting the API.
# To build the index inside the container run:
#   docker run --rm -v $(pwd)/data:/app/data <image> python scripts/build_index.py --data_dir /app/data
CMD ["bash", "-c", "python scripts/build_index.py --data_dir /app/dataset_raw/20_newsgroups && uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
