# Semantic Search System — 20 Newsgroups

A lightweight, production-structured semantic search engine with fuzzy clustering and
an in-memory semantic cache.  No Redis, no cloud dependencies.

---

## Architecture

```
Query ──► EmbeddingModel
              │
              ▼
         FuzzyClusterer (GMM)
              │  dominant_cluster_id
              ▼
         SemanticCache.lookup()
           ├─ HIT  ──► return cached result
           └─ MISS
                │
                ▼
           VectorStore (FAISS)
                │  top-3 docs
                ▼
           SemanticCache.store()
                │
                ▼
           return result
```

---

## Folder Structure

```
semantic_search/
├── data_loader.py        # Load raw 20NG files from disk
├── preprocessing.py      # Header stripping, cleaning, normalisation
├── embedding_model.py    # all-MiniLM-L6-v2 sentence embeddings
├── vector_store.py       # FAISS index (persist / search)
├── clustering.py         # GMM fuzzy clusterer + PCA + cluster selection
├── semantic_cache.py     # In-memory cluster-partitioned semantic cache
├── search_engine.py      # Façade used by the API
├── api/
│   └── main.py           # FastAPI endpoints
├── scripts/
│   ├── build_index.py    # One-time pipeline: embed → index → cluster
│   └── cluster_analysis.py  # Interpretability utilities
├── vector_db/            # FAISS index (auto-created by build_index.py)
├── cache_store/          # GMM + PCA models (auto-created)
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build the index

```bash
# Full dataset (≈18 k docs, ~10 min on CPU first run)
python scripts/build_index.py --data_dir /path/to/20_newsgroups

# Fast dev run (2000 docs)
python scripts/build_index.py --data_dir /path/to/20_newsgroups --max_docs 2000

# Skip cluster sweep if you already know K
python scripts/build_index.py --data_dir /path/to/20_newsgroups --n_clusters 20
```

### 3. Start the API

```bash
uvicorn api.main:app --reload
```

Swagger UI: http://localhost:8000/docs

---

## API Reference

### POST /query

```json
{
  "query": "How do I install a GPU driver on Linux?"
}
```

Response:
```json
{
  "query": "How do I install a GPU driver on Linux?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0.62,
  "result": "[1] [comp.os.ms-windows.misc] ...",
  "dominant_cluster": 7
}
```

### GET /cache/stats

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

### DELETE /cache

Clears all cache entries and resets statistics.

### GET /health

Returns index size, cluster count, and engine status.

---

## Cluster Analysis

```bash
python scripts/cluster_analysis.py
python scripts/cluster_analysis.py --top_n 5 --n_boundary 15
```

Prints:
- Cluster summaries (dominant newsgroup, purity)
- Top documents per cluster
- Boundary documents (uncertain membership, high entropy)

---

## Docker

```bash
# Build image
docker build -t semantic-search .

# Build the index (mount dataset)
docker run --rm \
  -v $(pwd)/dataset_raw/20_newsgroups:/data \
  -v $(pwd)/vector_db:/app/vector_db \
  -v $(pwd)/cache_store:/app/cache_store \
  semantic-search \
  python scripts/build_index.py --data_dir /data

# Run the API
docker run -p 8000:8000 \
  -v $(pwd)/vector_db:/app/vector_db \
  -v $(pwd)/cache_store:/app/cache_store \
  semantic-search
```

---

## Design Decisions

### Embeddings
`all-MiniLM-L6-v2` (22 M params, 384-dim) delivers strong semantic quality with
fast CPU inference. Embeddings are L2-normalised so cosine similarity equals dot
product — this simplifies both FAISS search and cache lookup.

### Vector Store (FAISS)
`IndexFlatIP` is exact nearest-neighbour search, appropriate for ≤50 k documents.
The index and metadata are persisted as a binary file + pickle, making restarts fast.

### Fuzzy Clustering (GMM)
A Gaussian Mixture Model provides probabilistic soft membership: each document
belongs to every cluster with a probability. This is richer than hard K-Means.
We reduce to 50 PCA dimensions before fitting to keep covariance matrices
numerically stable. Cluster count is chosen by minimising BIC over a sweep.

### Semantic Cache
The cache is partitioned by cluster ID so lookup is O(bucket_size) rather than
O(total_entries). A cosine similarity threshold (default 0.85) governs cache hits.
Oldest-entry eviction keeps bucket sizes bounded (default 200 per cluster).
No external dependencies — pure Python + NumPy.
