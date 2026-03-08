"""
api/main.py
-----------
FastAPI service exposing semantic search and cache management endpoints.

Startup: loads the SearchEngine (which loads FAISS + GMM from disk).
All endpoints share the single engine instance via a module-level variable
that is initialised in the lifespan handler.
"""

import sys, os
# Make the project root importable when running from inside api/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from search_engine import SearchEngine

# ------------------------------------------------------------------ #
# Application state                                                   #
# ------------------------------------------------------------------ #

engine: Optional[SearchEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy resources once at startup."""
    global engine
    try:
        engine = SearchEngine(
            vector_store_dir=os.environ.get("VECTOR_STORE_DIR", "vector_db"),
            cluster_dir=os.environ.get("CLUSTER_DIR", "cache_store"),
            cache_threshold=float(os.environ.get("CACHE_THRESHOLD", "0.85")),
        )
    except RuntimeError as exc:
        print(f"[startup] ERROR: {exc}")
        # Allow the app to start in degraded mode – /query will return 503
        engine = None
    yield
    # Shutdown: nothing to clean up (all state is in-memory)


app = FastAPI(
    title="Semantic Search API",
    description="Lightweight semantic search over 20 Newsgroups with fuzzy clustering and semantic cache.",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ #
# Request / Response models                                           #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query:            str
    cache_hit:        bool
    matched_query:    Optional[str]
    similarity_score: float
    result:           str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count:     int
    miss_count:    int
    hit_rate:      float


# ------------------------------------------------------------------ #
# Endpoints                                                           #
# ------------------------------------------------------------------ #

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    """
    Perform a semantic search query.

    1. Embeds the query text.
    2. Determines the dominant cluster via GMM.
    3. Checks the semantic cache (cluster-scoped cosine similarity lookup).
    4. On cache miss: searches the FAISS vector database and stores the result.
    """
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialised. Run scripts/build_index.py first.",
        )
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    result = engine.query(req.query.strip())
    return QueryResponse(**result)


@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats():
    """Return current cache hit/miss statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    return CacheStatsResponse(**engine.cache.stats.to_dict())


@app.delete("/cache")
def clear_cache():
    """Clear all semantic cache entries and reset statistics."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready.")
    engine.cache.clear()
    return {"message": "Cache cleared successfully."}


@app.get("/health")
def health():
    """Simple health check."""
    return {
        "status":        "ok" if engine is not None else "degraded",
        "index_size":    engine.store.size if engine else 0,
        "n_clusters":    engine.clusterer.n_clusters if engine else 0,
    }
