"""
semantic_cache.py
-----------------
An in-memory semantic cache with cluster-scoped lookup.

Cache design rationale
-----------------------
Traditional caches key on exact string equality.  A semantic cache keys on
*meaning*: two queries phrased differently but semantically equivalent should
share a cache entry.

Lookup algorithm
-----------------
1. Embed the incoming query → query_vec.
2. Ask the clusterer for its dominant cluster → cluster_id.
3. Only iterate over cache entries in that cluster bucket.
   This limits the comparison work from O(total_entries) to O(bucket_size),
   which is typically much smaller.
4. Compute cosine similarity between query_vec and each stored embedding.
   (Vectors are L2-normalised so dot product ≡ cosine similarity.)
5. If max_similarity >= threshold → cache hit, return stored result.
6. Otherwise → cache miss, let the caller perform a vector search and then
   call store() to populate the cache.

Data structure
--------------
cache = {
    cluster_id (int): [
        {
            "query"     : str,
            "embedding" : np.ndarray (dim,) float32,
            "result"    : str,
            "timestamp" : float,
        },
        ...
    ]
}

Thread safety: this implementation is NOT thread-safe. A production deployment
with concurrent workers would need a lock around the cache dict or an
external store. For this single-worker uvicorn deployment it is sufficient.

Eviction: simplest-possible LRU eviction per bucket (drop oldest entry when
bucket exceeds MAX_BUCKET_SIZE). This bounds memory without a full LRU heap.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

DEFAULT_THRESHOLD     = 0.85   # cosine similarity threshold for a cache hit
MAX_BUCKET_SIZE       = 200    # max entries per cluster bucket before eviction


@dataclass
class CacheStats:
    hit_count:     int = 0
    miss_count:    int = 0
    total_entries: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "total_entries": self.total_entries,
            "hit_count":     self.hit_count,
            "miss_count":    self.miss_count,
            "hit_rate":      round(self.hit_rate, 4),
        }


class SemanticCache:
    """
    Cluster-partitioned semantic cache.
    """

    def __init__(
        self,
        threshold:       float = DEFAULT_THRESHOLD,
        max_bucket_size: int   = MAX_BUCKET_SIZE,
    ):
        self.threshold       = threshold
        self.max_bucket_size = max_bucket_size

        # Main storage: cluster_id → list of cache entries
        self._cache: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        self._stats = CacheStats()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_id: int,
    ) -> Tuple[bool, Optional[str], Optional[str], float]:
        """
        Try to find a semantically similar cached query.

        Parameters
        ----------
        query_embedding : np.ndarray  shape (dim,), float32, L2-normalised
        cluster_id : int

        Returns
        -------
        (hit, matched_query, result, similarity_score)
        """
        bucket = self._cache.get(cluster_id, [])

        best_score      = -1.0
        best_entry      = None

        for entry in bucket:
            # Dot product of two L2-normalised vectors = cosine similarity
            score = float(np.dot(query_embedding, entry["embedding"]))
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry is not None:
            self._stats.hit_count += 1
            return True, best_entry["query"], best_entry["result"], best_score

        self._stats.miss_count += 1
        return False, None, None, best_score

    def store(
        self,
        query:           str,
        query_embedding: np.ndarray,
        cluster_id:      int,
        result:          str,
    ) -> None:
        """
        Insert a new entry into the appropriate cluster bucket.
        Evicts the oldest entry if the bucket exceeds max_bucket_size.
        """
        bucket = self._cache[cluster_id]

        # LRU-style eviction: remove the oldest (first) entry
        if len(bucket) >= self.max_bucket_size:
            bucket.pop(0)

        bucket.append({
            "query":     query,
            "embedding": query_embedding.astype(np.float32),
            "result":    result,
            "timestamp": time.time(),
        })
        self._stats.total_entries = sum(len(b) for b in self._cache.values())

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._cache.clear()
        self._stats = CacheStats()

    @property
    def stats(self) -> CacheStats:
        return self._stats

    def bucket_sizes(self) -> Dict[int, int]:
        """Return a mapping of cluster_id → number of entries (for diagnostics)."""
        return {cid: len(b) for cid, b in self._cache.items()}
