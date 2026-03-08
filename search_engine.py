"""
search_engine.py
----------------
Ties together the embedding model, vector store, clusterer, and semantic cache.
"""

from typing import Any, Dict

import numpy as np

from embedding_model import EmbeddingModel, get_model
from vector_store     import VectorStore
from clustering       import FuzzyClusterer
from semantic_cache   import SemanticCache

TOP_K = 3


class SearchEngine:
    def __init__(
        self,
        vector_store_dir: str   = "vector_db",
        cluster_dir:      str   = "cache_store",
        cache_threshold:  float = 0.85,
        embedding_dim:    int   = 256,
    ):
        # Embedding model
        self.embedder: EmbeddingModel = get_model(model_dir=cluster_dir)

        # Vector store
        self.store = VectorStore(dim=self.embedder.dim, store_dir=vector_store_dir)
        if not self.store.load():
            raise RuntimeError(
                f"No vector index found in '{vector_store_dir}'. "
                "Run scripts/build_index.py first."
            )

        # Clusterer
        self.clusterer = FuzzyClusterer(n_clusters=0, cluster_dir=cluster_dir)
        if not self.clusterer.load():
            raise RuntimeError(
                f"No GMM model found in '{cluster_dir}'. "
                "Run scripts/build_index.py first."
            )

        # Semantic cache
        self.cache = SemanticCache(threshold=cache_threshold)
        print("[search_engine] Ready.")

    def query(self, text: str) -> Dict[str, Any]:
        q_vec = self.embedder.embed_single(text)
        cluster_id, proba = self.clusterer.predict_single(q_vec)
        hit, matched_query, cached_result, sim_score = self.cache.lookup(q_vec, cluster_id)

        if hit:
            return {
                "query":            text,
                "cache_hit":        True,
                "matched_query":    matched_query,
                "similarity_score": round(sim_score, 4),
                "result":           cached_result,
                "dominant_cluster": cluster_id,
            }

        results    = self.store.search(q_vec, top_k=TOP_K)
        result_text = self._format_results(results)
        self.cache.store(text, q_vec, cluster_id, result_text)

        return {
            "query":            text,
            "cache_hit":        False,
            "matched_query":    None,
            "similarity_score": round(sim_score, 4),
            "result":           result_text,
            "dominant_cluster": cluster_id,
        }

    @staticmethod
    def _format_results(results) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            score    = r.get("score", 0)
            category = r.get("category", "unknown")
            text     = r.get("text", "")[:500]
            doc_id   = r.get("doc_id", "")
            parts.append(f"[{i}] [{category}] (score={score:.3f}) {doc_id}\n{text}")
        return "\n\n---\n\n".join(parts) if parts else "No results found."
