"""
vector_store.py
---------------
Persistent vector database backed by NumPy matrix operations.

Design decision – NumPy vs FAISS vs ChromaDB
---------------------------------------------
This implementation uses NumPy for offline / air-gapped environments.
O(N·dim) per query; fine for ≤50 k documents. To swap in FAISS:

    import faiss
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    scores, indices = index.search(query.reshape(1,-1), top_k)

Persistence
-----------
  vector_db/embeddings.npy   – float32 matrix (N, dim)
  vector_db/metadata.pkl     – List[dict] parallel to rows
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

DEFAULT_STORE_DIR = "vector_db"


class VectorStore:
    def __init__(self, dim: int, store_dir: str = DEFAULT_STORE_DIR):
        self.dim = dim
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

        self._emb_path  = self.store_dir / "embeddings.npy"
        self._meta_path = self.store_dir / "metadata.pkl"

        self._embeddings: np.ndarray = np.empty((0, dim), dtype=np.float32)
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]) -> None:
        assert len(embeddings) == len(metadata_list)
        if self._embeddings.shape[0] == 0:
            self._embeddings = embeddings.astype(np.float32)
        else:
            self._embeddings = np.vstack([self._embeddings, embeddings.astype(np.float32)])
        self.metadata.extend(metadata_list)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """Cosine similarity search (vectors are L2-normalised → dot product)."""
        if self._embeddings.shape[0] == 0:
            return []
        q = query_embedding.astype(np.float32).reshape(1, -1)
        scores = (self._embeddings @ q.T).squeeze()
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results = []
        for idx in top_idx:
            entry = dict(self.metadata[idx])
            entry["score"] = float(scores[idx])
            results.append(entry)
        return results

    def save(self) -> None:
        np.save(str(self._emb_path), self._embeddings)
        with open(self._meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[vector_store] Saved {len(self.metadata)} vectors → {self.store_dir}")

    def load(self) -> bool:
        if not self._emb_path.exists() or not self._meta_path.exists():
            return False
        self._embeddings = np.load(str(self._emb_path))
        with open(self._meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        if self._embeddings.ndim == 2:
            self.dim = self._embeddings.shape[1]
        print(f"[vector_store] Loaded {len(self.metadata)} vectors from {self.store_dir}")
        return True

    @property
    def size(self) -> int:
        return len(self.metadata)

    def get_all_embeddings(self) -> np.ndarray:
        return self._embeddings.copy()
