"""
embedding_model.py
------------------
Produces document embeddings via TF-IDF + Truncated SVD (LSA).

Production note
---------------
If `sentence-transformers` is available (pip install sentence-transformers),
the EmbeddingModel class can be swapped for a SentenceTransformer wrapper with
no changes to the rest of the system — the interface (embed / embed_single /
dim) is identical.

Why TF-IDF + SVD here?
-----------------------
- `sentence-transformers` requires network access to download pre-trained model
  weights.  This offline-first implementation uses only scikit-learn (always
  available) so the index can be built in an air-gapped or CI environment.
- Truncated SVD (LSA) on TF-IDF features is the classic distributional
  semantics approach and still produces good topical similarity.
- We target 256 latent dimensions — a sweet spot between expressiveness and
  GMM stability.
- All output vectors are L2-normalised so cosine similarity equals dot product,
  matching the FAISS IndexFlatIP contract and the cache lookup.

Swapping to sentence-transformers later
----------------------------------------
Replace the class body with:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, normalize_embeddings=True, ...)

Everything else (VectorStore, SemanticCache, ClusterEngine) stays unchanged.
"""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

DEFAULT_DIM        = 256   # LSA embedding dimensionality
MIN_DF             = 2     # ignore terms that appear in fewer than 2 documents
MAX_FEATURES       = 50_000
MODEL_DIR          = "cache_store"


class EmbeddingModel:
    """
    TF-IDF + Truncated SVD (LSA) embedding model.

    The vectoriser and SVD transformer are fit once during build_index.py
    and then persisted so the API can transform new queries at inference time.
    """

    def __init__(self, dim: int = DEFAULT_DIM, model_dir: str = MODEL_DIR):
        self.dim       = dim
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = f"lsa-tfidf-d{dim}"

        self._tfidf_path = self.model_dir / "tfidf.pkl"
        self._svd_path   = self.model_dir / "svd.pkl"

        self.vectorizer: Optional[TfidfVectorizer]  = None
        self.svd:        Optional[TruncatedSVD]     = None

    # ------------------------------------------------------------------ #
    # Fit (called once by build_index.py)                                  #
    # ------------------------------------------------------------------ #

    def fit(self, texts: List[str]) -> "EmbeddingModel":
        """Fit TF-IDF and SVD on the corpus."""
        print(f"[embedding_model] Fitting TF-IDF on {len(texts)} documents …")
        self.vectorizer = TfidfVectorizer(
            min_df=MIN_DF,
            max_features=MAX_FEATURES,
            sublinear_tf=True,   # log(1+tf) — reduces impact of high-freq terms
            strip_accents="unicode",
        )
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[embedding_model] TF-IDF shape: {tfidf_matrix.shape}")

        actual_dim = min(self.dim, tfidf_matrix.shape[1] - 1)
        print(f"[embedding_model] Fitting TruncatedSVD(n_components={actual_dim}) …")
        self.svd = TruncatedSVD(n_components=actual_dim, random_state=42)
        self.svd.fit(tfidf_matrix)
        self.dim = actual_dim
        explained = self.svd.explained_variance_ratio_.sum()
        print(f"[embedding_model] SVD explains {explained*100:.1f}% of variance.  dim={self.dim}")
        return self

    def save(self) -> None:
        with open(self._tfidf_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(self._svd_path, "wb") as f:
            pickle.dump(self.svd, f)
        print(f"[embedding_model] Saved TF-IDF + SVD → {self.model_dir}")

    def load(self) -> bool:
        if not self._tfidf_path.exists() or not self._svd_path.exists():
            return False
        with open(self._tfidf_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(self._svd_path, "rb") as f:
            self.svd = pickle.load(f)
        self.dim = self.svd.n_components
        print(f"[embedding_model] Loaded TF-IDF + SVD (dim={self.dim}) from {self.model_dir}")
        return True

    # ------------------------------------------------------------------ #
    # Encode                                                               #
    # ------------------------------------------------------------------ #

    def embed(self, texts: List[str], batch_size: int = 512, show_progress: bool = True) -> np.ndarray:
        """
        Transform texts → L2-normalised float32 embeddings of shape (N, dim).
        """
        assert self.vectorizer is not None and self.svd is not None, \
            "Call fit() or load() before embed()."
        tfidf = self.vectorizer.transform(texts)
        vecs  = self.svd.transform(tfidf)
        vecs  = normalize(vecs, norm="l2")
        return vecs.astype(np.float32)

    def embed_single(self, text: str) -> np.ndarray:
        """Convenience wrapper for a single query."""
        return self.embed([text], show_progress=False)[0]


# Module-level singleton
_model_instance: Optional[EmbeddingModel] = None


def get_model(dim: int = DEFAULT_DIM, model_dir: str = MODEL_DIR) -> EmbeddingModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = EmbeddingModel(dim=dim, model_dir=model_dir)
        if not _model_instance.load():
            raise RuntimeError(
                f"Embedding model not found in '{model_dir}'. "
                "Run scripts/build_index.py first."
            )
    return _model_instance


if __name__ == "__main__":
    texts = [
        "Astronomy and space exploration missions",
        "GPU drivers crash on Windows 10",
        "Space telescope discovers new exoplanet",
    ]
    model = EmbeddingModel()
    model.fit(texts)
    vecs = model.embed(texts)
    print(f"Shape : {vecs.shape}")
    print(f"Norm  : {np.linalg.norm(vecs[0]):.4f}")
    print(f"sim(0,2) = {np.dot(vecs[0], vecs[2]):.4f}  (space vs space, should be high)")
    print(f"sim(0,1) = {np.dot(vecs[0], vecs[1]):.4f}  (space vs GPU, should be low)")
