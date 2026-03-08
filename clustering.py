"""
clustering.py
-------------
Fuzzy document clustering using a Gaussian Mixture Model (GMM).

Why GMM over hard K-Means or Fuzzy C-Means?
--------------------------------------------
- GMM is a probabilistic model: every document receives a FULL probability
  distribution over clusters, not just a distance-based membership score.
- It handles non-spherical cluster shapes via the covariance matrix.
- It is natively supported by scikit-learn with a stable, well-tested API.
- Fuzzy C-Means (skfuzzy) requires an extra dependency and offers no material
  advantage over GMM for high-dimensional semantic embeddings.

Cluster count selection
------------------------
We use three complementary signals to select n_components (K):
  1. BIC  (Bayesian Information Criterion) – lower is better; penalises
     complexity.  The "elbow" in BIC is the best-generalising K.
  2. Silhouette score on hard assignments – measures how well-separated the
     clusters are.  Higher is better (range –1…1).
  3. Elbow on inertia (K-Means inertia proxy) – visual sanity check.

Working with 384-dim embeddings
---------------------------------
Full covariance GMMs are ill-conditioned in high dimensions. We therefore
apply PCA to reduce to 50 components before fitting the GMM. 50 dimensions
retain ~95 % of variance for typical sentence-embedding distributions and make
the covariance matrices numerically stable.

Persistence
-----------
We pickle the GMM and PCA objects to disk so the API service can load them
at startup without re-fitting.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

DEFAULT_CLUSTER_DIR = "cache_store"
N_COMPONENTS_PCA    = 50    # PCA target dimensionality
COVARIANCE_TYPE     = "diag"  # "diag" is a good balance: fewer params than "full",
                               # more expressive than "spherical"
RANDOM_STATE        = 42


class FuzzyClusterer:
    """
    Fits a GMM on PCA-reduced embeddings and exposes cluster probabilities.
    """

    def __init__(
        self,
        n_clusters: int,
        pca_components: int = N_COMPONENTS_PCA,
        cluster_dir: str = DEFAULT_CLUSTER_DIR,
    ):
        self.n_clusters    = n_clusters
        self.pca_components = pca_components
        self.cluster_dir   = Path(cluster_dir)
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

        self._gmm_path = self.cluster_dir / "gmm.pkl"
        self._pca_path = self.cluster_dir / "pca.pkl"

        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, embeddings: np.ndarray) -> "FuzzyClusterer":
        """
        Fit PCA then GMM on the embedding matrix.

        Parameters
        ----------
        embeddings : np.ndarray  shape (N, D)
        """
        actual_components = min(self.pca_components, embeddings.shape[1], embeddings.shape[0] - 1)
        print(f"[clustering] Fitting PCA({actual_components}) on {embeddings.shape} …")
        self.pca = PCA(n_components=actual_components, random_state=RANDOM_STATE)
        reduced = self.pca.fit_transform(embeddings)
        explained = self.pca.explained_variance_ratio_.sum()
        print(f"[clustering] PCA explains {explained*100:.1f}% of variance.")

        print(f"[clustering] Fitting GMM(n={self.n_clusters}, cov={COVARIANCE_TYPE}) …")
        self.gmm = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type=COVARIANCE_TYPE,
            max_iter=200,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        self.gmm.fit(reduced)
        print(f"[clustering] GMM converged: {self.gmm.converged_}")
        return self

    # ------------------------------------------------------------------ #
    # Predict                                                              #
    # ------------------------------------------------------------------ #

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Return probability matrix of shape (N, n_clusters).
        Each row is a probability distribution (sums to 1).
        """
        assert self.pca is not None and self.gmm is not None, "Call fit() first."
        reduced = self.pca.transform(embeddings)
        return self.gmm.predict_proba(reduced)

    def dominant_cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Return the argmax cluster index for each embedding. Shape (N,)."""
        return self.predict_proba(embeddings).argmax(axis=1)

    def predict_single(self, embedding: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Returns (dominant_cluster_id, probability_vector) for one embedding.
        """
        proba = self.predict_proba(embedding.reshape(1, -1))[0]
        return int(proba.argmax()), proba

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self) -> None:
        with open(self._gmm_path, "wb") as f:
            pickle.dump(self.gmm, f)
        with open(self._pca_path, "wb") as f:
            pickle.dump(self.pca, f)
        print(f"[clustering] Saved GMM + PCA → {self.cluster_dir}")

    def load(self) -> bool:
        if not self._gmm_path.exists() or not self._pca_path.exists():
            return False
        with open(self._gmm_path, "rb") as f:
            self.gmm = pickle.load(f)
        with open(self._pca_path, "rb") as f:
            self.pca = pickle.load(f)
        self.n_clusters = self.gmm.n_components
        print(f"[clustering] Loaded GMM(n={self.n_clusters}) from {self.cluster_dir}")
        return True


# ------------------------------------------------------------------ #
# Cluster count selection utilities                                   #
# ------------------------------------------------------------------ #

def select_n_clusters(
    embeddings: np.ndarray,
    k_range: range = range(10, 31, 5),
    pca_components: int = N_COMPONENTS_PCA,
) -> Dict:
    """
    Sweep over a range of K values and compute BIC and silhouette scores
    to help choose the best number of clusters.

    Returns a dict with lists 'k', 'bic', 'silhouette'.
    Prints a summary table.
    """
    print("[cluster_selection] Fitting PCA …")
    actual_components = min(pca_components, embeddings.shape[1], embeddings.shape[0] - 1)
    pca = PCA(n_components=actual_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)

    results = {"k": [], "bic": [], "silhouette": []}
    print(f"{'K':>4}  {'BIC':>12}  {'Silhouette':>12}")
    print("-" * 32)
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=COVARIANCE_TYPE,
            max_iter=150,
            random_state=RANDOM_STATE,
        )
        gmm.fit(reduced)
        bic  = gmm.bic(reduced)
        hard = gmm.predict(reduced)
        sil  = silhouette_score(reduced, hard, sample_size=min(2000, len(reduced)))
        results["k"].append(k)
        results["bic"].append(bic)
        results["silhouette"].append(sil)
        print(f"{k:>4}  {bic:>12.1f}  {sil:>12.4f}")
    return results
