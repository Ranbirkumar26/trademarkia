"""
scripts/cluster_analysis.py
----------------------------
Post-hoc analysis utilities for the fitted GMM clusters.

Provides:
  - Top documents per cluster (highest probability membership)
  - Boundary / uncertain documents (low max-probability, high entropy)
  - Cluster topic summaries (most common category per cluster)

Usage:
    python scripts/cluster_analysis.py
    python scripts/cluster_analysis.py --top_n 5 --n_boundary 10
"""

import argparse
import os
import sys
import pickle
from typing import List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clustering import FuzzyClusterer


def entropy(proba_row: np.ndarray) -> float:
    """Shannon entropy of a probability distribution."""
    p = proba_row[proba_row > 0]
    return float(-np.sum(p * np.log(p)))


def load_cluster_data(cluster_dir: str = "cache_store"):
    path = os.path.join(cluster_dir, "cluster_proba.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"cluster_proba.pkl not found in '{cluster_dir}'. "
            "Run scripts/build_index.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def show_top_documents(data: dict, top_n: int = 3):
    """Print top_n documents per cluster (highest membership probability)."""
    proba    = data["proba"]       # (N, K)
    doc_ids  = data["doc_ids"]
    cats     = data["categories"]
    dominant = data["dominant"]

    n_clusters = proba.shape[1]

    print("\n" + "=" * 70)
    print("TOP DOCUMENTS PER CLUSTER")
    print("=" * 70)

    for c in range(n_clusters):
        # Get all docs that have this as dominant cluster
        mask  = dominant == c
        idxs  = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        # Sort by probability of cluster c, descending
        sorted_idxs = idxs[np.argsort(proba[idxs, c])[::-1][:top_n]]
        print(f"\nCluster {c}  ({mask.sum()} docs)")
        print("-" * 40)
        for idx in sorted_idxs:
            prob = proba[idx, c]
            print(f"  [{prob:.3f}] {cats[idx]} / {doc_ids[idx].split('/')[-1]}")


def show_boundary_documents(data: dict, n_boundary: int = 10):
    """Print documents with the highest membership uncertainty (entropy)."""
    proba   = data["proba"]
    doc_ids = data["doc_ids"]
    cats    = data["categories"]

    entropies  = np.array([entropy(row) for row in proba])
    sorted_idx = np.argsort(entropies)[::-1][:n_boundary]

    print("\n" + "=" * 70)
    print("BOUNDARY / UNCERTAIN DOCUMENTS (highest membership entropy)")
    print("=" * 70)
    for rank, idx in enumerate(sorted_idx, 1):
        top2 = proba[idx].argsort()[::-1][:2]
        print(
            f"{rank:>3}. entropy={entropies[idx]:.3f} | "
            f"top clusters: {top2[0]}({proba[idx,top2[0]]:.2f}), "
            f"{top2[1]}({proba[idx,top2[1]]:.2f}) | "
            f"{cats[idx]} / {doc_ids[idx].split('/')[-1]}"
        )


def show_cluster_summaries(data: dict):
    """Print cluster summaries: dominant newsgroup category per cluster."""
    from collections import Counter

    proba    = data["proba"]
    cats     = data["categories"]
    dominant = data["dominant"]
    n_clusters = proba.shape[1]

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARIES")
    print("=" * 70)
    print(f"{'CID':>4}  {'Size':>6}  {'Dominant Category':<35}  {'Purity':>7}")
    print("-" * 60)

    for c in range(n_clusters):
        mask = dominant == c
        size = mask.sum()
        if size == 0:
            continue
        cluster_cats = [cats[i] for i in range(len(cats)) if mask[i]]
        counter      = Counter(cluster_cats)
        top_cat, top_cnt = counter.most_common(1)[0]
        purity = top_cnt / size
        print(f"{c:>4}  {size:>6}  {top_cat:<35}  {purity:>7.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_dir", default="cache_store")
    parser.add_argument("--top_n",       type=int, default=3)
    parser.add_argument("--n_boundary",  type=int, default=10)
    args = parser.parse_args()

    data = load_cluster_data(args.cluster_dir)
    print(f"\nLoaded cluster data: {len(data['doc_ids'])} documents, "
          f"{data['proba'].shape[1]} clusters.")

    show_cluster_summaries(data)
    show_top_documents(data, top_n=args.top_n)
    show_boundary_documents(data, n_boundary=args.n_boundary)


if __name__ == "__main__":
    main()
