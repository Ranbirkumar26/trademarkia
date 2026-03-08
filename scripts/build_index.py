"""
scripts/build_index.py
----------------------
End-to-end pipeline: load → preprocess → embed → index → cluster → save.

Usage:
    python scripts/build_index.py --data_dir ../dataset_raw/20_newsgroups
    python scripts/build_index.py --data_dir ../dataset_raw/20_newsgroups --max_docs 2000
    python scripts/build_index.py --data_dir ../dataset_raw/20_newsgroups --n_clusters 20
"""

import argparse
import os
import sys
import pickle

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data_loader     import load_dataset
from preprocessing   import preprocess_documents
from embedding_model import EmbeddingModel
from vector_store    import VectorStore
from clustering      import FuzzyClusterer, select_n_clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="../dataset_raw/20_newsgroups")
    parser.add_argument("--max_docs",    type=int, default=None)
    parser.add_argument("--n_clusters",  type=int, default=None)
    parser.add_argument("--vector_dir",  default="vector_db")
    parser.add_argument("--cluster_dir", default="cache_store")
    parser.add_argument("--dim",         type=int, default=256, help="LSA embedding dim")
    args = parser.parse_args()

    # 1. Load & preprocess
    print("\n=== STEP 1: Load & Preprocess ===")
    docs = load_dataset(args.data_dir, max_docs=args.max_docs)
    docs = preprocess_documents(docs)
    texts  = [d.clean_text for d in docs]
    labels = [d.category   for d in docs]
    ids    = [d.doc_id     for d in docs]

    # 2. Fit embedding model + embed
    print("\n=== STEP 2: Fit & Embed ===")
    model = EmbeddingModel(dim=args.dim, model_dir=args.cluster_dir)
    model.fit(texts)
    model.save()
    embeddings = model.embed(texts, show_progress=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # 3. Build vector index
    print("\n=== STEP 3: Build Vector Index ===")
    store = VectorStore(dim=model.dim, store_dir=args.vector_dir)
    metadata = [{"doc_id": ids[i], "category": labels[i], "text": texts[i]}
                for i in range(len(docs))]
    store.add(embeddings, metadata)
    store.save()

    # 4. Select cluster count
    print("\n=== STEP 4: Select Cluster Count ===")
    if args.n_clusters:
        n_clusters = args.n_clusters
        print(f"Using user-specified n_clusters = {n_clusters}")
    else:
        max_k = min(30, max(5, len(docs) // 100))
        step  = max(2, max_k // 6)
        k_range = range(5, max_k + 1, step)
        sweep = select_n_clusters(embeddings, k_range=k_range)
        best_idx   = int(np.argmin(sweep["bic"]))
        n_clusters = sweep["k"][best_idx]
        print(f"\nAuto-selected n_clusters = {n_clusters} (lowest BIC)")

    # 5. Fit GMM
    print(f"\n=== STEP 5: Fit GMM (K={n_clusters}) ===")
    clusterer = FuzzyClusterer(n_clusters=n_clusters, cluster_dir=args.cluster_dir)
    clusterer.fit(embeddings)
    clusterer.save()

    # 6. Save per-doc cluster probabilities
    print("\n=== STEP 6: Save Cluster Probabilities ===")
    proba    = clusterer.predict_proba(embeddings)
    dominant = proba.argmax(axis=1)
    with open(os.path.join(args.cluster_dir, "cluster_proba.pkl"), "wb") as f:
        pickle.dump({"doc_ids": ids, "categories": labels,
                     "proba": proba, "dominant": dominant}, f)

    unique, counts = np.unique(dominant, return_counts=True)
    print("\nCluster size distribution:")
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid:>3d}: {cnt:>5d} docs")

    print("\n=== Build complete. Start API: uvicorn api.main:app --reload ===\n")


if __name__ == "__main__":
    main()
