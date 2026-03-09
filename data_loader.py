"""
data_loader.py
--------------
Loads the 20 Newsgroups dataset from the raw directory structure.

Design decisions:
- We walk the directory tree so we stay format-agnostic.
- Label is derived from the parent folder name, which is the original newsgroup name.
- We keep a document ID that encodes (category, filename) for traceability.
- We intentionally load raw bytes and decode with errors='replace' so we never
  crash on the occasional non-UTF-8 message.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Document:
    doc_id: str          # e.g. "sci.space/40100"
    category: str        # newsgroup label
    filename: str        # original filename
    raw_text: str        # unchanged text from disk
    clean_text: str = "" # filled in by preprocessing step


def load_dataset(data_dir: str, max_docs: Optional[int] = None) -> List[Document]:
    """
    Walk data_dir, treating each sub-directory as a category and each file
    inside as a single document.

    Parameters
    ----------
    data_dir : str
        Root directory that contains one sub-folder per newsgroup.
    max_docs : int, optional
        If set, stop after loading this many documents (useful for development).

    Returns
    -------
    List[Document]
    """
    # Resolve dataset path robustly (works whether called from project root or scripts/)
    data_path = Path(data_dir)

    if not data_path.is_absolute():
        project_root = Path(__file__).resolve().parent
        candidate = (project_root / data_path).resolve()

        # If the joined path does not exist (e.g. "../mini_newsgroups"),
        # fall back to looking for the dataset inside the project root.
        if candidate.exists():
            data_path = candidate
        else:
            fallback = (project_root / data_path.name).resolve()
            data_path = fallback

    # If the provided path is the project root, redirect to the actual dataset folder
    if (data_path / "mini_newsgroups").exists():
        data_path = data_path / "mini_newsgroups"
    elif (data_path / "20_newsgroups").exists():
        data_path = data_path / "20_newsgroups"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    docs: List[Document] = []

    # Each immediate child of data_dir is a newsgroup category
    for category_dir in sorted(data_path.iterdir()):
        if not category_dir.is_dir():
            continue
        # Ignore system/project folders that might exist in the repo
        if category_dir.name.startswith(".") or "pycache" in category_dir.name or category_dir.name in {"api", "scripts", "frontend", "vector_db", "cache_store"}:
            continue
        category = category_dir.name

        for doc_file in sorted(category_dir.iterdir()):
            if not doc_file.is_file():
                continue
            # Only accept raw newsgroup message files (they normally have no extension)
            if doc_file.suffix not in {""}:
                continue

            # Decode with replacement so broken encodings don't crash load
            raw_text = doc_file.read_bytes().decode("utf-8", errors="replace")

            docs.append(Document(
                doc_id=f"{category}/{doc_file.name}",
                category=category,
                filename=doc_file.name,
                raw_text=raw_text,
            ))

            if max_docs and len(docs) >= max_docs:
                return docs

    print(f"[data_loader] Loaded {len(docs)} documents from {len(set(d.category for d in docs))} categories.")
    return docs


def get_categories(docs: List[Document]) -> List[str]:
    """Return sorted list of unique category names."""
    return sorted(set(d.category for d in docs))


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "20_newsgroups"
    docs = load_dataset(data_dir, max_docs=100)
    print(f"Sample doc_id : {docs[0].doc_id}")
    print(f"Sample category: {docs[0].category}")
    print(f"Raw text snippet:\n{docs[0].raw_text[:300]}")
