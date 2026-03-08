"""
preprocessing.py
----------------
Cleans raw newsgroup messages into model-ready text.

Design decisions:
------------------
1. HEADER REMOVAL
   Newsgroup posts include metadata headers (Path, Message-ID, Xref, NNTP-Posting-Host,
   Organization, Lines, etc.) that carry routing information, not semantics.
   We strip the entire header block (everything before the first blank line) but
   KEEP the Subject line because it is a compact semantic summary of the post.

2. MINIMAL TOKENISATION
   We do NOT lemmatise or stem. Transformer sentence models work on sub-word tokens
   and handle morphological variation internally. Aggressive stemming would discard
   information the model can use.

3. LOWERCASING
   Uniform casing prevents the same word being treated differently. Sentence-
   transformer models are cased but lowercasing is harmless here.

4. EMAIL/URL REMOVAL
   Emails and URLs add noise without semantic value in the context of topical search.

5. WHITESPACE NORMALISATION
   Multiple consecutive whitespace characters are collapsed to a single space so the
   model context window is used efficiently.

6. MINIMUM LENGTH GUARD
   After cleaning we discard documents shorter than MIN_CHARS because they are
   likely empty or contain only noise; they would produce noisy embeddings.
"""

import re
from typing import List
from data_loader import Document

# Headers to strip (case-insensitive prefix match inside the header block)
# "From" is also stripped as it is a personal identifier, not a topical signal.
_STRIP_HEADERS = {
    "xref", "path", "from", "message-id", "nntp-posting-host",
    "organization", "lines", "newsgroups", "date", "article-i.d.",
    "expires", "distribution", "reply-to", "references",
    "sender", "followup-to",
}

_EMAIL_RE   = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_URL_RE     = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE   = re.compile(r"\s+")
_PUNCT_RE   = re.compile(r"[^\w\s]")   # optional – currently not applied (see below)

MIN_CHARS = 40  # documents below this length are discarded


def _extract_subject_and_body(raw: str) -> str:
    """
    Split a newsgroup message into header block + body.
    Return the Subject value (if found) concatenated with the body text.
    """
    # The header block ends at the first blank line
    lines = raw.splitlines()
    subject = ""
    body_lines: List[str] = []
    in_header = True

    for line in lines:
        if in_header:
            if line.strip() == "":
                # Blank line marks end of header block
                in_header = False
                continue
            # Check if this header should be kept (Subject only)
            lower = line.lower()
            if lower.startswith("subject:"):
                # Keep the subject text (strip the "Subject:" prefix)
                subject = line[8:].strip()
            # All other headers are silently dropped
        else:
            body_lines.append(line)

    body = "\n".join(body_lines)
    if subject:
        return f"{subject}\n{body}"
    return body


def clean_document(raw: str) -> str:
    """
    Full cleaning pipeline for one raw newsgroup document.
    Returns cleaned text string.
    """
    # Step 1 – keep Subject + body, drop routing headers
    text = _extract_subject_and_body(raw)

    # Step 2 – lowercase
    text = text.lower()

    # Step 3 – remove email addresses (personal identifiers, not topical)
    text = _EMAIL_RE.sub(" ", text)

    # Step 4 – remove URLs
    text = _URL_RE.sub(" ", text)

    # Step 5 – normalise whitespace (tabs, newlines → single space)
    text = _SPACE_RE.sub(" ", text).strip()

    return text


def preprocess_documents(docs: List[Document]) -> List[Document]:
    """
    Apply clean_document to every Document in-place (populates clean_text).
    Returns the same list, filtering out documents that are too short after cleaning.
    """
    valid: List[Document] = []
    short_count = 0

    for doc in docs:
        doc.clean_text = clean_document(doc.raw_text)
        if len(doc.clean_text) >= MIN_CHARS:
            valid.append(doc)
        else:
            short_count += 1

    print(f"[preprocessing] {len(valid)} documents kept, {short_count} discarded (too short).")
    return valid


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_dataset

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../dataset_raw/20_newsgroups"
    docs = load_dataset(data_dir, max_docs=5)
    docs = preprocess_documents(docs)
    for d in docs[:2]:
        print(f"=== {d.doc_id} ===")
        print(d.clean_text[:400])
        print()
