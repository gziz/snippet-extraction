"""Random-access doc body fetcher using a one-time byte-offset index.

The index is built once with `build_doc_index.py` and then used to seek
directly to any docid's line in the 22GB docs.tsv file.
"""

from __future__ import annotations

import pickle
from pathlib import Path


def load_index(index_path: Path) -> dict[str, int]:
    """Load the docid -> byte_offset index."""
    with index_path.open("rb") as f:
        return pickle.load(f)


def fetch_docs(
    docs_tsv: Path,
    needed: set[str],
    index: dict[str, int],
) -> dict[str, tuple[str, str, str]]:
    """Return {docid: (url, title, body)} for the requested docids via random access."""
    out: dict[str, tuple[str, str, str]] = {}
    with docs_tsv.open("rb") as f:
        for docid in needed:
            offset = index.get(docid)
            if offset is None:
                continue
            f.seek(offset)
            line = f.readline().decode("utf-8", errors="replace").rstrip("\n")
            parts = line.split("\t")
            if len(parts) != 4 or parts[0] != docid:
                continue
            _, url, title, body = parts
            out[docid] = (url, title, body)
    return out
