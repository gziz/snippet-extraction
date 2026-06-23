"""Append-only JSONL caches for the snippet-extraction eval.

Three stage outputs, each keyed so a re-run skips work already done:

    provider_results.jsonl   key: (provider, query_id)
    bodies.jsonl             key: url
    gold.jsonl               key: (query_id, url)

A fourth file is *derived* from these (pure function, cheap to re-run):

    doc_eval.jsonl           key: (provider, query_id, url)

The keying choices reflect what is actually expensive:

  - A provider call returns *all* docs for a query, so we key by
    (provider, query_id) — same query asked twice would hit the cache.
  - A scrape is per-URL and the same URL is often surfaced by multiple
    providers / multiple queries — key by url alone.
  - The labeler runs over (query, body); for one canonical body the judgement
    only depends on the query, so key by (query_id, url).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Overwrite ``path`` with the given rows (used by the derived score step)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- typed loaders / key extractors ----------------------------------------


def load_provider_results(path: Path) -> dict[tuple[str, str], dict]:
    """key = (provider, query_id) -> full row."""
    return {(r["provider"], r["query_id"]): r for r in read_jsonl(path)}


def load_bodies(path: Path) -> dict[str, dict]:
    """key = url -> {url, markdown, status, fetched_at}."""
    return {r["url"]: r for r in read_jsonl(path)}


def load_gold(path: Path) -> dict[tuple[str, str], dict]:
    """key = (query_id, url) -> full gold row."""
    return {(r["query_id"], r["url"]): r for r in read_jsonl(path)}
