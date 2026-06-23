"""The canonical label-row schema and JSONL store.

Two row shapes exist in the corpus, both defined here and nowhere else:

- **scrape rows** (``make_scrape_row``): one per (query, document) produced by
  a scrape driver. Carries the document (url/title/body/units) and query
  metadata; ``relevant_unit_ids`` is empty until a labeler runs.
- **label rows** (``make_label_row``): one per labeled (query, document) pair.
  Carries the labeler identity (provider/model/prompt_version/prompt_hash) and
  token/latency accounting.

Conventions:

- Files are **append-only** JSONL; resume logic skips work already present.
- ``query_id`` is always compared **as a string**. Sources disagree (MS MARCO
  uses ints, synthetic qids are strings); coercing at the comparison boundary
  kills that bug class.
- Label dedupe key: ``(query_id, doc_id, labeler_provider, labeler_model,
  prompt_version)``.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------


def read_jsonl(path: Path, *, skip_bad: bool = False) -> Iterator[dict[str, Any]]:
    """Yield rows; ``skip_bad=True`` tolerates truncated/corrupt lines."""
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not skip_bad:
                    raise


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Row constructors
# ---------------------------------------------------------------------------


def make_label_row(
    *,
    query_id: int | str,
    doc_id: str,
    query: str,
    body: str,
    units: list[str],
    relevant_unit_ids: list[int] | None,
    labeler_provider: str,
    labeler_model: str,
    prompt_version: str,
    prompt_hash: str,
    tokens_in: int,
    tokens_out: int,
    latency_s: float,
    origin: str,
    status: str = "ok",
) -> dict:
    return {
        "query_id": query_id,
        "doc_id": doc_id,
        "origin": origin,
        "labeler_provider": labeler_provider,
        "labeler_model": labeler_model,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "status": status,
        "query": query,
        "body": body,
        "units": units,
        "relevant_unit_ids": relevant_unit_ids,
        "n_units": len(units),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_s": round(latency_s, 3),
        "timestamp": utc_now(),
    }


def append_label(path: Path, **kwargs) -> None:
    """Build a label row and append it (the old ``storage.append_label``)."""
    append_jsonl(path, make_label_row(**kwargs))


def make_scrape_row(
    *,
    qid: str,
    query: str,
    topic: str | None,
    intent: str | None,
    url: str,
    title: str | None,
    description: str | None,
    body: str,
    units: list[str],
    position: int | None,
    origin: str,
    fetcher_name: str,
    fetcher_version: str,
    skipped_reason: str | None,
) -> dict:
    # labeler_provider/labeler_model record the *fetcher* here (e.g. tavily) so
    # the row is self-describing before an LLM labeler ever runs on it.
    return {
        "query_id": qid,
        "doc_id": url,
        "origin": origin,
        "labeler_provider": origin,
        "labeler_model": fetcher_name,
        "prompt_version": fetcher_version,
        "status": skipped_reason or "ok",
        "query": query,
        "topic": topic,
        "intent": intent,
        "url": url,
        "title": title,
        "description": description,
        "body": body,
        "units": units,
        "relevant_unit_ids": [],
        "n_units": len(units),
        "n_kept": 0,
        "position": position,
        "timestamp": utc_now(),
    }


# ---------------------------------------------------------------------------
# Resume / dedupe keys
# ---------------------------------------------------------------------------


def load_done_qids(path: Path) -> set[str]:
    """query_ids with at least one row in ``path`` (scrape-driver resume)."""
    qids: set[str] = set()
    if not path.exists():
        return qids
    for row in read_jsonl(path, skip_bad=True):
        if "query_id" in row:
            qids.add(str(row["query_id"]))
    return qids


def load_done_keys(
    path: Path,
    labeler_provider: str,
    labeler_model: str,
    prompt_version: str,
) -> set[tuple[str, str]]:
    """(qid, did) pairs already labeled with this (provider, model, prompt_version).

    Keys are string-coerced; callers must coerce their side too.
    """
    if not path.exists():
        return set()
    done: set[tuple[str, str]] = set()
    for row in read_jsonl(path, skip_bad=True):
        # Legacy rows without labeler_provider are assumed to be "azure"
        # (the only provider in use before this field existed).
        row_provider = row.get("labeler_provider", "azure")
        if (
            row_provider == labeler_provider
            and row.get("labeler_model") == labeler_model
            and row.get("prompt_version") == prompt_version
        ):
            done.add((str(row["query_id"]), row["doc_id"]))
    return done
