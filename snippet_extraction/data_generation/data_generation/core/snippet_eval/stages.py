"""Four pipeline stages, each reading from / writing to JSONL caches.

Stages are restartable: re-running skips work that's already cached. Expensive
work (provider calls, Firecrawl scrapes, labeler calls) is persisted to disk so
re-scoring with a different metric or labeler prompt is free.

    collect_provider_results(queries, provider, out_path)
    fetch_bodies(provider_results_paths, fc, out_path)
    label_bodies(queries, bodies, labeler, out_path)
    score(queries, provider_results_paths, bodies, gold, out_path)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from ..labeling.clients import AsyncLabelerClient
from ..retrievers.firecrawl import FirecrawlSearch
from ..segment import segment
from .metrics import join_snippets, token_prf
from .providers import ProviderAdapter, Surfaced
from .store import (
    append_jsonl,
    load_bodies,
    load_gold,
    load_provider_results,
    read_jsonl,
    write_jsonl,
)


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Stage 1: collect provider results
# ---------------------------------------------------------------------------


async def collect_provider_results(
    queries: list[dict],
    provider: ProviderAdapter,
    out_path: Path,
    *,
    resume: bool = True,
) -> None:
    """For each query, call ``provider.search`` and persist the surfaced docs.

    Row schema (one per (provider, query_id)):
        provider, query_id, query, intent, fetched_at,
        docs: [{url, snippets, score, title}, ...],
        error: str | None
    """
    done = load_provider_results(out_path) if resume else {}
    for row in queries:
        qid = row["qid"]
        key = (provider.name, qid)
        if key in done:
            continue
        rec: dict[str, Any] = {
            "provider": provider.name,
            "query_id": qid,
            "query": row["query"],
            "intent": row.get("intent"),
            "fetched_at": _now(),
            "docs": [],
            "error": None,
        }
        try:
            surfaced: list[Surfaced] = await provider.search(row["query"])
            rec["docs"] = [
                {
                    "url": s.url,
                    "snippets": list(s.snippets),
                    "score": s.score,
                    "title": s.title,
                }
                for s in surfaced
            ]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        append_jsonl(out_path, rec)


# ---------------------------------------------------------------------------
# Stage 2: fetch canonical bodies
# ---------------------------------------------------------------------------


def _all_urls(provider_paths: list[Path]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in provider_paths:
        for row in read_jsonl(p):
            for d in row.get("docs", []):
                u = d.get("url")
                if u and u not in seen:
                    seen.add(u)
                    out.append(u)
    return out


async def fetch_bodies(
    provider_paths: list[Path],
    fetcher: FirecrawlSearch,
    out_path: Path,
    *,
    resume: bool = True,
) -> None:
    """Scrape every URL surfaced by any provider exactly once.

    Row schema (one per url):
        url, markdown, status ("ok" | "empty" | "error:<msg>"), fetched_at
    """
    done = load_bodies(out_path) if resume else {}
    for url in _all_urls(provider_paths):
        if url in done:
            continue
        markdown = ""
        status = "ok"
        try:
            markdown = await fetcher.scrape(url)
            if not markdown.strip():
                status = "empty"
        except Exception as e:
            status = f"error:{type(e).__name__}"
        append_jsonl(
            out_path,
            {"url": url, "markdown": markdown, "status": status, "fetched_at": _now()},
        )


# ---------------------------------------------------------------------------
# Stage 3: label bodies -> gold
# ---------------------------------------------------------------------------


def _query_url_pairs(provider_paths: list[Path]) -> list[tuple[str, str, str]]:
    """De-duped (query_id, query, url) triples across all provider files."""
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str, str]] = []
    for p in provider_paths:
        for row in read_jsonl(p):
            qid = row["query_id"]
            q = row["query"]
            for d in row.get("docs", []):
                u = d.get("url")
                if not u:
                    continue
                k = (qid, u)
                if k in seen:
                    continue
                seen.add(k)
                out.append((qid, q, u))
    return out


async def label_bodies(
    provider_paths: list[Path],
    bodies_path: Path,
    labeler: AsyncLabelerClient,
    out_path: Path,
    *,
    resume: bool = True,
) -> None:
    """For each (query_id, url), segment the body and ask the labeler.

    Row schema (one per (query_id, url)):
        query_id, query, url,
        units: [str], gold_unit_ids: [int], gold_text: str,
        status: "ok" | "no_body" | "no_units" | "<labeler_status>",
        labeler_provider, labeler_deployment, tokens_in, tokens_out, latency_s,
        labeled_at
    """
    bodies = load_bodies(bodies_path)
    done = load_gold(out_path) if resume else {}
    for qid, query, url in _query_url_pairs(provider_paths):
        key = (qid, url)
        if key in done:
            continue
        body_row = bodies.get(url)
        markdown = (body_row or {}).get("markdown", "")
        base: dict[str, Any] = {
            "query_id": qid,
            "query": query,
            "url": url,
            "units": [],
            "gold_unit_ids": [],
            "gold_text": "",
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_s": 0.0,
            "labeler_provider": getattr(labeler, "provider", None),
            "labeler_deployment": getattr(labeler, "deployment", None),
            "labeled_at": _now(),
        }
        if not markdown.strip():
            append_jsonl(out_path, {**base, "status": "no_body"})
            continue
        units = segment(markdown)
        if not units:
            append_jsonl(out_path, {**base, "status": "no_units"})
            continue
        res = await labeler.label(query, units)
        if res.status != "ok" or res.relevant_unit_ids is None:
            append_jsonl(
                out_path,
                {
                    **base,
                    "units": units,
                    "status": res.status,
                    "tokens_in": res.tokens_in,
                    "tokens_out": res.tokens_out,
                    "latency_s": res.latency_s,
                },
            )
            continue
        gold_units = [units[i] for i in res.relevant_unit_ids]
        append_jsonl(
            out_path,
            {
                **base,
                "units": units,
                "gold_unit_ids": list(res.relevant_unit_ids),
                "gold_text": join_snippets(gold_units),
                "status": "ok",
                "tokens_in": res.tokens_in,
                "tokens_out": res.tokens_out,
                "latency_s": res.latency_s,
            },
        )


# ---------------------------------------------------------------------------
# Stage 4: score (pure function over the three caches)
# ---------------------------------------------------------------------------


def score(
    queries: list[dict],
    provider_paths: list[Path],
    bodies_path: Path,
    gold_path: Path,
    out_path: Path,
) -> list[dict]:
    """Build the per-doc eval JSONL from cached stages.

    Row schema (one per (provider, query_id, url)):
        provider, query_id, query, intent, url,
        snippets: [str], pred_text: str,
        units: [str], gold_unit_ids: [int], gold_text: str,
        precision, recall, f1, n_pred_tokens, n_gold_tokens, n_overlap,
        body_status, gold_status, has_body, has_gold
    """
    bodies = load_bodies(bodies_path)
    gold = load_gold(gold_path)
    qmeta = {q["qid"]: q for q in queries}

    rows: list[dict] = []
    for p in provider_paths:
        for pr in read_jsonl(p):
            provider = pr["provider"]
            qid = pr["query_id"]
            query = pr["query"]
            intent = pr.get("intent") or qmeta.get(qid, {}).get("intent")
            for d in pr.get("docs", []):
                url = d.get("url", "")
                snippets = d.get("snippets") or []
                pred_text = join_snippets(snippets)
                body_status = (bodies.get(url) or {}).get("status", "missing")
                gold_row = gold.get((qid, url))
                gold_status = (gold_row or {}).get("status", "missing")
                units = (gold_row or {}).get("units") or []
                gold_unit_ids = (gold_row or {}).get("gold_unit_ids") or []
                gold_text = (gold_row or {}).get("gold_text", "")
                prf = token_prf(pred_text, gold_text)
                rows.append(
                    {
                        "provider": provider,
                        "query_id": qid,
                        "query": query,
                        "intent": intent,
                        "url": url,
                        "snippets": snippets,
                        "pred_text": pred_text,
                        "units": units,
                        "gold_unit_ids": gold_unit_ids,
                        "gold_text": gold_text,
                        "precision": prf.precision,
                        "recall": prf.recall,
                        "f1": prf.f1,
                        "n_pred_tokens": prf.n_pred,
                        "n_gold_tokens": prf.n_gold,
                        "n_overlap": prf.n_overlap,
                        "body_status": body_status,
                        "gold_status": gold_status,
                        "has_body": body_status == "ok",
                        "has_gold": gold_status == "ok",
                    }
                )
    write_jsonl(out_path, rows)
    return rows
