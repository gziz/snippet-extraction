"""Aggregation over doc-level eval rows.

The per-doc rows are produced by ``stages.score`` (pure function over the
collect/fetch/labeler caches). This module turns those rows into per-provider
and per-(provider, intent) summary tables.

The "scored" subset = rows where the canonical body was fetched AND the labeler
ran successfully. Rows that failed at the body or labeler stage are reported but
excluded from precision/recall/F1 means.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass


@dataclass
class Aggregate:
    key: str
    n_docs: int
    n_scored: int
    n_no_body: int
    n_no_gold: int  # body ok but labeler failed / returned no_units
    n_gold_empty: int  # labeler ran ok but found no answer in this doc
    precision: float
    recall: float
    f1: float
    median_pred_tokens: float
    median_gold_tokens: float


def _mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else float("nan")


def aggregate(rows: list[dict], key: str) -> Aggregate:
    scored = [r for r in rows if r["has_gold"]]
    return Aggregate(
        key=key,
        n_docs=len(rows),
        n_scored=len(scored),
        n_no_body=sum(1 for r in rows if not r["has_body"]),
        n_no_gold=sum(1 for r in rows if r["has_body"] and not r["has_gold"]),
        n_gold_empty=sum(1 for r in scored if r["n_gold_tokens"] == 0),
        precision=_mean([r["precision"] for r in scored]),
        recall=_mean([r["recall"] for r in scored]),
        f1=_mean([r["f1"] for r in scored]),
        median_pred_tokens=_median([r["n_pred_tokens"] for r in scored]),
        median_gold_tokens=_median([r["n_gold_tokens"] for r in scored]),
    )


def summarize(rows: list[dict]) -> dict:
    by_provider: dict[str, list[dict]] = defaultdict(list)
    by_provider_intent: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_provider[r["provider"]].append(r)
        by_provider_intent[(r["provider"], r.get("intent") or "?")].append(r)
    return {
        "overall": [asdict(aggregate(rs, p)) for p, rs in sorted(by_provider.items())],
        "by_intent": [
            asdict(aggregate(rs, f"{p}/{intent}"))
            for (p, intent), rs in sorted(by_provider_intent.items())
        ],
    }


def format_summary(summary: dict) -> str:
    hdr = (
        f"{'key':<22} {'docs':>5} {'scor':>5} {'noBd':>5} {'noG':>5} "
        f"{'gold∅':>6} {'P':>6} {'R':>6} {'F1':>6} {'pTok':>6} {'gTok':>6}"
    )
    out: list[str] = ["=== OVERALL (per provider) ===", hdr]
    for a in summary["overall"]:
        out.append(_row(a))
    out.append("")
    out.append("=== BY INTENT (provider/intent) ===")
    out.append(hdr)
    for a in summary["by_intent"]:
        out.append(_row(a))
    return "\n".join(out)


def _row(a: dict) -> str:
    return (
        f"{a['key']:<22} {a['n_docs']:>5} {a['n_scored']:>5} {a['n_no_body']:>5} "
        f"{a['n_no_gold']:>5} {a['n_gold_empty']:>6} "
        f"{a['precision']:>6.3f} {a['recall']:>6.3f} {a['f1']:>6.3f} "
        f"{a['median_pred_tokens']:>6.0f} {a['median_gold_tokens']:>6.0f}"
    )
