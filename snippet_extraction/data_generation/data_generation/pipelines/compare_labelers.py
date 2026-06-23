"""Compare Haiku 4.5 vs Sonnet 4.6 labels on the same (query_id, doc_id) pairs.

Treats Sonnet 4.6 as ground truth and reports per-row precision/recall/F1
on the kept-unit-id sets, plus aggregate agreement metrics and a few
disagreement samples for spot-checking.

Usage:
    python -m data_generation.pipelines.compare_labelers \
        --gold data_generation/data/labels/firecrawl_v1_sonnet46.jsonl \
        --pred data_generation/data/labels/firecrawl_v1_haiku45.jsonl \
        --samples 10
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Per-MTok USD pricing for the two Bedrock models we compare. Source:
# Anthropic Claude pricing page (Haiku 4.5: $1 in / $5 out; Sonnet 4.6: $3 in / $15 out).
# OpenRouter DeepSeek V4 pricing from openrouter.ai/deepseek (Apr 2026).
PRICE = {
    "us.anthropic.claude-haiku-4-5-20251001-v1:0": (1.0, 5.0),
    "global.anthropic.claude-sonnet-4-6": (3.0, 15.0),
    "deepseek/deepseek-v4-pro": (0.435, 0.87),
    "deepseek/deepseek-v4-flash": (0.0983, 0.1966),
}


def load_keyed(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        out[(r["query_id"], r["doc_id"])] = r
    return out


def prf(pred: set[int], gold: set[int]) -> tuple[float, float, float]:
    if not gold and not pred:
        return 1.0, 1.0, 1.0  # both empty → perfect agreement
    if not pred:
        return 1.0, 0.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0
    tp = len(pred & gold)
    p = tp / len(pred)
    r = tp / len(gold)
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def cost(rows: list[dict]) -> tuple[float, int, int]:
    tok_in = sum(r["tokens_in"] for r in rows)
    tok_out = sum(r["tokens_out"] for r in rows)
    model = rows[0]["labeler_model"] if rows else None
    pi, po = PRICE.get(model, (0.0, 0.0))
    usd = tok_in / 1_000_000 * pi + tok_out / 1_000_000 * po
    return usd, tok_in, tok_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=Path, help="Reference labels (Sonnet 4.6).")
    ap.add_argument("--pred", required=True, type=Path, help="Candidate labels (Haiku 4.5).")
    ap.add_argument(
        "--samples", type=int, default=10, help="Number of worst-disagreement rows to dump."
    )
    args = ap.parse_args()

    gold_rows = load_keyed(args.gold)
    pred_rows = load_keyed(args.pred)
    keys = sorted(set(gold_rows) & set(pred_rows))
    print(f"gold rows ok: {len(gold_rows)}")
    print(f"pred rows ok: {len(pred_rows)}")
    print(f"shared keys : {len(keys)}\n")

    ps, rs, fs, js = [], [], [], []
    gold_sizes, pred_sizes = [], []
    both_empty = pred_empty_only = gold_empty_only = exact_match = 0
    disagreements = []  # (f1, key, gold_set, pred_set, row)

    for k in keys:
        g_row = gold_rows[k]
        p_row = pred_rows[k]
        g = set(g_row["relevant_unit_ids"] or [])
        p = set(p_row["relevant_unit_ids"] or [])
        gold_sizes.append(len(g))
        pred_sizes.append(len(p))
        if not g and not p:
            both_empty += 1
        if not p and g:
            gold_empty_only += 1
        if not g and p:
            pred_empty_only += 1
        if g == p:
            exact_match += 1
        pr, rc, f1 = prf(p, g)
        ps.append(pr)
        rs.append(rc)
        fs.append(f1)
        js.append(jaccard(p, g))
        disagreements.append((f1, k, g, p, g_row))

    n = len(keys)
    pred_all = list(pred_rows.values())
    gold_all = list(gold_rows.values())
    pred_usd, pred_in, pred_out = cost(pred_all)
    gold_usd, gold_in, gold_out = cost(gold_all)

    print("=== Agreement (Haiku-as-pred vs Sonnet-as-gold) ===")
    print(f"  exact set match : {exact_match}/{n} ({exact_match / n:.1%})")
    print(f"  both empty      : {both_empty}/{n} ({both_empty / n:.1%})")
    print(
        f"  Haiku=[], Sonnet!=[] (missed positives) : {gold_empty_only}/{n} ({gold_empty_only / n:.1%})"
    )
    print(
        f"  Sonnet=[], Haiku!=[] (extra positives)  : {pred_empty_only}/{n} ({pred_empty_only / n:.1%})"
    )
    print()
    print("=== Per-row PRF (mean) ===")
    print(f"  precision : {statistics.mean(ps):.3f}")
    print(f"  recall    : {statistics.mean(rs):.3f}")
    print(f"  F1        : {statistics.mean(fs):.3f}")
    print(f"  jaccard   : {statistics.mean(js):.3f}")
    print()
    print("=== Per-row PRF (median) ===")
    print(f"  precision : {statistics.median(ps):.3f}")
    print(f"  recall    : {statistics.median(rs):.3f}")
    print(f"  F1        : {statistics.median(fs):.3f}")
    print()
    print("=== Kept-set sizes ===")
    print(
        f"  Sonnet mean={statistics.mean(gold_sizes):.2f} median={statistics.median(gold_sizes)} max={max(gold_sizes)}"
    )
    print(
        f"  Haiku  mean={statistics.mean(pred_sizes):.2f} median={statistics.median(pred_sizes)} max={max(pred_sizes)}"
    )
    print()
    print("=== Cost (on the rows in each file, not just shared) ===")
    print(
        f"  Sonnet rows={len(gold_all):>4}  tok_in={gold_in:>9,}  tok_out={gold_out:>6,}  USD={gold_usd:.4f}"
    )
    print(
        f"  Haiku  rows={len(pred_all):>4}  tok_in={pred_in:>9,}  tok_out={pred_out:>6,}  USD={pred_usd:.4f}"
    )
    if pred_usd > 0:
        print(f"  ratio Sonnet/Haiku USD: {gold_usd / pred_usd:.2f}x")
    print()

    # F1 distribution buckets
    buckets = [
        (0.0, 0.001),
        (0.001, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 0.999),
        (0.999, 1.0001),
    ]
    print("=== F1 distribution ===")
    for lo, hi in buckets:
        c = sum(1 for f in fs if lo <= f < hi)
        label = f"[{lo:.2f},{hi:.2f})" if hi <= 1 else f"[{lo:.2f},1.00]"
        print(f"  {label}  {c:>4}  ({c / n:.1%})")
    print()

    # Worst disagreements (lowest F1, excluding both-empty perfect matches)
    print(f"=== Worst {args.samples} disagreements (lowest F1) ===")
    disagreements.sort(key=lambda x: (x[0], -abs(len(x[2]) - len(x[3]))))
    shown = 0
    for f1, key, g, p, row in disagreements:
        if shown >= args.samples:
            break
        qid, did = key
        only_g = sorted(g - p)
        only_p = sorted(p - g)
        common = sorted(g & p)
        print(f"  qid={qid} did={did[:80]}")
        print(f"    query   : {row['query'][:120]}")
        print(
            f"    n_units : {row['n_units']}  Sonnet kept={len(g)}  Haiku kept={len(p)}  F1={f1:.2f}"
        )
        print(f"    common  : {common}")
        print(f"    only Sonnet (missed by Haiku): {only_g}")
        print(f"    only Haiku  (extra vs Sonnet): {only_p}")
        shown += 1


if __name__ == "__main__":
    main()
