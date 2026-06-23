"""A/B harness: compare two labeler prompt versions on the same pairs.

Runs the SAME sampled (query, units) pairs through both labeler prompts on the
same Bedrock model, so the only variable is the prompt. The point is to check
whether a prompt change disturbs the strict ``relevant_unit_ids`` set we
already trust. Same version in both arms = self-consistency run (temp=0 noise
floor).

Outputs (visualizer-loadable label rows) into --out-dir:
  - ab_arm_a_<va>.jsonl          relevant_unit_ids = arm A strict answer
  - ab_arm_b_<vb>.jsonl          relevant_unit_ids = arm B strict answer
  - ab_arm_b_v3_context.jsonl    arm B context set as relevant (visual diff;
                                 only when arm B is v3)

Then prints an agreement report comparing the two *strict* sets.

Usage:
    python -m data_generation.pipelines.ab_labeler \
        --in data_generation/data/labels/fc2_02_segmented.jsonl \
        --arm-a v2 --arm-b v3 --n 50 --seed 13 --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
from pathlib import Path

from ..core.labeling import PROMPTS, make_async_labeler
from ..core.paths import LABELS_DIR


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _row(
    src: dict,
    *,
    relevant: list[int],
    labeler_model: str,
    prompt_version: str,
    prompt_hash: str,
    status: str,
    tokens_in: int,
    tokens_out: int,
    latency_s: float,
    context: list[int] | None = None,
) -> dict:
    out = {
        "query_id": src["query_id"],
        "doc_id": src["doc_id"],
        "origin": src.get("origin", "firecrawl"),
        "labeler_provider": "bedrock",
        "labeler_model": labeler_model,
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash,
        "status": status,
        "query": src["query"],
        "units": src["units"],
        "relevant_unit_ids": relevant,
        "n_units": len(src["units"]),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "latency_s": round(latency_s, 3),
        "timestamp": _now(),
    }
    if context is not None:
        out["context_unit_ids"] = context
    return out


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


async def main_async(args) -> None:
    rows = [json.loads(l) for l in args.in_path.open()]
    rows = [r for r in rows if r.get("status") == "ok" and r.get("units") and r.get("query")]
    random.Random(args.seed).shuffle(rows)
    sample = rows[: args.n]

    labeler_a = make_async_labeler("bedrock", version=args.arm_a)
    labeler_b = make_async_labeler("bedrock", version=args.arm_b)
    va, ha = args.arm_a, PROMPTS[args.arm_a].prompt_hash()
    vb, hb = args.arm_b, PROMPTS[args.arm_b].prompt_hash()
    model = labeler_a.deployment
    self_consistency = args.arm_a == args.arm_b

    print(f"in={args.in_path.name}  eligible={len(rows)}  sampled={len(sample)}  seed={args.seed}")
    print(f"arm A: {va} (hash={ha})  |  arm B: {vb} (hash={hb})")
    if self_consistency:
        print("mode : SELF-CONSISTENCY (same prompt both arms) — measures temp=0 noise floor")
    else:
        print("mode : CROSS (different prompts) — measures prompt effect")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = [None] * len(sample)  # type: ignore

    async def run_pair(i: int, src: dict) -> None:
        async with sem:
            ra, rb = await asyncio.gather(
                labeler_a.label(src["query"], src["units"]),
                labeler_b.label(src["query"], src["units"]),
            )
        results[i] = {"src": src, "a": ra, "b": rb}
        done = sum(1 for x in results if x is not None)
        ctx_b = rb.context_unit_ids
        ctx_str = f"+ctx{len(ctx_b)}" if ctx_b is not None else ""
        print(
            f"[{done:>3}/{len(sample)}] qid={src['query_id']} "
            f"A:{ra.status}/{len(ra.relevant_unit_ids or [])} "
            f"B:{rb.status}/{len(rb.relevant_unit_ids or [])}{ctx_str}",
            flush=True,
        )

    await asyncio.gather(*(run_pair(i, s) for i, s in enumerate(sample)))

    # Write visualizer-loadable outputs, one file per arm.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fa = args.out_dir / f"ab_arm_a_{va}.jsonl"
    fb = args.out_dir / f"ab_arm_b_{vb}.jsonl"
    written = [fa, fb]
    with fa.open("w") as oa, fb.open("w") as ob:
        for r in results:
            src, a, b = r["src"], r["a"], r["b"]
            oa.write(
                json.dumps(
                    _row(
                        src,
                        relevant=a.relevant_unit_ids or [],
                        labeler_model=model,
                        prompt_version=va,
                        prompt_hash=ha,
                        status=a.status,
                        tokens_in=a.tokens_in,
                        tokens_out=a.tokens_out,
                        latency_s=a.latency_s,
                        context=a.context_unit_ids,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
            ob.write(
                json.dumps(
                    _row(
                        src,
                        relevant=b.relevant_unit_ids or [],
                        labeler_model=model,
                        prompt_version=vb,
                        prompt_hash=hb,
                        status=b.status,
                        tokens_in=b.tokens_in,
                        tokens_out=b.tokens_out,
                        latency_s=b.latency_s,
                        context=b.context_unit_ids,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
    # If arm B is v3, also emit a context-as-relevant file for visual diffing.
    if args.arm_b == "v3":
        fb_ctx = args.out_dir / "ab_arm_b_v3_context.jsonl"
        with fb_ctx.open("w") as oc:
            for r in results:
                src, b = r["src"], r["b"]
                oc.write(
                    json.dumps(
                        _row(
                            src,
                            relevant=b.context_unit_ids or [],
                            labeler_model=model,
                            prompt_version=vb + "_context",
                            prompt_hash=hb,
                            status=b.status,
                            tokens_in=b.tokens_in,
                            tokens_out=b.tokens_out,
                            latency_s=b.latency_s,
                        ),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        written.append(fb_ctx)

    # ---- Agreement report (arm A strict vs arm B strict) ----
    ok = [r for r in results if r["a"].status == "ok" and r["b"].status == "ok"]
    n_ok = len(ok)
    exact = both_empty = a_sizes = b_sizes = 0
    jac_sum = 0.0
    tok_in = tok_out = 0
    # context stats only meaningful when an arm is v3
    ctx_sizes = ctx_expansion = superset_violations = 0
    have_ctx = False
    for r in ok:
        sa = set(r["a"].relevant_unit_ids or [])
        sb = set(r["b"].relevant_unit_ids or [])
        if sa == sb:
            exact += 1
        if not sa and not sb:
            both_empty += 1
        jac_sum += _jaccard(sa, sb)
        a_sizes += len(sa)
        b_sizes += len(sb)
        tok_in += r["a"].tokens_in + r["b"].tokens_in
        tok_out += r["a"].tokens_out + r["b"].tokens_out
        ctx_b = r["b"].context_unit_ids
        if ctx_b is not None:
            have_ctx = True
            sc = set(ctx_b)
            ctx_sizes += len(sc)
            ctx_expansion += len(sc) - len(sb)
            if not sb.issubset(sc):
                superset_violations += 1

    print()
    print("=" * 64)
    label = "self-consistency" if self_consistency else "prompt effect"
    print(f"comparison: {va} vs {vb}  ({label})")
    print(f"pairs ok (both arms succeeded): {n_ok}/{len(results)}")
    if n_ok:
        print(f"strict-set EXACT match (A==B)  : {exact}/{n_ok}  ({exact / n_ok:.0%})")
        print(f"strict-set mean Jaccard (A,B)  : {jac_sum / n_ok:.3f}")
        print(f"both-empty pairs               : {both_empty}/{n_ok}")
        print(f"mean |strict| A / B            : {a_sizes / n_ok:.2f} / {b_sizes / n_ok:.2f}")
        if have_ctx:
            print(f"mean |context| (v3 arm)        : {ctx_sizes / n_ok:.2f}")
            print(f"mean context expansion (ctx-ans): {ctx_expansion / n_ok:.2f}")
            print(
                f"superset violations (ctx⊉ans)  : {superset_violations}  (enforced in code, should be 0)"
            )
        print(f"tokens in / out (both arms)    : {tok_in:,} / {tok_out:,}")
    print("=" * 64)
    print("wrote:")
    for p in written:
        print(f"  {p}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=LABELS_DIR / "ab_test")
    ap.add_argument("--arm-a", choices=sorted(PROMPTS), default="v2")
    ap.add_argument(
        "--arm-b",
        choices=sorted(PROMPTS),
        default="v2",
        help="Same as --arm-a runs self-consistency (default). Use v3 for the cross comparison.",
    )
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--concurrency", type=int, default=5)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
