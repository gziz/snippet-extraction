"""Ad-hoc speed benchmark for a SentenceCompressor checkpoint.

Times forward passes on a held-out split with proper CUDA sync. Reports
per-doc latency stats and throughput (docs/s, tokens/s).

Usage:
    .venv/bin/python -m snippets_training.bench_speed \\
        --ckpt checkpoints/run6 \\
        --data data/modernbert_8k_v1/test.jsonl \\
        --limit 50 --warmup 3
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from transformers import AutoTokenizer

from .dataset import _build_body_and_spans
from .model import SentenceCompressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--base", default="answerdotai/ModernBERT-base")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--bf16", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available(), "This benchmark requires a GPU."
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Device: {torch.cuda.get_device_name(device)}")

    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    model = SentenceCompressor(base=args.base, attn_implementation=args.attn_impl).to(device)
    state = torch.load(f"{args.ckpt}/model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"Loaded {args.ckpt} (attn_impl={args.attn_impl})")

    rows = []
    with open(args.data) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= args.limit + args.warmup:
                break
    print(f"Loaded {len(rows)} rows from {args.data}")

    # Pre-tokenize so we time the forward pass, not python/tokenizer overhead.
    encs = []
    for d in rows:
        body, _ = _build_body_and_spans(d["units"])
        enc = tok(
            d["query"],
            body,
            truncation="only_second",
            max_length=args.max_length,
            return_tensors="pt",
        )
        encs.append(
            {
                "input_ids": enc["input_ids"].to(device),
                "attention_mask": enc["attention_mask"].to(device),
                "n_tokens": int(enc["input_ids"].shape[1]),
                "n_units": d.get("n_units", len(d["units"])),
            }
        )

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    autocast = torch.autocast(device_type="cuda", dtype=autocast_dtype)

    # Warm-up: first few forwards are slower (CUDA kernel selection, allocator).
    print(f"Warming up ({args.warmup} forwards) ...")
    with torch.no_grad(), autocast:
        for i in range(args.warmup):
            _ = model(input_ids=encs[i]["input_ids"], attention_mask=encs[i]["attention_mask"])
    torch.cuda.synchronize()

    # Timed loop.
    per_doc_ms: list[float] = []
    n_tokens: list[int] = []
    n_units: list[int] = []
    timed = encs[args.warmup : args.warmup + args.limit]
    print(f"Timing {len(timed)} forwards ...")
    with torch.no_grad(), autocast:
        wall_start = time.perf_counter()
        for e in timed:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(input_ids=e["input_ids"], attention_mask=e["attention_mask"])
            torch.cuda.synchronize()
            per_doc_ms.append((time.perf_counter() - t0) * 1000.0)
            n_tokens.append(e["n_tokens"])
            n_units.append(e["n_units"])
        wall_total = time.perf_counter() - wall_start

    tot_tokens = sum(n_tokens)
    print("\n=== Speed (per-document forward, batch=1) ===")
    print(f"  docs timed:        {len(per_doc_ms)}")
    print(
        f"  tokens timed:      {tot_tokens}  "
        f"(median={statistics.median(n_tokens)} max={max(n_tokens)})"
    )
    print(f"  units / doc:       median={statistics.median(n_units)} max={max(n_units)}")
    print(
        f"  latency ms/doc:    "
        f"mean={statistics.mean(per_doc_ms):.1f}  "
        f"median={statistics.median(per_doc_ms):.1f}  "
        f"p90={_p(per_doc_ms, 0.90):.1f}  "
        f"p99={_p(per_doc_ms, 0.99):.1f}  "
        f"min={min(per_doc_ms):.1f}  max={max(per_doc_ms):.1f}"
    )
    print(
        f"  throughput:        "
        f"{len(per_doc_ms) / wall_total:.2f} docs/s   "
        f"{tot_tokens / wall_total:.0f} tokens/s"
    )
    mem_peak = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"  peak GPU mem:      {mem_peak:.2f} GB")

    # Length-bucketed latency: see how cost scales with input length.
    print("\n=== Latency by token-length bucket ===")
    buckets = [(0, 1024), (1024, 2048), (2048, 4096), (4096, 6144), (6144, 8192)]
    for lo, hi in buckets:
        ms = [m for m, n in zip(per_doc_ms, n_tokens) if lo <= n < hi]
        if not ms:
            continue
        print(
            f"  [{lo:>4}, {hi:>4}):  n={len(ms):>3}  "
            f"mean={statistics.mean(ms):>6.1f} ms  "
            f"median={statistics.median(ms):>6.1f} ms"
        )


def _p(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round(q * (len(xs) - 1)))))
    return xs[k]


if __name__ == "__main__":
    main()
