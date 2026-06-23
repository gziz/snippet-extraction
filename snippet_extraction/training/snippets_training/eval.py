"""Evaluate a saved checkpoint on a pre-built split file.

Run:
    .venv/bin/python -m snippets_training.eval \\
        --ckpt checkpoints/run_full \\
        --data data/modernbert_8k_v1/test.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from .dataset import Collator, CompressionDataset, LengthBucketSampler, _tokenizer_tag, load_jsonl
from .model import SentenceCompressor
from .train import (  # noqa: F401  (MODEL_KEYS re-exported for back-compat)
    MODEL_KEYS,
    SENT_THRESHOLDS,
    evaluate,
)


def _check_manifest(data_path: Path, tok, max_length: int) -> None:
    """If a `manifest.json` sits next to the split file, warn loudly when its
    tokenizer hash or max_length disagree with what eval is using."""
    manifest_path = data_path.parent / "manifest.json"
    if not manifest_path.exists():
        return
    with open(manifest_path) as f:
        m = json.load(f)
    cur_hash = _tokenizer_tag(tok)
    if m.get("tokenizer_hash") != cur_hash:
        print(
            f"WARNING: tokenizer hash mismatch — manifest={m.get('tokenizer_hash')} "
            f"current={cur_hash}. n_tokens / drop decisions may be stale."
        )
    if int(m.get("max_length", -1)) != int(max_length):
        print(
            f"WARNING: max_length mismatch — manifest={m.get('max_length')} "
            f"current={max_length}. Rows that fit in the manifest may not fit now."
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", required=True, help="Checkpoint dir (contains model.pt + tokenizer files)."
    )
    p.add_argument(
        "--data",
        required=True,
        help="Path to a split jsonl produced by snippets_training.prepro "
        "(e.g. data/<run>/test.jsonl).",
    )
    p.add_argument("--base", default="answerdotai/ModernBERT-base")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=16384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument(
        "--per-source",
        action="store_true",
        default=True,
        help="Also report metrics for the subset of the split that came from each source jsonl.",
    )
    p.add_argument("--no-per-source", dest="per_source", action="store_false")
    p.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated list of token,sent threshold pairs, e.g. "
        "'0.5,0.5;0.4,0.5;0.3,0.4'. Default uses SENT_THRESHOLDS.",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    if args.thresholds:
        thrs = []
        for pair in args.thresholds.split(";"):
            t, s = pair.split(",")
            thrs.append((float(t), float(s)))
        # Monkey-patch the module-level constant.
        import snippets_training.train as _t

        _t.SENT_THRESHOLDS = tuple(thrs)

    data_path = Path(args.data)
    print(f"Loading {data_path} ...")
    examples = load_jsonl(data_path)
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    _check_manifest(data_path, tok, args.max_length)
    full = CompressionDataset(examples, tok, max_length=args.max_length)
    print(f"  -> {len(full)} examples")

    collator = Collator(tok)

    def build_loader(indices: list[int]) -> DataLoader:
        sub = Subset(full, indices)
        lens = [full.lengths[i] for i in indices]
        sampler = LengthBucketSampler(
            lens,
            batch_size=args.batch_size,
            shuffle=False,
            bucket_size=50,
            seed=args.seed,
            max_tokens=args.max_tokens or None,
        )
        return DataLoader(
            sub,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    loader = build_loader(list(range(len(full))))

    model = SentenceCompressor(base=args.base).to(device)
    state = torch.load(f"{args.ckpt}/model.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    print(
        f"Loaded checkpoint: train args = {state.get('args', {}).get('wandb_run_name', '?')}"
        f"  (epoch={state.get('epoch', '?')})"
    )

    split_name = data_path.stem.upper()
    print(f"\n=== {split_name} (overall, n={len(full)}) ===")
    overall = evaluate(model, loader, device, autocast_dtype)
    print(json.dumps(overall, indent=2, sort_keys=True))

    if args.per_source:
        by_src: dict[str, list[int]] = defaultdict(list)
        for i, ex in enumerate(examples):
            by_src[ex.source].append(i)
        if len(by_src) > 1:
            print(f"\n--- per-source breakdown of {split_name} ---")
            for src, idxs in sorted(by_src.items()):
                print(f"\n=== {split_name} subset from {Path(src).name} (n={len(idxs)}) ===")
                if not idxs:
                    print("  (empty)")
                    continue
                sub_loader = build_loader(idxs)
                sub_metrics = evaluate(model, sub_loader, device, autocast_dtype)
                print(json.dumps(sub_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
