"""Pre-process raw LLM-judged jsonl into train/val/test split files.

For each kept row we tokenize once (with the same tokenizer training will use)
and stamp `n_tokens` onto the row. Rows whose full tokenization exceeds
`--max-length` are split into consecutive windows at sentence boundaries with
the same `pack_windows` logic inference (`compress_long`) uses, and each
window becomes its own output row with `relevant_unit_ids` re-indexed to the
window — training never silently truncates past a relevant unit, and the model
sees the same mid-document / all-negative windows it will face at inference.
Splits are assigned by hashing (seed, query, full body) so membership is
stable across data churn and all windows of one document land in the same
split.

Output layout::

    <out-dir>/
        train.jsonl       # rows with n_tokens, source fields added
        val.jsonl
        test.jsonl
        manifest.json     # tokenizer hash, max_length, seed, per-source counts

Run::

    uv run python -m snippets_training.prepro \\
        --in data/all_labels.jsonl \\
        --out-dir data/modernbert_8k_v1 \\
        --base answerdotai/ModernBERT-base \\
        --max-length 8192
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from .dataset import _build_body_and_spans, _tokenizer_tag
from .windowing import pack_windows

SPLITS = ("train", "val", "test")


@dataclass
class SourceStats:
    path: str
    n_raw: int = 0
    n_status_skipped: int = 0
    n_empty_units: int = 0
    n_windowed_docs: int = 0  # docs > max_length, split into windows
    n_window_rows: int = 0  # rows emitted from those windows
    n_oversize_unit_dropped: int = 0  # windows whose single unit still exceeds max_length
    n_kept: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "n_raw": self.n_raw,
            "n_status_skipped": self.n_status_skipped,
            "n_empty_units": self.n_empty_units,
            "n_windowed_docs": self.n_windowed_docs,
            "n_window_rows": self.n_window_rows,
            "n_oversize_unit_dropped": self.n_oversize_unit_dropped,
            "n_kept": self.n_kept,
        }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in", dest="inputs", nargs="+", required=True, help="One or more raw jsonl files."
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory. Will be created. Will contain "
        "{train,val,test}.jsonl and manifest.json.",
    )
    p.add_argument(
        "--base",
        default="answerdotai/ModernBERT-base",
        help="HF tokenizer / base model id used to compute n_tokens.",
    )
    p.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Rows whose full tokenization exceeds this are split "
        "into windows at sentence boundaries (one output row "
        "per window).",
    )
    p.add_argument(
        "--window-margin",
        type=int,
        default=8,
        help="Token slack per window absorbing joint-tokenization "
        "drift vs. per-unit counts. Must match the margin "
        "used by compress_long at inference.",
    )
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--require-status-ok",
        action="store_true",
        default=True,
        help="Drop rows whose `status` field is not 'ok'. "
        "Matches the historical filter in load_jsonl.",
    )
    p.add_argument("--no-require-status-ok", dest="require_status_ok", action="store_false")
    return p.parse_args()


def iter_raw_rows(
    paths: list[Path], require_status_ok: bool, stats: dict[str, SourceStats]
) -> Iterator[tuple[dict, str]]:
    """Yield (row, source_path) for raw rows that pass basic filters."""
    for p in paths:
        s = SourceStats(path=str(p))
        stats[str(p)] = s
        with open(p) as f:
            for line in f:
                s.n_raw += 1
                d = json.loads(line)
                if require_status_ok and d.get("status") != "ok":
                    s.n_status_skipped += 1
                    continue
                if not d.get("units"):
                    s.n_empty_units += 1
                    continue
                yield d, str(p)


def tokenize_length(tok: PreTrainedTokenizerFast, query: str, body: str) -> int:
    """Untruncated token count of (query, body) including special tokens."""
    enc = tok(query, body, add_special_tokens=True, truncation=False)
    return len(enc["input_ids"])


def assign_split(query: str, body: str, seed: int, val_frac: float, test_frac: float) -> str:
    """Deterministic per-row split. Hashing (seed, query, body) makes
    membership stable: adding or removing rows can't move other rows between
    splits."""
    key = f"{seed}\0{query}\0{body}".encode()
    h = hashlib.sha1(key).digest()
    u = int.from_bytes(h[:8], "big") / 2**64
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    assert tok.is_fast, "Need a fast tokenizer."
    tok_hash = _tokenizer_tag(tok)

    print(f"Tokenizer: {args.base}  (hash={tok_hash})")
    print(
        f"max_length={args.max_length}  seed={args.seed}  "
        f"val_frac={args.val_frac}  test_frac={args.test_frac}"
    )
    print(f"Inputs: {args.inputs}")
    print(f"Out:    {out_dir}")

    stats: dict[str, SourceStats] = {}
    writers = {s: open(out_dir / f"{s}.jsonl", "w") for s in SPLITS}
    split_counts = {s: 0 for s in SPLITS}

    specials = tok.num_special_tokens_to_add(pair=True)

    def emit(row: dict, split: str, src: str, n_tokens: int) -> None:
        out_row = dict(row)
        out_row["n_tokens"] = int(n_tokens)
        out_row["source"] = src
        writers[split].write(json.dumps(out_row) + "\n")
        split_counts[split] += 1
        stats[src].n_kept += 1

    try:
        for i, (row, src) in enumerate(
            iter_raw_rows([Path(p) for p in args.inputs], args.require_status_ok, stats)
        ):
            units = row["units"]
            query = row["query"]
            body, _spans = _build_body_and_spans(units)
            n_tokens = tokenize_length(tok, query, body)
            # Split is decided by the full document, so all windows of one
            # doc share a split and short-doc membership matches older runs.
            split = assign_split(query, body, args.seed, args.val_frac, args.test_frac)
            if n_tokens <= args.max_length:
                emit(row, split, src, n_tokens)
            else:
                # Same windowing as compress_long at inference: pack whole
                # sentences into windows that fit alongside the query.
                stats[src].n_windowed_docs += 1
                q_len = len(tok(query, add_special_tokens=False)["input_ids"])
                capacity = max(64, args.max_length - q_len - specials - args.window_margin)
                unit_lens = [len(ids) for ids in tok(units, add_special_tokens=False)["input_ids"]]
                rel = set(row.get("relevant_unit_ids") or [])
                for ws, we in pack_windows(unit_lens, capacity):
                    w_units = units[ws:we]
                    w_body, _ = _build_body_and_spans(w_units)
                    w_tokens = tokenize_length(tok, query, w_body)
                    if w_tokens > args.max_length:
                        # Single unit bigger than the window; truncating it
                        # could cut a relevant span, so drop it instead.
                        stats[src].n_oversize_unit_dropped += 1
                        continue
                    w_row = dict(row)
                    w_row["units"] = w_units
                    w_row["relevant_unit_ids"] = sorted(r - ws for r in rel if ws <= r < we)
                    w_row["window"] = [ws, we]
                    emit(w_row, split, src, w_tokens)
                    stats[src].n_window_rows += 1
            if (i + 1) % 1000 == 0:
                print(f"  processed {i + 1} rows ...")
    finally:
        for w in writers.values():
            w.close()

    sha1s = {s: file_sha1(out_dir / f"{s}.jsonl") for s in SPLITS}

    manifest = {
        "tokenizer_name_or_path": args.base,
        "tokenizer_hash": tok_hash,
        "max_length": int(args.max_length),
        "window_margin": int(args.window_margin),
        "seed": int(args.seed),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "require_status_ok": bool(args.require_status_ok),
        "sources": [stats[k].to_dict() for k in sorted(stats)],
        "splits": {s: {"n": split_counts[s], "sha1": sha1s[s]} for s in SPLITS},
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print("\n=== Done ===")
    for s in SPLITS:
        print(f"  {s}: {split_counts[s]:>6d}  ({out_dir / f'{s}.jsonl'})")
    print("\nPer-source:")
    for k in sorted(stats):
        st = stats[k]
        print(
            f"  {k}: raw={st.n_raw} kept={st.n_kept} "
            f"windowed_docs={st.n_windowed_docs} "
            f"window_rows={st.n_window_rows} "
            f"oversize_unit_dropped={st.n_oversize_unit_dropped} "
            f"status_skipped={st.n_status_skipped} "
            f"empty_units={st.n_empty_units}"
        )
    print(f"\nManifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
