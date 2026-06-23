"""Run a trained SentenceCompressor on a jsonl file and emit per-unit predictions.

Output rows mirror the input rows with two extra fields:
    predicted_unit_ids: list[int]   # units predicted relevant at given thresholds
    unit_scores: list[float]        # mean sigmoid(token logit) per unit (rounded)

Usage:
    uv run python -m snippets_training.predict \
        --ckpt checkpoints/run2 \
        --data evals/exa_v1.jsonl \
        --out evals/exa_v1.pred_run2.jsonl \
        --limit 50 \
        --token-threshold 0.5 --sent-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .dataset import _build_body_and_spans
from .model import SentenceCompressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="answerdotai/ModernBERT-base")
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--token-threshold", type=float, default=0.5)
    ap.add_argument("--sent-threshold", type=float, default=0.5)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument(
        "--load-in-8bit", action="store_true", help="Load encoder in 8-bit via bitsandbytes."
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)

    if args.load_in_8bit:
        import tempfile

        import torch.nn as nn
        from transformers import AutoModel, BitsAndBytesConfig

        # 1) Materialize the fine-tuned encoder to a temp HF dir.
        state = torch.load(f"{args.ckpt}/model.pt", map_location="cpu", weights_only=False)
        fp_encoder = AutoModel.from_pretrained(args.base)
        enc_state = {
            k[len("encoder.") :]: v for k, v in state["model"].items() if k.startswith("encoder.")
        }
        missing, unexpected = fp_encoder.load_state_dict(enc_state, strict=False)
        if missing:
            print(f"  encoder missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"  encoder unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
        with tempfile.TemporaryDirectory() as td:
            fp_encoder.save_pretrained(td)
            del fp_encoder
            # 2) Reload quantized.
            bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
            encoder = AutoModel.from_pretrained(
                td,
                quantization_config=bnb_cfg,
                attn_implementation=args.attn_impl,
            )
        # 3) Build SentenceCompressor shell around the quantized encoder.
        model = SentenceCompressor.__new__(SentenceCompressor)
        nn.Module.__init__(model)
        model.encoder = encoder
        h = encoder.config.hidden_size
        model.dropout = nn.Dropout(0.1)
        model.head = nn.Linear(h, 1).to(device)
        model.pos_weight = None
        head_state = {
            k.split(".", 1)[1]: v for k, v in state["model"].items() if k.startswith("head.")
        }
        model.head.load_state_dict(head_state)
        print(f"8-bit load: quantized encoder + FP head ({len(head_state)} head tensors)")
    else:
        model = SentenceCompressor(base=args.base, attn_implementation=args.attn_impl).to(device)
        state = torch.load(f"{args.ckpt}/model.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()

    # Stream + pre-filter: keep only docs whose (query, body) tokenizes to <= max_length.
    rows_in = []
    skipped_long = 0
    skipped_status = 0
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("status") != "ok" or not d.get("units"):
                skipped_status += 1
                continue
            body, _ = _build_body_and_spans(d["units"])
            n_tok = len(tok(d["query"], body, add_special_tokens=True)["input_ids"])
            if n_tok > args.max_length:
                skipped_long += 1
                continue
            d["_n_tokens"] = n_tok
            rows_in.append(d)
            if len(rows_in) >= args.limit:
                break
    print(
        f"Loaded {len(rows_in)} rows (<= {args.max_length} tokens); "
        f"skipped {skipped_long} too-long, {skipped_status} non-ok"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout, torch.no_grad():
        for k, d in enumerate(rows_in):
            units = d["units"]
            body, spans = _build_body_and_spans(units)
            enc = tok(
                d["query"],
                body,
                truncation="only_second",
                max_length=args.max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            seq_ids = enc.sequence_ids(0)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits, _ = model(input_ids=input_ids, attention_mask=attn)
            probs = torch.sigmoid(logits[0]).float().cpu().tolist()

            # Aggregate per unit.
            n_units = len(units)
            sums = [0.0] * n_units
            counts = [0] * n_units
            above = [0] * n_units
            for i, ((s, e), sid) in enumerate(zip(offsets, seq_ids)):
                if sid != 1 or e <= s:
                    continue
                mid = (s + e) // 2
                # Linear scan; n_units is small.
                u_idx = -1
                for j, (us, ue) in enumerate(spans):
                    if us <= mid < ue:
                        u_idx = j
                        break
                    if mid < us:
                        break
                if u_idx < 0:
                    continue
                p = probs[i]
                sums[u_idx] += p
                counts[u_idx] += 1
                if p >= args.token_threshold:
                    above[u_idx] += 1

            unit_scores = []
            predicted = []
            for j in range(n_units):
                mean_p = sums[j] / counts[j] if counts[j] else 0.0
                frac_above = above[j] / counts[j] if counts[j] else 0.0
                unit_scores.append(round(mean_p, 4))
                if counts[j] > 0 and frac_above >= args.sent_threshold:
                    predicted.append(j)

            d_out = dict(d)
            d_out["predicted_unit_ids"] = predicted
            d_out["unit_scores"] = unit_scores
            d_out["pred_meta"] = {
                "ckpt": str(args.ckpt),
                "token_threshold": args.token_threshold,
                "sent_threshold": args.sent_threshold,
            }
            fout.write(json.dumps(d_out) + "\n")
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{len(rows_in)}")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
