"""Minimal training loop for the sentence compressor.

Run (single GPU):
    uv run python -m snippets_training.train \
        --train data/modernbert_8k_v1/train.jsonl \
        --val   data/modernbert_8k_v1/val.jsonl

Run (multi-GPU DDP, one process per GPU):
    uv run torchrun --standalone --nproc_per_node=2 -m snippets_training.train \
        --train data/modernbert_8k_v1/train.jsonl \
        --val   data/modernbert_8k_v1/val.jsonl

Split files are produced by `snippets_training.prepro`.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from contextlib import nullcontext as _nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

try:
    import wandb
except ImportError:
    wandb = None

from .dataset import (
    IGNORE_INDEX,
    Collator,
    CompressionDataset,
    LengthBucketSampler,
    load_jsonl,
)
from .model import SentenceCompressor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train", required=True, help="Path to train.jsonl produced by snippets_training.prepro."
    )
    p.add_argument(
        "--val", required=True, help="Path to val.jsonl produced by snippets_training.prepro."
    )
    p.add_argument("--base", default="answerdotai/ModernBERT-base")
    p.add_argument("--max-length", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch = batch_size * world_size * grad_accum.",
    )
    p.add_argument("--attn-impl", default=None, help="e.g. flash_attention_2, sdpa, eager")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument(
        "--pos-weight",
        type=float,
        default=0.0,
        help="BCE positive-class weight. 0 disables. Try ~ (1-p)/p of label distribution.",
    )
    p.add_argument(
        "--head-bias-init",
        type=float,
        default=None,
        help="Initial bias for the classification head (logit of expected positive rate).",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--out", default="checkpoints/run1")
    p.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    p.add_argument(
        "--monitor",
        default="sent_t0.5_s0.3/f1",
        help="Val metric to select the best checkpoint. Use a '/loss' metric for lower-is-better.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early-stop after this many epochs without monitor improvement. 0 disables.",
    )
    p.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum monitor change to count as an improvement.",
    )
    p.add_argument(
        "--length-bucket",
        action="store_true",
        default=True,
        help="Group same-length examples per batch (less padding).",
    )
    p.add_argument("--no-length-bucket", dest="length_bucket", action="store_false")
    p.add_argument(
        "--bucket-size",
        type=int,
        default=50,
        help="Mega-batch size (in batches) for length bucketing.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=0,
        help="If >0, cap tokens-per-batch (bs * max_len_in_batch). Requires --length-bucket.",
    )
    p.add_argument("--wandb", action="store_true", default=True, help="Log to Weights & Biases.")
    p.add_argument("--no-wandb", dest="wandb", action="store_false")
    p.add_argument(
        "--wandb-project", default=os.environ.get("WANDB_PROJECT", "context-compression")
    )
    p.add_argument("--wandb-run-name", default=None)
    return p.parse_args()


MODEL_KEYS = ("input_ids", "attention_mask", "labels", "sentence_ids", "sent_labels")
SENT_THRESHOLDS = ((0.5, 0.5), (0.3, 0.5), (0.5, 0.3), (0.3, 0.3))


def setup_ddp() -> tuple[int, int, int, bool]:
    """Initialize torch.distributed if launched via torchrun.

    Returns (rank, world_size, local_rank, is_distributed).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _all_reduce_sums(values: dict[str, float], device: torch.device) -> dict[str, float]:
    """Sum-reduce a dict of scalars across ranks. No-op when single process."""
    if not (dist.is_available() and dist.is_initialized()):
        return values
    keys = list(values.keys())
    t = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return {k: t[i].item() for i, k in enumerate(keys)}


def _sentence_metrics(
    probs: torch.Tensor,  # (T,) float
    sentence_ids: torch.Tensor,  # (T,) long, -1 outside doc
    relevant: list[int],
    n_units: int,
    token_threshold: float,
    sentence_threshold: float,
) -> tuple[int, int, int, float, float]:
    """Return (tp, fp, fn, kept_count, total_count) for one example."""
    if n_units == 0:
        return 0, 0, 0, 0.0, 0.0
    rel_set = set(relevant)
    kept: set[int] = set()
    # Per-sentence: fraction of tokens with prob >= token_threshold.
    for j in range(n_units):
        mask = sentence_ids == j
        if not mask.any():
            continue
        p = probs[mask]
        frac_above = (p >= token_threshold).float().mean().item()
        if frac_above >= sentence_threshold:
            kept.add(j)
    tp = len(kept & rel_set)
    fp = len(kept - rel_set)
    fn = len(rel_set - kept)
    return tp, fp, fn, float(len(kept)), float(n_units)


def _unit_mean_probs(
    probs: torch.Tensor,  # (T,) float
    sentence_ids: torch.Tensor,  # (T,) long, -1 outside doc
    n_units: int,
) -> torch.Tensor:
    """Mean token-prob per unit -> (n_units,)."""
    out = torch.zeros(n_units)
    for j in range(n_units):
        mask = sentence_ids == j
        if mask.any():
            out[j] = probs[mask].mean()
    return out


def _ranking_metrics(
    unit_probs: torch.Tensor,
    relevant: list[int],
    n_units: int,
) -> tuple[float | None, float | None]:
    """Per-doc ranking quality, independent of any threshold.

    Returns (auc, recall_at_k) where auc is the Mann-Whitney probability that a
    relevant unit outscores a non-relevant one, and recall_at_k uses k=|relevant|
    (the oracle keep-budget). Returns (None, None) when AUC is undefined
    (no positives or no negatives).
    """
    rel_set = {int(r) for r in relevant if 0 <= int(r) < n_units}
    n_pos = len(rel_set)
    if n_units == 0 or n_pos == 0 or n_pos == n_units:
        return None, None
    n_neg = n_units - n_pos
    scores = unit_probs[:n_units]
    order = torch.argsort(scores, descending=True)
    topk = set(order[:n_pos].tolist())
    recall = len(topk & rel_set) / n_pos
    # AUC via rank-sum (handles ties at 0.5 through average ranks).
    ranks = torch.argsort(torch.argsort(scores)).float() + 1.0
    pos_idx = torch.tensor(sorted(rel_set), dtype=torch.long)
    sum_ranks_pos = ranks[pos_idx].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc, recall


def make_splits(full, val_frac: float, test_frac: float, seed: int):
    """DEPRECATED. Kept temporarily for any external callers; raises so misuse
    is loud. Splits now live on disk (produced by snippets_training.prepro)."""
    raise NotImplementedError(
        "make_splits has been removed. Run snippets_training.prepro to produce "
        "train/val/test jsonl files and pass them via --train/--val."
    )


def save_checkpoint(out_dir, model, tok, args, metrics, epoch):
    """Persist model state + tokenizer + the val metrics it was selected on."""
    model_to_save = model.module if isinstance(model, DDP) else model
    ckpt = out_dir / "model.pt"
    torch.save(
        {
            "model": model_to_save.state_dict(),
            "args": vars(args),
            "metrics": metrics,
            "epoch": epoch,
        },
        ckpt,
    )
    tok.save_pretrained(out_dir)
    return ckpt


@torch.no_grad()
def evaluate(model, loader, device, autocast_dtype):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    total_tok = 0
    tp = fp = fn = tn = 0
    # Sentence-level accumulators, keyed by (tok_thr, sent_thr).
    sent_tp = {k: 0 for k in SENT_THRESHOLDS}
    sent_fp = {k: 0 for k in SENT_THRESHOLDS}
    sent_fn = {k: 0 for k in SENT_THRESHOLDS}
    sent_kept = {k: 0.0 for k in SENT_THRESHOLDS}
    sent_total = 0.0
    # Threshold-free ranking accumulators (macro over docs).
    auc_sum = 0.0
    auc_n = 0
    rec_sum = 0.0
    rec_n = 0

    for batch in loader:
        model_in = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k in MODEL_KEYS}
        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
            logits, loss = model(**model_in)

        labels = model_in["labels"]
        mask = labels != IGNORE_INDEX
        n = mask.sum().item()
        total_loss += loss.item()
        n_batches += 1
        total_tok += n
        preds = (logits[mask] > 0).long()
        lab = labels[mask].long()
        tp += ((preds == 1) & (lab == 1)).sum().item()
        fp += ((preds == 1) & (lab == 0)).sum().item()
        fn += ((preds == 0) & (lab == 1)).sum().item()
        tn += ((preds == 0) & (lab == 0)).sum().item()

        # Sentence-level: pool token probs by sentence_ids.
        probs_all = torch.sigmoid(logits).float().cpu()
        sids_all = batch["sentence_ids"]  # CPU long tensor
        for i in range(probs_all.shape[0]):
            relevant = batch["relevant"][i]
            n_units = batch["n_units"][i]
            sent_total += n_units
            for thr in SENT_THRESHOLDS:
                t_tok, t_sent = thr
                etp, efp, efn, ekept, _ = _sentence_metrics(
                    probs_all[i], sids_all[i], relevant, n_units, t_tok, t_sent
                )
                sent_tp[thr] += etp
                sent_fp[thr] += efp
                sent_fn[thr] += efn
                sent_kept[thr] += ekept

            unit_probs = _unit_mean_probs(probs_all[i], sids_all[i], n_units)
            d_auc, d_rec = _ranking_metrics(unit_probs, relevant, n_units)
            if d_auc is not None:
                auc_sum += d_auc
                auc_n += 1
                rec_sum += d_rec
                rec_n += 1

    model.train()

    # Aggregate across ranks before computing rates.
    sums = {
        "total_loss": total_loss,
        "n_batches": float(n_batches),
        "total_tok": float(total_tok),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "sent_total": sent_total,
        "auc_sum": auc_sum,
        "auc_n": float(auc_n),
        "rec_sum": rec_sum,
        "rec_n": float(rec_n),
    }
    for thr in SENT_THRESHOLDS:
        tag = f"{thr[0]}_{thr[1]}"
        sums[f"stp_{tag}"] = float(sent_tp[thr])
        sums[f"sfp_{tag}"] = float(sent_fp[thr])
        sums[f"sfn_{tag}"] = float(sent_fn[thr])
        sums[f"skept_{tag}"] = float(sent_kept[thr])
    sums = _all_reduce_sums(sums, device)

    tp, fp, fn, tn = sums["tp"], sums["fp"], sums["fn"], sums["tn"]
    total_loss = sums["total_loss"]
    n_batches = max(sums["n_batches"], 1.0)
    total_tok = sums["total_tok"]
    sent_total = sums["sent_total"]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    out = {
        "loss": total_loss / n_batches,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "pos_frac": (tp + fn) / max(total_tok, 1),
        "ranking/auc": sums["auc_sum"] / max(sums["auc_n"], 1.0),
        "ranking/recall_at_k": sums["rec_sum"] / max(sums["rec_n"], 1.0),
    }
    for thr in SENT_THRESHOLDS:
        t_tok, t_sent = thr
        tag = f"{thr[0]}_{thr[1]}"
        s_tp, s_fp, s_fn = sums[f"stp_{tag}"], sums[f"sfp_{tag}"], sums[f"sfn_{tag}"]
        s_kept = sums[f"skept_{tag}"]
        s_prec = s_tp / max(s_tp + s_fp, 1)
        s_rec = s_tp / max(s_tp + s_fn, 1)
        s_f1 = 2 * s_prec * s_rec / max(s_prec + s_rec, 1e-9)
        keep_rate = s_kept / max(sent_total, 1)
        name = f"sent_t{t_tok}_s{t_sent}"
        out[f"{name}/precision"] = s_prec
        out[f"{name}/recall"] = s_rec
        out[f"{name}/f1"] = s_f1
        out[f"{name}/keep_rate"] = keep_rate  # fraction of sentences kept (1 - compression)
    return out


def main():
    args = parse_args()
    rank, world_size, local_rank, is_dist = setup_ddp()
    is_main = rank == 0

    def log(msg: str) -> None:
        if is_main:
            print(msg)

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank if is_dist else 0)
    else:
        device = torch.device("cpu")
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16

    log(f"DDP: world_size={world_size} rank={rank} local_rank={local_rank} device={device}")
    log(f"Loading train from {args.train} ...")
    train_examples = load_jsonl(args.train)
    log(f"  -> {len(train_examples)} train examples")
    log(f"Loading val from {args.val} ...")
    val_examples = load_jsonl(args.val)
    log(f"  -> {len(val_examples)} val examples")

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    train_ds = CompressionDataset(train_examples, tok, max_length=args.max_length)
    val_ds = CompressionDataset(val_examples, tok, max_length=args.max_length)

    collator = Collator(tok)

    if args.length_bucket:
        train_lengths = train_ds.lengths
        val_lengths = val_ds.lengths
        if is_main:
            import statistics as _s

            print(
                f"  train length: median={_s.median(train_lengths)} "
                f"max={max(train_lengths)} mean={int(_s.mean(train_lengths))}"
            )
        train_sampler = LengthBucketSampler(
            train_lengths,
            batch_size=args.batch_size,
            shuffle=True,
            bucket_size=args.bucket_size,
            seed=args.seed,
            max_tokens=args.max_tokens or None,
            num_replicas=world_size,
            rank=rank,
        )
        val_sampler = LengthBucketSampler(
            val_lengths,
            batch_size=args.batch_size,
            shuffle=False,
            bucket_size=args.bucket_size,
            seed=args.seed,
            max_tokens=args.max_tokens or None,
            num_replicas=world_size,
            rank=rank,
        )
        train_loader = DataLoader(
            train_ds,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        train_sampler = None
        val_sampler_inst = None
        if is_dist:
            train_sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=args.seed,
                drop_last=True,
            )
            val_sampler_inst = DistributedSampler(
                val_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                drop_last=True,
            )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler_inst,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    model = SentenceCompressor(
        base=args.base,
        attn_implementation=args.attn_impl,
        pos_weight=args.pos_weight or None,
        head_bias_init=args.head_bias_init,
    ).to(device)
    if is_dist:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=False,
        )

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
    )

    micro_per_epoch = len(train_loader)
    optim_per_epoch = math.ceil(micro_per_epoch / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else optim_per_epoch * args.epochs
    warmup = int(total_steps * args.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(optim, warmup, total_steps)

    out_dir = Path(args.out)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    if is_dist:
        dist.barrier()

    use_wandb = is_main and args.wandb and wandb is not None and os.environ.get("WANDB_API_KEY")
    if is_main and args.wandb and not use_wandb:
        log("  wandb disabled (missing wandb package or WANDB_API_KEY)")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={**vars(args), "world_size": world_size},
            dir=str(out_dir),
        )

    eff_batch = args.batch_size * world_size * max(1, args.grad_accum)
    log(
        f"Training: {total_steps} optim steps, warmup={warmup}, "
        f"per-rank-batch={args.batch_size}, grad_accum={args.grad_accum}, "
        f"world_size={world_size}, eff_batch={eff_batch}, lr={args.lr}, bf16={args.bf16}, "
        f"pos_weight={args.pos_weight}, head_bias_init={args.head_bias_init}"
    )

    model.train()
    step = 0  # optimizer steps
    micro_step = 0  # forward/backward count
    running = 0.0
    running_n = 0
    t0 = time.time()
    done = False
    accum = max(1, args.grad_accum)
    # Best-checkpoint / early-stopping state.
    monitor = args.monitor
    higher_better = "loss" not in monitor
    best_metric = None
    best_epoch = -1
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        for batch in train_loader:
            model_in = {
                k: v.to(device, non_blocking=True) for k, v in batch.items() if k in MODEL_KEYS
            }
            is_accum_boundary = (micro_step + 1) % accum == 0
            # Skip DDP gradient sync on intermediate accumulation steps for speed.
            sync_ctx = model.no_sync() if (is_dist and not is_accum_boundary) else _nullcontext()
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _, loss = model(**model_in)
                (loss / accum).backward()
            micro_step += 1
            running += loss.item()
            running_n += 1
            if not is_accum_boundary:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 and is_main:
                dt = time.time() - t0
                ex_per_s = micro_step * args.batch_size * world_size / dt
                mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0
                avg_loss = running / running_n
                lr_now = sched.get_last_lr()[0]
                print(
                    f"step {step}/{total_steps} epoch {epoch} "
                    f"loss {avg_loss:.4f} "
                    f"lr {lr_now:.2e} "
                    f"ex/s {ex_per_s:.1f} peak_mem {mem:.2f}GB"
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/lr": lr_now,
                            "train/ex_per_s": ex_per_s,
                            "train/peak_mem_gb": mem,
                            "train/epoch": epoch,
                        },
                        step=step,
                    )
            if step % args.log_every == 0:
                running = 0.0
                running_n = 0

            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break
        if done:
            break

        metrics = evaluate(model, val_loader, device, autocast_dtype)
        log(f"[epoch {epoch}] val {metrics}")
        if use_wandb:
            wandb.log(
                {f"val/{k}": v for k, v in metrics.items()} | {"train/epoch": epoch}, step=step
            )

        # Best-checkpoint selection + early stopping. All ranks see identical
        # (all-reduced) metrics, so this decision is consistent across DDP.
        cur = metrics.get(monitor)
        if cur is None:
            raise KeyError(
                f"--monitor='{monitor}' not in val metrics. Available: {sorted(metrics)}"
            )
        improved = best_metric is None or (
            cur > best_metric + args.min_delta
            if higher_better
            else cur < best_metric - args.min_delta
        )
        if improved:
            best_metric = cur
            best_epoch = epoch
            epochs_no_improve = 0
            if is_main:
                ckpt = save_checkpoint(out_dir, model, tok, args, metrics, epoch)
                log(f"  \u2713 new best {monitor}={cur:.4f} (epoch {epoch}) -> saved {ckpt}")
            if use_wandb:
                wandb.log({"val_best/metric": best_metric, "val_best/epoch": best_epoch}, step=step)
        else:
            epochs_no_improve += 1
            log(
                f"  no improvement ({epochs_no_improve}/{args.patience}) "
                f"best {monitor}={best_metric:.4f} @ epoch {best_epoch}"
            )
            if args.patience > 0 and epochs_no_improve >= args.patience:
                log(f"  early stopping at epoch {epoch}.")
                break

    if best_metric is None:
        # No epoch-level checkpoint was saved (e.g. --max-steps probe run).
        metrics = evaluate(model, val_loader, device, autocast_dtype)
        log(f"[final] val {metrics}")
        if use_wandb:
            wandb.log({f"val_final/{k}": v for k, v in metrics.items()}, step=step)
        if is_main:
            ckpt = save_checkpoint(out_dir, model, tok, args, metrics, epoch)
            print(f"Saved checkpoint to {ckpt}")
    else:
        log(f"[done] best {monitor}={best_metric:.4f} @ epoch {best_epoch} (checkpoint kept)")
    if use_wandb:
        wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
