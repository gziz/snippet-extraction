"""Inference helper: given (query, document), return compressed snippet.

Token probs -> sentence pool (mean) -> dual threshold (token T_tok, sentence T_sent).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from snippets_common.spans import build_body_and_spans
from snippets_common.windowing import pack_windows
from transformers import AutoTokenizer

from .model import SentenceCompressor

# Half precision for the encoder forward pass. The model was trained AND
# validated under bf16 autocast (see snippets_training/train.py: --bf16 default
# True, and evaluate() runs the same autocast), so the 0.5/0.5 keep-drop
# thresholds and checkpoint selection were calibrated against bf16 logits.
# Matching that precision at inference avoids train/inference skew (fp32 is the
# real mismatch here, not bf16) and cuts the forward pass ~3.4x on an L4. Set to
# ``None`` to force fp32.
INFERENCE_DTYPE: torch.dtype | None = torch.bfloat16


@dataclass
class CompressResult:
    kept_indices: list[int]
    kept_sentences: list[str]
    snippet: str
    sentence_scores: list[float]  # frac of unit tokens above token_threshold
    compression_ratio: float  # kept_chars / total_chars
    # Mean token probability per unit: a finer-grained ranking key than
    # sentence_scores (which saturates at 1.0), used by budgeted selection.
    sentence_mean_probs: list[float] = field(default_factory=list)


def _split_sentences(text: str) -> list[str]:
    # Raw-string documents are segmented with the canonical snippets-common
    # segmenter (not a generic sentence splitter) so the runtime pools scores
    # onto the same unit layout training and serving use. Callers that already
    # have units should pass a list[str] and skip this path.
    from snippets_common.segment import segment

    return segment(text)


def _pool_window(
    offsets,
    seq_ids,
    probs,
    spans,
    n_units: int,
    token_threshold: float,
    sentence_threshold: float,
) -> tuple[list[float], list[float], list[int]]:
    """Pool per-token probs into per-unit (sent_scores, mean_probs, kept).

    Shared by the single-window (:func:`compress`) and batched
    (:func:`compress_long_batch`) paths so the keep/drop decision rule can never
    drift between them. ``offsets`` / ``seq_ids`` / ``probs`` are the per-token
    arrays for one sequence (body tokens carry ``seq_id == 1``); ``spans`` are
    the char spans of this window's units from ``build_body_and_spans``.

    Vectorized with numpy: a token is assigned to the unit whose char span
    contains its midpoint. ``spans`` are sorted and consecutive (from
    ``build_body_and_spans``), so ``searchsorted`` on the start offsets reproduces
    the original linear scan exactly (verified equal kept-set on the long-tail
    PDFs), at ~40x the speed of the per-token Python loop.
    """
    if n_units == 0:
        return [], [], []
    offs = np.asarray(offsets, dtype=np.int64)
    sids = np.asarray([s if s is not None else -1 for s in seq_ids], dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    s, e = offs[:, 0], offs[:, 1]
    body = (sids == 1) & (e > s)

    sent_scores = np.zeros(n_units, dtype=np.float64)
    mean_probs = np.zeros(n_units, dtype=np.float64)
    if not body.any():
        return sent_scores.tolist(), mean_probs.tolist(), []

    mids = (s[body] + e[body]) // 2
    pb = p[body]
    starts = np.fromiter((sp[0] for sp in spans), dtype=np.int64, count=n_units)
    ends = np.fromiter((sp[1] for sp in spans), dtype=np.int64, count=n_units)
    # Unit whose start <= mid, then confirm mid < that unit's end (matches the
    # original ``us <= mid < ue`` with an in-order break).
    cand = np.searchsorted(starts, mids, side="right") - 1
    valid = (cand >= 0) & (mids < ends[np.clip(cand, 0, n_units - 1)])
    u = cand[valid]
    pv = pb[valid]

    counts = np.bincount(u, minlength=n_units).astype(np.float64)
    sum_prob = np.bincount(u, weights=pv, minlength=n_units)
    sum_above = np.bincount(
        u, weights=(pv >= token_threshold).astype(np.float64), minlength=n_units
    )
    has = counts > 0
    sent_scores[has] = sum_above[has] / counts[has]
    mean_probs[has] = sum_prob[has] / counts[has]
    kept = [int(j) for j in np.nonzero(has & (sent_scores >= sentence_threshold))[0]]
    return sent_scores.tolist(), mean_probs.tolist(), kept


@torch.no_grad()
def compress(
    model: SentenceCompressor,
    tokenizer,
    query: str,
    document: str | list[str],
    *,
    max_length: int = 8192,
    token_threshold: float = 0.5,
    sentence_threshold: float = 0.5,
    device: torch.device | None = None,
) -> CompressResult:
    units = document if isinstance(document, list) else _split_sentences(document)
    body, spans = build_body_and_spans(units)

    device = device or next(model.parameters()).device
    enc = tokenizer(
        query,
        body,
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    seq_ids = enc.sequence_ids()
    enc = {k: v.to(device) for k, v in enc.items()}

    model.eval()
    if INFERENCE_DTYPE is not None and torch.device(device).type == "cuda":
        with torch.autocast("cuda", dtype=INFERENCE_DTYPE):
            logits, _ = model(**enc)
    else:
        logits, _ = model(**enc)
    probs = torch.sigmoid(logits)[0].float().cpu().tolist()

    sent_scores, mean_probs, kept = _pool_window(
        offsets,
        seq_ids,
        probs,
        spans,
        len(units),
        token_threshold,
        sentence_threshold,
    )

    kept_sents = [units[j] for j in kept]
    snippet = " ".join(kept_sents)
    total_chars = sum(len(u) for u in units) or 1
    ratio = sum(len(s) for s in kept_sents) / total_chars
    return CompressResult(
        kept_indices=kept,
        kept_sentences=kept_sents,
        snippet=snippet,
        sentence_scores=sent_scores,
        compression_ratio=ratio,
        sentence_mean_probs=mean_probs,
    )


@torch.no_grad()
def compress_long(
    model: SentenceCompressor,
    tokenizer,
    query: str,
    document: str | list[str],
    *,
    max_length: int = 8192,
    token_threshold: float = 0.5,
    sentence_threshold: float = 0.5,
    device: torch.device | None = None,
    window_margin: int = 8,
) -> CompressResult:
    """``compress`` for documents longer than the encoder window.

    Units are packed into consecutive windows that fit ``max_length`` alongside
    the query (windows break only at unit boundaries, so per-sentence pooling
    is unaffected), each window is compressed with the same query, and the
    per-window results are merged back into one document-level result. Each
    window looks to the model like a full document, which matches how it was
    trained (prepro windows long training docs the same way).
    """
    units = document if isinstance(document, list) else _split_sentences(document)
    if not units:
        return CompressResult(
            kept_indices=[],
            kept_sentences=[],
            snippet="",
            sentence_scores=[],
            compression_ratio=0.0,
            sentence_mean_probs=[],
        )

    q_tokens = len(tokenizer(query, add_special_tokens=False)["input_ids"])
    specials = tokenizer.num_special_tokens_to_add(pair=True)
    # window_margin absorbs joint-tokenization drift vs. per-unit counts.
    capacity = max(64, max_length - q_tokens - specials - window_margin)
    # One batched tokenizer call for all unit lengths, not one call per unit
    # (the per-unit loop was a large host-side cost on long docs).
    unit_lens = [len(ids) for ids in tokenizer(units, add_special_tokens=False)["input_ids"]]

    kept: list[int] = []
    sent_scores: list[float] = []
    mean_probs: list[float] = []
    for start, end in pack_windows(unit_lens, capacity):
        r = compress(
            model,
            tokenizer,
            query,
            units[start:end],
            max_length=max_length,
            token_threshold=token_threshold,
            sentence_threshold=sentence_threshold,
            device=device,
        )
        kept.extend(start + j for j in r.kept_indices)
        sent_scores.extend(r.sentence_scores)
        mean_probs.extend(r.sentence_mean_probs)

    kept_sents = [units[j] for j in kept]
    total_chars = sum(len(u) for u in units) or 1
    return CompressResult(
        kept_indices=kept,
        kept_sentences=kept_sents,
        snippet=" ".join(kept_sents),
        sentence_scores=sent_scores,
        compression_ratio=sum(len(s) for s in kept_sents) / total_chars,
        sentence_mean_probs=mean_probs,
    )


@dataclass
class DocRequest:
    """One document's scoring request for :func:`compress_long_batch`.

    ``document`` should already be a unit list (the caller runs the
    training-faithful segmenter); a raw string falls back to NLTK exactly like
    the single-doc path. Thresholds are per-document so a mixed batch is scored
    correctly even if callers configured different values.
    """

    query: str
    document: str | list[str]
    token_threshold: float = 0.5
    sentence_threshold: float = 0.5


@dataclass
class _WindowJob:
    """One encoder-sized window queued for batched scoring."""

    doc_id: int
    start: int  # global unit index (inclusive) into the doc's units
    end: int  # global unit index (exclusive)
    query: str
    units: list[str]  # this window's units
    body: str
    spans: list[tuple[int, int]]
    est_len: int  # ~tokenized length, for length-bucketing
    token_threshold: float
    sentence_threshold: float


def _tokenize_minibatch(tokenizer, jobs, max_length, timings=None):
    """Tokenize one padded minibatch on the CPU (no GPU touched).

    Returns ``(enc, offsets_b, seq_ids_b)`` with ``enc`` still on CPU — the H2D
    copy and forward happen in :func:`_forward_and_pool` on the main thread. Kept
    separate from the forward so the pipeline can prefetch this (GIL-releasing
    HF-Rust) work on a worker thread while the GPU runs the previous minibatch.
    """
    t = time.perf_counter() if timings is not None else 0.0
    enc = tokenizer(
        [j.query for j in jobs],
        [j.body for j in jobs],
        truncation="only_second",
        max_length=max_length,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets_b = enc.pop("offset_mapping").tolist()
    seq_ids_b = [enc.sequence_ids(i) for i in range(len(jobs))]
    if timings is not None:
        timings["tokenize"] += time.perf_counter() - t
    return enc, offsets_b, seq_ids_b


@torch.no_grad()
def _forward_and_pool(
    model: SentenceCompressor,
    jobs: list[_WindowJob],
    prepared,
    *,
    device,
    sent_acc: list[list[float]],
    mean_acc: list[list[float]],
    kept_acc: list[list[int]],
    timings: dict[str, float] | None = None,
) -> None:
    """Move a tokenized minibatch to the GPU, run the forward, pool + scatter.

    The GPU-touching half of a minibatch. Runs on the main thread so all CUDA
    ops (H2D, forward, autocast, D2H sync) stay on one thread while a worker
    thread prefetches the next minibatch's tokenization.
    """
    enc, offsets_b, seq_ids_b = prepared
    is_cuda = torch.device(device).type == "cuda"
    t = time.perf_counter() if timings is not None else 0.0

    enc = {k: v.to(device) for k, v in enc.items()}
    if INFERENCE_DTYPE is not None and is_cuda:
        with torch.autocast("cuda", dtype=INFERENCE_DTYPE):
            logits, _ = model(**enc)
    else:
        logits, _ = model(**enc)
    probs_b = torch.sigmoid(logits).float().cpu().tolist()  # .cpu() forces the D2H sync
    if timings is not None:
        timings["forward"] += time.perf_counter() - t
        t = time.perf_counter()

    for job, offsets, seq_ids, probs in zip(jobs, offsets_b, seq_ids_b, probs_b):
        sent, mean, kept_local = _pool_window(
            offsets,
            seq_ids,
            probs,
            job.spans,
            len(job.units),
            job.token_threshold,
            job.sentence_threshold,
        )
        for li, gi in enumerate(range(job.start, job.end)):
            sent_acc[job.doc_id][gi] = sent[li]
            mean_acc[job.doc_id][gi] = mean[li]
        kept_acc[job.doc_id].extend(job.start + j for j in kept_local)
    if timings is not None:
        timings["pool"] += time.perf_counter() - t


def _bucket_minibatches(jobs: list[_WindowJob], max_batch_tokens: int) -> list[list[_WindowJob]]:
    """Length-bucket jobs into padded minibatches under the rows*padded_len cap.

    Jobs are sorted by estimated length and greedily packed (bucketing keeps
    padding waste low); each returned bucket becomes one padded forward pass.
    """
    order = sorted(range(len(jobs)), key=lambda k: jobs[k].est_len)
    minibatches: list[list[_WindowJob]] = []
    i = 0
    while i < len(order):
        batch: list[_WindowJob] = []
        max_len = 0
        while i < len(order):
            cand = max(max_len, jobs[order[i]].est_len)
            if batch and cand * (len(batch) + 1) > max_batch_tokens:
                break
            batch.append(jobs[order[i]])
            max_len = cand
            i += 1
        minibatches.append(batch)
    return minibatches


def plan_jobs(
    requests: list[DocRequest],
    tokenizer,
    *,
    max_length: int = 8192,
    window_margin: int = 8,
) -> tuple[list[list[str]], list[_WindowJob]]:
    """CPU phase: segment + window every doc into a flat list of jobs.

    No model or GPU is touched here — only the tokenizer (host-side) is used to
    size windows. Returns ``(docs_units, jobs)``: the per-doc unit lists (kept
    for result assembly) and every encoder-sized window flattened into one job
    list ready for batched forward passes.
    """
    docs_units: list[list[str]] = []
    jobs: list[_WindowJob] = []
    for doc_id, req in enumerate(requests):
        units = req.document if isinstance(req.document, list) else _split_sentences(req.document)
        docs_units.append(units)
        if not units:
            continue
        q_tokens = len(tokenizer(req.query, add_special_tokens=False)["input_ids"])
        specials = tokenizer.num_special_tokens_to_add(pair=True)
        capacity = max(64, max_length - q_tokens - specials - window_margin)
        # One batched tokenizer call for all unit lengths (was one call per
        # unit — a large host-side cost on long docs).
        unit_lens = [len(ids) for ids in tokenizer(units, add_special_tokens=False)["input_ids"]]
        for start, end in pack_windows(unit_lens, capacity):
            w_units = units[start:end]
            body, spans = build_body_and_spans(w_units)
            est = min(max_length, q_tokens + specials + sum(unit_lens[start:end]))
            jobs.append(
                _WindowJob(
                    doc_id=doc_id,
                    start=start,
                    end=end,
                    query=req.query,
                    units=w_units,
                    body=body,
                    spans=spans,
                    est_len=est,
                    token_threshold=req.token_threshold,
                    sentence_threshold=req.sentence_threshold,
                )
            )
    return docs_units, jobs


@torch.no_grad()
def _run_jobs(
    model: SentenceCompressor,
    tokenizer,
    jobs: list[_WindowJob],
    *,
    max_length: int,
    device,
    max_batch_tokens: int,
    sent_acc: list[list[float]],
    mean_acc: list[list[float]],
    kept_acc: list[list[int]],
    timings: dict[str, float] | None = None,
    pipeline: bool = True,
) -> None:
    """GPU phase: bucket jobs into padded minibatches and run the forwards.

    This is the only GPU-bound work in the pipeline. With ``pipeline`` (default),
    the tokenization of minibatch N+1 is prefetched on a worker thread while the
    GPU runs the forward of minibatch N: both the HF fast tokenizer and torch's
    CUDA sync release the GIL, so the CPU-bound tokenize genuinely overlaps the
    forward and is hidden behind it (only the first minibatch's tokenize can't be
    overlapped — the pipeline-fill cost). ``pipeline=False`` runs the plain
    sequential path; the two are equivalent in output and exist to A/B the win.
    """
    minibatches = _bucket_minibatches(jobs, max_batch_tokens)
    if not minibatches:
        return

    def forward(batch, prepared):
        _forward_and_pool(
            model,
            batch,
            prepared,
            device=device,
            sent_acc=sent_acc,
            mean_acc=mean_acc,
            kept_acc=kept_acc,
            timings=timings,
        )

    # A single minibatch has nothing to overlap, so skip the thread entirely.
    if not pipeline or len(minibatches) == 1:
        for batch in minibatches:
            forward(batch, _tokenize_minibatch(tokenizer, batch, max_length, timings))
        return

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_tokenize_minibatch, tokenizer, minibatches[0], max_length, timings)
        for k, batch in enumerate(minibatches):
            prepared = future.result()  # first iter pays the pipeline-fill tokenize
            if k + 1 < len(minibatches):
                # Prefetch N+1's tokenize now so it runs during this forward.
                future = pool.submit(
                    _tokenize_minibatch, tokenizer, minibatches[k + 1], max_length, timings
                )
            forward(batch, prepared)


def assemble_results(
    docs_units: list[list[str]],
    sent_acc: list[list[float]],
    mean_acc: list[list[float]],
    kept_acc: list[list[int]],
) -> list[CompressResult]:
    """CPU phase: fold the per-doc accumulators into one result per document.

    No model or GPU is touched — pure bookkeeping over the pooled scores
    produced by the run phase. Results come back in input order.
    """
    results: list[CompressResult] = []
    for units, sent, mean, kept in zip(docs_units, sent_acc, mean_acc, kept_acc):
        if not units:
            results.append(
                CompressResult(
                    kept_indices=[],
                    kept_sentences=[],
                    snippet="",
                    sentence_scores=[],
                    compression_ratio=0.0,
                    sentence_mean_probs=[],
                )
            )
            continue
        kept_sorted = sorted(kept)
        kept_sents = [units[j] for j in kept_sorted]
        total_chars = sum(len(u) for u in units) or 1
        results.append(
            CompressResult(
                kept_indices=kept_sorted,
                kept_sentences=kept_sents,
                snippet=" ".join(kept_sents),
                sentence_scores=sent,
                compression_ratio=sum(len(s) for s in kept_sents) / total_chars,
                sentence_mean_probs=mean,
            )
        )
    return results


@torch.no_grad()
def compress_long_batch(
    model: SentenceCompressor,
    tokenizer,
    requests: list[DocRequest],
    *,
    max_length: int = 8192,
    device: torch.device | str | None = None,
    window_margin: int = 8,
    max_batch_tokens: int = 16384,
) -> list[CompressResult]:
    """Score many documents in shared GPU minibatches.

    Per document this is equivalent to :func:`compress_long`, but every window
    of every document is flattened into one list, length-bucketed, and run
    through padded minibatches — so a single ``compress_long_batch`` call replaces
    ``sum(windows)`` serial forward passes with a handful of batched ones.
    Per-document results are scattered back in input order. ``max_batch_tokens``
    caps ``rows * padded_len`` per forward pass to bound GPU memory. The cap is
    set to ~4 full 8192-token windows per pass: the batch-token sweep measured
    only 7.4 GiB peak at 16384 on a 24 GiB L4, so doubling it (~15 GiB) halves
    the number of serial forward passes on long multi-window docs with margin.

    The body is split into three explicit phases so the GPU-bound part is
    visibly tiny: :func:`plan_jobs` (CPU) → :func:`_run_jobs` (GPU) →
    :func:`assemble_results` (CPU).
    """
    # next(model.parameters()) yields the model's first weight tensor; .device
    # reads which device it lives on, so callers can omit ``device``.
    device = device or next(model.parameters()).device
    model.eval()

    # 1. Plan (CPU): segment + window every doc into a flat job list.
    docs_units, jobs = plan_jobs(
        requests, tokenizer, max_length=max_length, window_margin=window_margin
    )

    # Per-doc accumulators (unscored units default to 0.0 / not kept).
    sent_acc: list[list[float]] = [[0.0] * len(u) for u in docs_units]
    mean_acc: list[list[float]] = [[0.0] * len(u) for u in docs_units]
    kept_acc: list[list[int]] = [[] for _ in docs_units]

    # 2. Run (GPU): the only GPU-bound phase — batched forward passes.
    _run_jobs(
        model,
        tokenizer,
        jobs,
        max_length=max_length,
        device=device,
        max_batch_tokens=max_batch_tokens,
        sent_acc=sent_acc,
        mean_acc=mean_acc,
        kept_acc=kept_acc,
    )

    # 3. Assemble (CPU): fold accumulators into per-doc results, in input order.
    return assemble_results(docs_units, sent_acc, mean_acc, kept_acc)


def load_for_inference(
    ckpt_dir: str, base: str = "answerdotai/ModernBERT-base", device: str = "cuda"
):
    import torch as _t

    tok = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    # Pin SDPA attention explicitly. Under bf16 it dispatches to the fused
    # flash kernel and is ~8x faster than eager at 8192 tokens on an L4
    # (measured: 0.29 s vs 2.44 s per window). ModernBERT currently defaults to
    # SDPA, but pinning it guards against a future transformers release silently
    # falling back to eager (e.g. when flash-attn isn't importable).
    model = SentenceCompressor(base=base, attn_implementation="sdpa").to(device)
    state = _t.load(f"{ckpt_dir}/model.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, tok
