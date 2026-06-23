"""Dataset: load pre-processed split jsonl and build per-token keep/drop labels.

Input rows (produced by `snippets_training.prepro`) have:
  query: str
  units: list[str]                # sentences (in order, concatenated == body modulo spaces)
  relevant_unit_ids: list[int]    # indices of kept sentences
  n_tokens: int                   # full tokenization length (<= max_length)
  source: str                     # original raw jsonl path, for per-source eval breakdown

We tokenize (query, body) as a pair, then for each document token figure out
which sentence (unit) it belongs to via character offsets, and assign label =
1 if that sentence is relevant else 0. Query / special / pad tokens get label
-100 (ignored by the loss).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

# The body/span contract is shared with the inference runtime, so it lives in
# the torch-free core. Re-bound to the historical private name so the existing
# callers in this package keep working unchanged.
from snippets_common.spans import build_body_and_spans as _build_body_and_spans
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizerFast

IGNORE_INDEX = -100


@dataclass
class Example:
    query: str
    body: str
    unit_spans: list[tuple[int, int]]  # char spans in body
    relevant: set[int]  # relevant unit indices
    n_tokens: int  # untruncated tokenization length
    source: str  # original raw jsonl path


def load_jsonl(path: str | Path) -> list[Example]:
    """Load examples from a single pre-processed split jsonl.

    The file must have been produced by `snippets_training.prepro`, which guarantees the
    `n_tokens` and `source` fields and that every row fits within the
    tokenizer/max_length used at prepro time.
    """
    p = Path(path)
    out: list[Example] = []
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            units = d["units"]
            body, spans = _build_body_and_spans(units)
            out.append(
                Example(
                    query=d["query"],
                    body=body,
                    unit_spans=spans,
                    relevant=set(d.get("relevant_unit_ids", []) or []),
                    n_tokens=int(d["n_tokens"]),
                    source=str(d.get("source", p)),
                )
            )
    return out


def _tokenizer_tag(tokenizer: PreTrainedTokenizerFast) -> str:
    """Short stable identifier for a tokenizer.

    Hashes the serialized tokenizer json so swapping vocab/normalizer/etc.
    changes the tag. Used by prepro to stamp split artifacts.
    """
    try:
        body = tokenizer.backend_tokenizer.to_str().encode()
    except Exception:
        body = str(getattr(tokenizer, "name_or_path", "") or "tok").encode()
    return hashlib.sha1(body).hexdigest()[:10]


class CompressionDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 8192,
    ):
        assert tokenizer.is_fast, "Need a fast tokenizer for offset_mapping."
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length
        # n_tokens is stamped by prepro; bucketing reads this directly.
        self.lengths: list[int] = [ex.n_tokens for ex in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        ex = self.examples[idx]
        enc = self.tok(
            ex.query,
            ex.body,
            truncation="only_second",
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"]
        seq_ids = enc.sequence_ids()  # None for special, 0 query, 1 doc

        labels = [IGNORE_INDEX] * len(offsets)
        sentence_ids = [-1] * len(offsets)  # per-token unit index, -1 if not in any unit
        # Precompute sentence index of each char by walking spans (sorted, disjoint-ish).
        spans = ex.unit_spans
        rel = ex.relevant

        for i, ((s, e), sid) in enumerate(zip(offsets, seq_ids)):
            if sid != 1 or e <= s:
                continue
            # Find which unit this token sits in (use midpoint).
            mid = (s + e) // 2
            # Linear scan is fine — n_units typically small (~20).
            unit_idx = -1
            for j, (us, ue) in enumerate(spans):
                if us <= mid < ue:
                    unit_idx = j
                    break
                if mid < us:
                    break
            if unit_idx < 0:
                # Token sits in inter-sentence whitespace; skip.
                continue
            labels[i] = 1 if unit_idx in rel else 0
            sentence_ids[i] = unit_idx

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
            "sentence_ids": sentence_ids,
            "relevant": sorted(rel),
            "n_units": len(spans),
        }


@dataclass
class Collator:
    tokenizer: PreTrainedTokenizerFast

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = self.tokenizer.pad_token_id

        def pad(seq, value, n):
            return seq + [value] * (n - len(seq))

        input_ids = torch.tensor(
            [pad(b["input_ids"], pad_id, max_len) for b in batch], dtype=torch.long
        )
        attn = torch.tensor([pad(b["attention_mask"], 0, max_len) for b in batch], dtype=torch.long)
        labels = torch.tensor(
            [pad(b["labels"], IGNORE_INDEX, max_len) for b in batch], dtype=torch.long
        )
        sent_ids = torch.tensor(
            [pad(b["sentence_ids"], -1, max_len) for b in batch], dtype=torch.long
        )
        # Per-sentence relevance labels, padded to max n_units in the batch.
        max_units = max(b["n_units"] for b in batch) if batch else 0
        sent_labels = torch.full((len(batch), max_units), IGNORE_INDEX, dtype=torch.long)
        for i, b in enumerate(batch):
            nu = b["n_units"]
            if nu == 0:
                continue
            rel = set(b["relevant"])
            for j in range(nu):
                sent_labels[i, j] = 1 if j in rel else 0
        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "sentence_ids": sent_ids,
            "sent_labels": sent_labels,
            "relevant": [b["relevant"] for b in batch],
            "n_units": [b["n_units"] for b in batch],
        }


class LengthBucketSampler(Sampler[list[int]]):
    """Batch sampler that groups examples of similar length to minimize padding.

    Algorithm: shuffle all indices, partition into "mega-batches" of size
    `bucket_size * batch_size`, sort each mega-batch by length, slice into
    batches, then shuffle the batch order. Standard fairseq/HF recipe.

    If `max_tokens` is set, batch size is shrunk per-bucket so that
    `batch_size * max_len_in_batch <= max_tokens`. This keeps memory bounded
    when long sequences are present.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size: int = 50,
        seed: int = 0,
        drop_last: bool = False,
        max_tokens: int | None = None,
        num_replicas: int = 1,
        rank: int = 0,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.seed = seed
        self.drop_last = drop_last
        self.max_tokens = max_tokens
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        if not (0 <= self.rank < self.num_replicas):
            raise ValueError(f"rank {rank} out of range for world_size {num_replicas}")
        self.epoch = 0
        self._cached_batches: list[list[int]] | None = None
        self._cached_epoch: int | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._cached_batches = None

    def _build(self) -> list[list[int]]:
        if self._cached_batches is not None and self._cached_epoch == self.epoch:
            return self._cached_batches
        n = len(self.lengths)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        order = torch.randperm(n, generator=g).tolist() if self.shuffle else list(range(n))

        mega = self.bucket_size * self.batch_size
        batches: list[list[int]] = []
        for i in range(0, n, mega):
            chunk = order[i : i + mega]
            chunk.sort(key=lambda idx: self.lengths[idx])
            j = 0
            while j < len(chunk):
                if self.max_tokens is not None:
                    # Pack greedily so that bs * max_len_in_batch <= max_tokens.
                    end = j
                    cur_max = 0
                    while end < len(chunk) and (end - j) < self.batch_size:
                        new_max = max(cur_max, self.lengths[chunk[end]])
                        if (end - j + 1) * new_max > self.max_tokens and end > j:
                            break
                        cur_max = new_max
                        end += 1
                    b = chunk[j:end]
                    j = end
                else:
                    b = chunk[j : j + self.batch_size]
                    j += self.batch_size
                if self.drop_last and len(b) < self.batch_size:
                    continue
                batches.append(b)

        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        if self.num_replicas > 1:
            # Drop the tail so every rank gets exactly the same number of
            # batches; avoids DDP hangs from uneven step counts.
            usable = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[:usable]
            batches = batches[self.rank :: self.num_replicas]

        self._cached_batches = batches
        self._cached_epoch = self.epoch
        return batches

    def __iter__(self):
        for b in self._build():
            yield b

    def __len__(self) -> int:
        return len(self._build())
