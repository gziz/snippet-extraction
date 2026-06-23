"""Sentence-level keep/drop head on top of an encoder.

Token logits are mean-pooled within each sentence span (via `sentence_ids`)
and BCE-with-logits is applied to the per-sentence relevance label. This
matches the eval metric and removes the label noise of marking every
stopword/punctuation token as positive.

Supports class-imbalance `pos_weight` and a sigmoid-prior bias init on the
head (see `head_bias_init`, `pos_weight`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

IGNORE_INDEX = -100


class SentenceCompressor(nn.Module):
    def __init__(
        self,
        base: str = "answerdotai/ModernBERT-base",
        dropout: float = 0.1,
        attn_implementation: str | None = None,
        pos_weight: float | None = None,
        head_bias_init: float | None = None,
    ):
        super().__init__()
        kwargs = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.encoder = AutoModel.from_pretrained(base, **kwargs)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(h, 1)
        if head_bias_init is not None:
            with torch.no_grad():
                self.head.bias.fill_(float(head_bias_init))
        if pos_weight is not None and pos_weight > 0:
            self.register_buffer("pos_weight", torch.tensor(float(pos_weight)), persistent=False)
        else:
            self.pos_weight = None

    @staticmethod
    def pool_sentence_logits(
        token_logits: torch.Tensor,  # (B, T)
        sentence_ids: torch.Tensor,  # (B, T), -1 outside any sentence
        max_units: int,
    ) -> torch.Tensor:
        """Mean-pool token logits per sentence -> (B, max_units)."""
        B, T = token_logits.shape
        device = token_logits.device
        ids = sentence_ids.clone().long()
        ids[(ids < 0) | (ids >= max_units)] = max_units  # ignore bucket
        sums = torch.zeros(B, max_units + 1, device=device, dtype=token_logits.dtype)
        counts = torch.zeros(B, max_units + 1, device=device, dtype=token_logits.dtype)
        sums.scatter_add_(1, ids, token_logits)
        counts.scatter_add_(1, ids, torch.ones_like(token_logits))
        sums = sums[:, :max_units]
        counts = counts[:, :max_units].clamp_min(1.0)
        return sums / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        sentence_ids: torch.Tensor | None = None,
        sent_labels: torch.Tensor | None = None,
    ):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(self.dropout(out.last_hidden_state)).squeeze(-1)  # (B, T)

        loss = None
        if sent_labels is not None and sentence_ids is not None:
            U = sent_labels.shape[1]
            sent_logits = self.pool_sentence_logits(logits, sentence_ids, U)
            mask = sent_labels != IGNORE_INDEX
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    sent_logits[mask],
                    sent_labels[mask].float(),
                    pos_weight=self.pos_weight,
                )
            else:
                loss = logits.sum() * 0.0
        elif labels is not None:
            mask = labels != IGNORE_INDEX
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits[mask],
                    labels[mask].float(),
                    pos_weight=self.pos_weight,
                )
            else:
                loss = logits.sum() * 0.0
        return logits, loss
