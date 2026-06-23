"""Token-level extraction metrics (provider- and splitter-independent).

We score how well a provider's extracted snippets cover the *gold* relevant
text for a query, using multiset (ROUGE-1 style) token overlap. No sentence
segmentation and no character-offset matching are involved: both the provider
prediction and the gold are reduced to token multisets and compared directly.

    precision = |pred ∩ gold| / |pred|
    recall    = |pred ∩ gold| / |gold|
    f1        = 2PR / (P + R)

`∩` is multiset intersection (min of per-token counts), so a snippet that
repeats a gold token twice only gets credit if the gold contains it twice.

Empty-set conventions (a "gold = nothing" case happens when the surfaced
document does not actually answer the query):
    - pred ∅ and gold ∅  -> P=R=F1=1.0   (correctly extracted nothing)
    - pred ∅, gold non-∅ -> P=1.0, R=0.0, F1=0.0
    - pred non-∅, gold ∅ -> P=0.0, R=1.0, F1=0.0  (extracted from non-answer doc)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, return alphanumeric word tokens."""
    return _WORD_RE.findall(_norm(text or ""))


@dataclass
class PRF:
    precision: float
    recall: float
    f1: float
    n_pred: int
    n_gold: int
    n_overlap: int


def _f1(p: float, r: float) -> float:
    return 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)


def token_prf(pred_text: str, gold_text: str) -> PRF:
    """Multiset token precision/recall/F1 of ``pred_text`` against ``gold_text``."""
    pred = Counter(tokenize(pred_text))
    gold = Counter(tokenize(gold_text))
    n_pred = sum(pred.values())
    n_gold = sum(gold.values())
    n_overlap = sum((pred & gold).values())

    if n_pred == 0 and n_gold == 0:
        return PRF(1.0, 1.0, 1.0, 0, 0, 0)
    if n_pred == 0:  # nothing predicted, but gold exists
        return PRF(1.0, 0.0, 0.0, 0, n_gold, 0)
    if n_gold == 0:  # predicted from a doc with no gold answer
        return PRF(0.0, 1.0, 0.0, n_pred, 0, 0)

    precision = n_overlap / n_pred
    recall = n_overlap / n_gold
    return PRF(precision, recall, _f1(precision, recall), n_pred, n_gold, n_overlap)


def join_snippets(snippets: list[str]) -> str:
    """Concatenate a provider's snippets / gold quotes into one text blob."""
    return "\n".join(s for s in snippets if s and s.strip())
