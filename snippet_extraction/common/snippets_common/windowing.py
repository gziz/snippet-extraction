"""Pure (torch-free) helpers for long-document inference.

Three pieces, all operating on sentence units:

- ``pack_windows``: split a unit list into consecutive windows that fit the
  encoder context, breaking only at unit boundaries.
- ``select_under_budget``: turn per-unit scores into a kept-set capped at a
  token budget (rank by score, fill greedily, re-sort to document order).
- ``render_snippet``: join kept units in document order, marking elided gaps
  with a separator so the agent can tell the snippet is non-contiguous.

Kept torch-free so the selection/rendering logic can be unit-tested (and
reused by eval harnesses) without the model stack installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def pack_windows(unit_token_lens: Sequence[int], capacity: int) -> list[tuple[int, int]]:
    """Pack consecutive units into windows of at most ``capacity`` tokens.

    Returns ``(start, end)`` index ranges (end exclusive) covering all units in
    order. A single unit larger than ``capacity`` gets its own window — the
    tokenizer's truncation handles it downstream rather than dropping it here.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    windows: list[tuple[int, int]] = []
    start = 0
    used = 0
    for i, n in enumerate(unit_token_lens):
        if i > start and used + n > capacity:
            windows.append((start, i))
            start, used = i, 0
        used += n
    if start < len(unit_token_lens):
        windows.append((start, len(unit_token_lens)))
    return windows


def select_under_budget(
    units: Sequence[str],
    scores: Sequence[float],
    budget_tokens: int,
    count_tokens: Callable[[str], int],
    *,
    min_score: float = 0.1,
) -> list[int]:
    """Pick unit indices maximizing score under a token budget.

    Greedy by descending score; units that don't fit are skipped (smaller,
    lower-ranked units may still fit). ``min_score`` stops the fill from
    padding the budget with junk when little in the document is relevant.
    Returned indices are sorted back into document order.
    """
    if len(units) != len(scores):
        raise ValueError(f"{len(units)} units vs {len(scores)} scores")
    order = sorted(range(len(units)), key=lambda j: scores[j], reverse=True)
    kept: list[int] = []
    used = 0
    for j in order:
        if scores[j] < min_score:
            break  # order is descending; everything after is junk too
        n = count_tokens(units[j])
        if used + n > budget_tokens:
            continue
        kept.append(j)
        used += n
    return sorted(kept)


def render_snippet(
    units: Sequence[str],
    kept_indices: Sequence[int],
    *,
    gap: str = "[...]",
) -> str:
    """Join kept units in document order, separating non-adjacent runs with ``gap``."""
    parts: list[str] = []
    prev = None
    for j in kept_indices:
        if prev is not None and j != prev + 1:
            parts.append(gap)
        parts.append(units[j])
        prev = j
    return " ".join(parts)
