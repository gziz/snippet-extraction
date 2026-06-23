"""Align Exa highlights to our segment.py units.

Exa returns highlights as ONE string per URL with fragments joined by '\n[...]\n'.
Each fragment is a real substring (or near-substring) of the doc body. We:

1. Split the highlight on `[...]` markers to get fragments.
2. Segment the doc body with `segment.py` -> list of units.
3. For each fragment, find the best-matching unit(s) by token overlap.
4. Return the sorted set of matched unit IDs as `relevant_unit_ids`.

Designed to be conservative: unmatched fragments are dropped (not guessed).
Returns alignment stats so we can monitor drop-rate at scale.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_HL_SPLIT = re.compile(r"\s*\n?\s*\[\s*\.\.\.\s*\]\s*\n?\s*")
_WORD_RE = re.compile(r"[a-z0-9]+")


def split_fragments(highlight: str) -> list[str]:
    parts = [p.strip() for p in _HL_SPLIT.split(highlight or "")]
    return [p for p in parts if len(p) >= 8]  # drop tiny noise like ", etc."


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return s.lower()


def _tokens(s: str) -> list[str]:
    return _WORD_RE.findall(_norm(s))


def _shingles(toks: list[str], n: int = 5) -> set[tuple[str, ...]]:
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


@dataclass
class AlignStats:
    n_fragments: int
    n_matched: int
    n_dropped: int
    matched_unit_ids: list[int]


def align(
    fragments: list[str],
    units: list[str],
    *,
    min_unit_coverage: float = 0.5,
    min_frag_recall: float = 0.5,
    shingle_n: int = 5,
    min_frag_tokens: int = 6,
) -> AlignStats:
    """Match fragments to units using 5-gram overlap.

    For each fragment, keep every unit that satisfies EITHER:
      - (shared shingles / unit shingles) >= min_unit_coverage   (this unit is
        mostly subsumed by the fragment), OR
      - the fragment's shingles overlap that unit substantially AND, taken
        together, those unit shingles cover the fragment (>= min_frag_recall).

    Concretely: we greedily add units in decreasing single-unit overlap until
    cumulative fragment coverage reaches min_frag_recall (or no unit adds >0).

    Short fragments (< min_frag_tokens) and ``min_frag_recall`` failures
    fall back to a substring containment check, then are dropped.
    """
    unit_shingles = [_shingles(_tokens(u), shingle_n) for u in units]
    unit_norm = [_norm(u) for u in units]

    matched: set[int] = set()
    n_matched = 0
    for frag in fragments:
        ftoks = _tokens(frag)
        if len(ftoks) < min_frag_tokens:
            # Junk fragment ("Thus, the", "16 project_dir"). Skip outright.
            continue

        fshing = _shingles(ftoks, shingle_n)
        if not fshing:
            continue

        # Per-unit overlap stats.
        scored: list[tuple[int, set, int]] = []  # (unit_idx, intersection, |intersect|)
        for i, us in enumerate(unit_shingles):
            if not us:
                continue
            inter = fshing & us
            if not inter:
                continue
            scored.append((i, inter, len(inter)))

        if not scored:
            # Fallback: substring containment (rare).
            needle = _norm(frag)[:80]
            for i, un in enumerate(unit_norm):
                if needle and needle in un:
                    matched.add(i)
                    n_matched += 1
                    break
            continue

        scored.sort(key=lambda x: x[2], reverse=True)

        # Strong subsumption: any unit ≥ min_unit_coverage of its own shingles in frag.
        strong_added = False
        for i, inter, _ in scored:
            if not unit_shingles[i]:
                continue
            if len(inter) / len(unit_shingles[i]) >= min_unit_coverage:
                matched.add(i)
                strong_added = True

        # Greedy coverage of the fragment.
        covered: set = set()
        chosen: list[int] = []
        for i, inter, _ in scored:
            new = inter - covered
            if not new:
                continue
            covered |= new
            chosen.append(i)
            if len(covered) / len(fshing) >= min_frag_recall:
                break

        if len(covered) / len(fshing) >= min_frag_recall:
            matched.update(chosen)
            n_matched += 1
        elif strong_added:
            n_matched += 1

    return AlignStats(
        n_fragments=len(fragments),
        n_matched=n_matched,
        n_dropped=len(fragments) - n_matched,
        matched_unit_ids=sorted(matched),
    )
