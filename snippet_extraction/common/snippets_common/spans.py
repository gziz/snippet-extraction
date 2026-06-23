"""Unit -> body assembly: the single source of truth for how a list of units
becomes one ``body`` string plus the char spans of each unit.

Torch-free on purpose: both the training data pipeline (``snippets_training``)
and the inference runtime (``snippets_runtime``) must build the body and pool
token scores against *identical* spans, or the model would see a different unit
layout at inference than it was trained on. Keeping this here makes that
contract impossible to drift.
"""

from __future__ import annotations


def build_body_and_spans(units: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate units with single spaces and return char spans of each unit."""
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, u in enumerate(units):
        if i > 0:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(u)
        cursor += len(u)
        spans.append((start, cursor))
    return "".join(parts), spans
