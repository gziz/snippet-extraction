"""Windowing, budgeted selection, and snippet rendering for snippets_common.

These are the torch-free selection/rendering primitives: ``pack_windows`` must
cover every unit at unit boundaries, ``select_under_budget`` is greedy-by-score
under a token budget and returns document order, and ``render_snippet`` only
marks elided gaps between non-adjacent runs.
"""

import pytest
from snippets_common.windowing import pack_windows, render_snippet, select_under_budget


def _word_count(text: str) -> int:
    return len(text.split())


def test_pack_windows_breaks_only_at_unit_boundaries_and_keeps_all():
    lens = [3, 3, 3, 3]
    windows = pack_windows(lens, capacity=6)
    assert windows == [(0, 2), (2, 4)]
    # Windows are consecutive, gapless, and cover every unit exactly once.
    assert windows[0][0] == 0
    assert windows[-1][1] == len(lens)
    for (_, prev_end), (next_start, _) in zip(windows, windows[1:]):
        assert next_start == prev_end


def test_pack_windows_oversized_unit_gets_its_own_window():
    # The middle unit alone exceeds capacity: it must not be dropped or merged.
    windows = pack_windows([2, 10, 2], capacity=5)
    assert windows == [(0, 1), (1, 2), (2, 3)]
    covered = [i for start, end in windows for i in range(start, end)]
    assert covered == [0, 1, 2]


def test_pack_windows_leading_oversized_unit():
    assert pack_windows([10, 2, 2], capacity=5) == [(0, 1), (1, 3)]


def test_pack_windows_empty_input():
    assert pack_windows([], capacity=5) == []


def test_pack_windows_nonpositive_capacity_raises():
    with pytest.raises(ValueError):
        pack_windows([1, 2, 3], capacity=0)
    with pytest.raises(ValueError):
        pack_windows([1, 2, 3], capacity=-4)


def test_select_under_budget_greedy_by_score_within_budget():
    units = ["aa", "bb", "cc", "dd"]
    scores = [0.9, 0.3, 0.8, 0.7]
    # Budget fits 2 single-token units; descending score picks 0 then 2.
    kept = select_under_budget(
        units, scores, budget_tokens=2, count_tokens=lambda s: 1, min_score=0.1
    )
    assert kept == [0, 2]


def test_select_under_budget_returns_document_order():
    units = ["x", "y", "z"]
    scores = [0.2, 0.9, 0.5]  # rank order 1,2,0 but result must be sorted
    kept = select_under_budget(
        units, scores, budget_tokens=10, count_tokens=lambda s: 1, min_score=0.1
    )
    assert kept == [0, 1, 2]


def test_select_under_budget_applies_min_score_cutoff():
    units = ["a", "b", "c"]
    scores = [0.9, 0.05, 0.8]  # 'b' is below the cutoff
    kept = select_under_budget(
        units, scores, budget_tokens=10, count_tokens=lambda s: 1, min_score=0.1
    )
    assert kept == [0, 2]


def test_select_under_budget_skips_unfit_but_keeps_smaller_lower_ranked():
    # Highest score is too big to fit; a smaller, lower-ranked unit still fits.
    units = ["big one here", "x", "y"]
    scores = [0.9, 0.6, 0.5]
    kept = select_under_budget(
        units, scores, budget_tokens=1, count_tokens=_word_count, min_score=0.1
    )
    assert kept == [1]


def test_select_under_budget_length_mismatch_raises():
    with pytest.raises(ValueError):
        select_under_budget(["a", "b"], [0.5], budget_tokens=10, count_tokens=lambda s: 1)


def test_render_snippet_joins_adjacent_runs_directly():
    snippet = render_snippet(["u0", "u1", "u2"], [0, 1, 2])
    assert snippet == "u0 u1 u2"


def test_render_snippet_inserts_gap_between_non_adjacent_runs():
    snippet = render_snippet(["u0", "u1", "u2", "u3", "u4"], [0, 1, 3])
    assert snippet == "u0 u1 [...] u3"


def test_render_snippet_custom_gap_marker_and_multiple_gaps():
    snippet = render_snippet(["a", "b", "c", "d", "e"], [0, 2, 4], gap="<<gap>>")
    assert snippet == "a <<gap>> c <<gap>> e"


def test_render_snippet_empty_selection():
    assert render_snippet(["a", "b"], []) == ""
