"""The torch-free numpy pooling helper ``inference._pool_window``.

``_pool_window`` turns per-token probabilities for one tokenized sequence into
per-unit ``(sent_scores, mean_probs, kept)``. It pools only body tokens
(``seq_id == 1``), assigns each token to the unit whose char span contains its
midpoint, scores a unit by the fraction of its tokens above ``token_threshold``,
and keeps units whose score meets ``sentence_threshold``. Spans come from the
real ``build_body_and_spans`` so the assignment geometry is exercised end to end.
"""

from snippets_common.spans import build_body_and_spans
from snippets_runtime.inference import _pool_window


def _make_seq(units, unit_probs):
    """Build (offsets, seq_ids, probs) for one query+body sequence.

    Layout: [CLS] query [SEP] <one token per unit, at its char span> [SEP].
    Special/query tokens carry ``seq_id != 1`` and zero-width offsets so they
    are excluded from pooling; each body token's offsets are the unit's span.
    """
    body, spans = build_body_and_spans(units)
    offsets = [(0, 0), (0, 1), (0, 0)]  # CLS, one query token, SEP
    seq_ids = [None, 0, None]
    probs = [0.0, 0.0, 0.0]
    for (start, end), prob in zip(spans, unit_probs):
        offsets.append((start, end))
        seq_ids.append(1)
        probs.append(prob)
    offsets.append((0, 0))  # trailing SEP
    seq_ids.append(None)
    probs.append(0.0)
    return body, spans, offsets, seq_ids, probs


def test_only_body_tokens_are_pooled():
    units = ["alpha", "beta"]
    _, spans, offsets, seq_ids, probs = _make_seq(units, [0.9, 0.2])
    # The query token sits at offset (0,1) which overlaps unit 0's span, but its
    # seq_id is 0, so it must be excluded from unit 0's score entirely.
    sent_scores, mean_probs, kept = _pool_window(
        offsets, seq_ids, probs, spans, len(units), 0.5, 0.5
    )
    assert sent_scores == [1.0, 0.0]
    assert mean_probs == [0.9, 0.2]
    assert kept == [0]


def test_token_assigned_to_unit_containing_its_midpoint():
    units = ["one", "two", "three"]
    body, spans = build_body_and_spans(units)
    # A single body token whose span midpoint falls inside unit 2's char range.
    s2, e2 = spans[2]
    offsets = [(0, 0), (s2, e2)]
    seq_ids = [None, 1]
    probs = [0.0, 0.8]
    sent_scores, mean_probs, kept = _pool_window(
        offsets, seq_ids, probs, spans, len(units), 0.5, 0.5
    )
    assert mean_probs[0] == 0.0 and mean_probs[1] == 0.0
    assert mean_probs[2] == 0.8
    assert kept == [2]


def test_sent_score_is_fraction_above_token_threshold():
    units = ["aaaa"]
    body, spans = build_body_and_spans(units)
    s, e = spans[0]
    # Four body tokens all inside the single unit; 3 of 4 are above threshold.
    offsets = [(0, 0), (s, s + 1), (s + 1, s + 2), (s + 2, s + 3), (s + 3, e)]
    seq_ids = [None, 1, 1, 1, 1]
    probs = [0.0, 0.9, 0.9, 0.1, 0.9]
    sent_scores, mean_probs, kept = _pool_window(
        offsets, seq_ids, probs, spans, len(units), 0.5, 0.5
    )
    assert sent_scores[0] == 0.75
    assert abs(mean_probs[0] - (0.9 + 0.9 + 0.1 + 0.9) / 4) < 1e-9
    assert kept == [0]


def test_kept_requires_sentence_threshold():
    units = ["alpha", "beta"]
    _, spans, offsets, seq_ids, probs = _make_seq(units, [0.9, 0.4])
    # Both tokens above token_threshold individually would give score 1.0, but
    # here unit1's only token (0.4) is below token_threshold -> score 0.0.
    sent_scores, mean_probs, kept = _pool_window(
        offsets, seq_ids, probs, spans, len(units), 0.5, 0.6
    )
    assert sent_scores == [1.0, 0.0]
    assert kept == [0]


def test_n_units_zero_returns_three_empty_lists():
    assert _pool_window([(0, 0)], [None], [0.0], [], 0, 0.5, 0.5) == ([], [], [])


def test_window_with_no_body_tokens_returns_zeroed_scores_and_no_kept():
    units = ["alpha", "beta"]
    _, spans = build_body_and_spans(units)
    # Only specials/query tokens, no seq_id == 1 token anywhere.
    offsets = [(0, 0), (0, 1), (0, 0)]
    seq_ids = [None, 0, None]
    probs = [0.0, 0.0, 0.0]
    sent_scores, mean_probs, kept = _pool_window(
        offsets, seq_ids, probs, spans, len(units), 0.5, 0.5
    )
    assert sent_scores == [0.0, 0.0]
    assert mean_probs == [0.0, 0.0]
    assert kept == []
