"""The unit -> body span contract: ``build_body_and_spans``.

Spans are the single source of truth shared by training and inference, so
``body[start:end]`` must reproduce each unit exactly, ranges must be half-open
and non-overlapping, and units must be joined by exactly one space.
"""

from snippets_common.spans import build_body_and_spans


def test_empty_list_returns_empty_body_and_spans():
    assert build_body_and_spans([]) == ("", [])


def test_single_unit_span_covers_whole_body():
    body, spans = build_body_and_spans(["hello world"])
    assert body == "hello world"
    assert spans == [(0, 11)]
    assert body[spans[0][0] : spans[0][1]] == "hello world"


def test_units_joined_with_single_spaces():
    body, spans = build_body_and_spans(["a", "b", "c"])
    assert body == "a b c"
    assert spans == [(0, 1), (2, 3), (4, 5)]


def test_spans_round_trip_each_unit():
    units = ["First sentence.", "Second one here.", "Third!"]
    body, spans = build_body_and_spans(units)
    assert len(spans) == len(units)
    for unit, (start, end) in zip(units, spans):
        assert body[start:end] == unit


def test_spans_are_half_open_and_non_overlapping():
    units = ["alpha", "beta", "gamma"]
    body, spans = build_body_and_spans(units)
    for start, end in spans:
        assert 0 <= start < end <= len(body)
    # Each gap between consecutive units is exactly one space.
    for (prev_start, prev_end), (next_start, next_end) in zip(spans, spans[1:]):
        assert next_start == prev_end + 1
        assert body[prev_end:next_start] == " "


def test_multibyte_unicode_units_round_trip():
    units = ["café ☕", "naïve façade", "日本語 のテスト", "emoji 😀🚀"]
    body, spans = build_body_and_spans(units)
    # Char (not byte) indexing must reproduce each unit verbatim.
    for unit, (start, end) in zip(units, spans):
        assert body[start:end] == unit
        assert end - start == len(unit)
