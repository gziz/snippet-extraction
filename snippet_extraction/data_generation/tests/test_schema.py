"""Row schema and resume/dedupe behavior of core.schema."""

import json

from data_generation.core.schema import (
    append_label,
    load_done_keys,
    load_done_qids,
    make_label_row,
    make_scrape_row,
)

LABEL_KW = dict(
    query_id=1,
    doc_id="D1",
    query="q",
    body="b",
    units=["u0", "u1"],
    relevant_unit_ids=[0],
    labeler_provider="azure",
    labeler_model="m",
    prompt_version="v2",
    prompt_hash="h",
    tokens_in=1,
    tokens_out=2,
    latency_s=0.1234,
    origin="qrel",
)

# Field order is part of the de-facto schema: existing files were written this
# way and humans diff them.
LABEL_FIELDS = [
    "query_id",
    "doc_id",
    "origin",
    "labeler_provider",
    "labeler_model",
    "prompt_version",
    "prompt_hash",
    "status",
    "query",
    "body",
    "units",
    "relevant_unit_ids",
    "n_units",
    "tokens_in",
    "tokens_out",
    "latency_s",
    "timestamp",
]

SCRAPE_FIELDS = [
    "query_id",
    "doc_id",
    "origin",
    "labeler_provider",
    "labeler_model",
    "prompt_version",
    "status",
    "query",
    "topic",
    "intent",
    "url",
    "title",
    "description",
    "body",
    "units",
    "relevant_unit_ids",
    "n_units",
    "n_kept",
    "position",
    "timestamp",
]


def test_label_row_field_order():
    row = make_label_row(**LABEL_KW)
    assert list(row.keys()) == LABEL_FIELDS
    assert row["n_units"] == 2
    assert row["latency_s"] == 0.123
    assert row["status"] == "ok"


def test_scrape_row_field_order():
    row = make_scrape_row(
        qid="syn_1",
        query="q",
        topic=None,
        intent=None,
        url="https://x",
        title="t",
        description=None,
        body="b",
        units=["u"],
        position=1,
        origin="tavily",
        fetcher_name="tavily-extract-v1",
        fetcher_version="tavily_v1",
        skipped_reason=None,
    )
    assert list(row.keys()) == SCRAPE_FIELDS
    assert row["doc_id"] == "https://x"
    assert row["labeler_provider"] == "tavily"
    assert row["status"] == "ok"


def test_scrape_row_skipped_reason_becomes_status():
    row = make_scrape_row(
        qid="1",
        query="q",
        topic=None,
        intent=None,
        url="u",
        title=None,
        description=None,
        body="b",
        units=[],
        position=None,
        origin="tavily",
        fetcher_name="f",
        fetcher_version="v",
        skipped_reason="skipped:long(400)",
    )
    assert row["status"] == "skipped:long(400)"


def test_append_and_resume_roundtrip(tmp_path):
    out = tmp_path / "labels.jsonl"
    append_label(out, **LABEL_KW)
    append_label(out, **{**LABEL_KW, "query_id": "syn_2", "doc_id": "D2"})

    done = load_done_keys(out, "azure", "m", "v2")
    # int and str query_ids both come back string-coerced
    assert done == {("1", "D1"), ("syn_2", "D2")}
    # different labeler identity -> nothing done yet
    assert load_done_keys(out, "bedrock", "m", "v2") == set()
    assert load_done_keys(out, "azure", "m", "v3") == set()

    assert load_done_qids(out) == {"1", "syn_2"}


def test_load_done_keys_assumes_azure_for_legacy_rows(tmp_path):
    out = tmp_path / "legacy.jsonl"
    row = make_label_row(**LABEL_KW)
    del row["labeler_provider"]
    out.write_text(json.dumps(row) + "\n")
    assert load_done_keys(out, "azure", "m", "v2") == {("1", "D1")}


def test_done_loaders_tolerate_corrupt_lines(tmp_path):
    out = tmp_path / "corrupt.jsonl"
    out.write_text('{"query_id": 7, "doc_id": "D"}\n{truncated\n')
    assert load_done_qids(out) == {"7"}


def test_missing_file_is_empty(tmp_path):
    assert load_done_qids(tmp_path / "nope.jsonl") == set()
    assert load_done_keys(tmp_path / "nope.jsonl", "a", "m", "v") == set()
