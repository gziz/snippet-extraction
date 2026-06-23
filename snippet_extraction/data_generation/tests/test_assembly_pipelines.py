"""Dataset assembly: safely_merge_jsons (corpus + aggregate), join_queries_units.

These stages decide what ends up in the training file; the tests pin their
filtering/dedup/projection rules on tiny synthetic inputs.
"""

import json
import sys

from data_generation.pipelines import join_queries_units
from data_generation.pipelines.safely_merge_jsons import DOC_FIELDS, merge


def _write(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _read(path):
    return [json.loads(l) for l in path.open()]


def _scrape_row(doc_id, body="some body", **extra):
    return {
        "doc_id": doc_id,
        "origin": "tavily",
        "query": "q",
        "query_id": "1",
        "topic": None,
        "intent": None,
        "url": doc_id,
        "title": "t",
        "description": None,
        "body": body,
        "position": 1,
        "units": ["u"],
        "status": "ok",
        **extra,
    }


# ---------------------------------------------------------------------------
# safely_merge_jsons -- corpus consolidation (dedupe + require + keep)
# ---------------------------------------------------------------------------


def _consolidate(inputs, out):
    merge(inputs, out, dedupe_key="doc_id", require_field="body", keep_fields=DOC_FIELDS)


def test_consolidate_dedups_first_seen_and_drops_bodyless(tmp_path):
    src_a = _write(
        tmp_path / "a.jsonl",
        [
            _scrape_row("D1", body="body from A"),
            _scrape_row("D2", body=""),  # dropped: no body
        ],
    )
    src_b = _write(
        tmp_path / "b.jsonl",
        [
            _scrape_row("D1", body="body from B (duplicate, must lose)"),
            _scrape_row("D3"),
        ],
    )
    out = tmp_path / "corpus.jsonl"
    _consolidate([src_a, src_b, tmp_path / "missing.jsonl"], out)  # missing source: skipped

    rows = {r["doc_id"]: r for r in _read(out)}
    assert set(rows) == {"D1", "D3"}
    assert rows["D1"]["body"] == "body from A", "first-seen must win"


def test_consolidate_projects_doc_fields_only(tmp_path):
    src = _write(
        tmp_path / "a.jsonl",
        [
            _scrape_row("D1", units=["u1", "u2"], relevant_unit_ids=[0], n_kept=1),
        ],
    )
    out = tmp_path / "corpus.jsonl"
    _consolidate([src], out)
    (row,) = _read(out)
    # labeled-pass fields (units, relevant_unit_ids, ...) must not leak into the corpus
    assert set(row) == set(DOC_FIELDS)


# ---------------------------------------------------------------------------
# join_queries_units
# ---------------------------------------------------------------------------


def test_join_pairs_queries_with_source_doc_units(tmp_path, monkeypatch, capsys):
    corpus = _write(
        tmp_path / "corpus.jsonl",
        [
            {
                "doc_id": "D1",
                "origin": "tavily",
                "url": "https://x",
                "title": "t",
                "body": "b",
                "units": ["u0", "u1"],
            },
            {
                "doc_id": "D2",
                "origin": "tavily",
                "url": "https://y",
                "title": "t",
                "body": "b",
                "units": [],
            },  # no units -> join must drop
        ],
    )
    queries = _write(
        tmp_path / "queries.jsonl",
        [
            {"qid": "syn_1", "query": "q1", "source_doc_id": "D1", "intent": "procedural"},
            {"qid": "syn_2", "query": "q2", "source_doc_id": "D2"},  # no units
            {"qid": "syn_3", "query": "q3", "source_doc_id": "D_missing"},  # no doc
        ],
    )
    out = tmp_path / "pairs.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "join_queries_units",
            "--queries",
            str(queries),
            "--corpus",
            str(corpus),
            "--out",
            str(out),
        ],
    )
    join_queries_units.main()

    rows = _read(out)
    assert len(rows) == 1
    row = rows[0]
    assert row["query_id"] == "syn_1" and row["doc_id"] == "D1"
    assert row["units"] == ["u0", "u1"]
    assert row["status"] == "ok"  # ready for label_units
    assert row["relevant_unit_ids"] == []  # not labeled yet
    assert row["source_doc_id"] == "D1"


# ---------------------------------------------------------------------------
# safely_merge_jsons -- label aggregation (concatenate + drop body)
# ---------------------------------------------------------------------------


def test_aggregate_concatenates_and_drops_body(tmp_path):
    msmarco = _write(
        tmp_path / "msmarco.jsonl",
        [
            {"origin": "qrel", "query_id": 1, "units": ["u"]},
            {"origin": "bm25_top100", "query_id": 2, "units": ["u"]},
        ],
    )
    qasper = _write(
        tmp_path / "qasper.jsonl",
        [
            {"origin": "qasper", "query_id": "qa1", "units": ["u"]},
        ],
    )
    fc2 = _write(
        tmp_path / "fc2.jsonl",
        [
            {
                "origin": "tavily",
                "query_id": "syn_1",
                "units": ["u"],
                "body": "HUGE BODY THAT MUST BE DROPPED",
            },
        ],
    )
    out = tmp_path / "agg.jsonl"
    merge([msmarco, qasper, fc2], out, drop_fields=["body"])

    rows = _read(out)
    origins = sorted(r["origin"] for r in rows)
    assert origins == ["bm25_top100", "qasper", "qrel", "tavily"]
    fc2_row = next(r for r in rows if r["origin"] == "tavily")
    assert "body" not in fc2_row, "aggregate must not carry document bodies"
