"""End-to-end labeling stage with a fake labeler: rows written, dedupe, resume.

This is the integration test for the path that produces training labels:
pairs file -> label_units.main_async -> label rows, twice (second run must be
a no-op thanks to the (provider, model, prompt_version) resume key).
"""

import asyncio
import json
from argparse import Namespace

from data_generation.core.labeling import LabelResult
from data_generation.core.labeling.prompts import PROMPTS
from data_generation.pipelines import label_units


class FakeLabeler:
    provider = "fake"
    deployment = "fake-model"

    def __init__(self):
        self.calls = 0

    async def label(self, query, units):
        self.calls += 1
        status = "content_filter" if "blocked" in query else "ok"
        ids = [0] if status == "ok" else None
        return LabelResult(
            relevant_unit_ids=ids,
            tokens_in=100,
            tokens_out=5,
            latency_s=0.01,
            raw="{}",
            status=status,
        )


def _pairs(path, rows):
    with path.open("w") as f:
        for r in rows:
            base = {"status": "ok", "origin": "tavily", "body": "b", "units": ["u0", "u1"], **r}
            f.write(json.dumps(base) + "\n")
    return path


def _args(in_path, out):
    return Namespace(
        in_path=in_path, out=out, provider="fake", prompt_version="v2", concurrency=2, limit=0
    )


def _run(args, labeler):
    # main_async builds its labeler via the factory; inject the fake there.
    orig = label_units.make_async_labeler
    label_units.make_async_labeler = lambda provider, version="v2": labeler
    try:
        asyncio.run(label_units.main_async(args))
    finally:
        label_units.make_async_labeler = orig


def test_labels_written_with_full_labeler_identity(tmp_path):
    pairs = _pairs(
        tmp_path / "pairs.jsonl",
        [
            {"query_id": "syn_1", "doc_id": "D1", "query": "good query"},
            {"query_id": 2, "doc_id": "D2", "query": "another good query"},
            {"query_id": "syn_3", "doc_id": "D3", "query": "blocked query"},
            # must be ignored: not judgeable
            {"query_id": "syn_4", "doc_id": "D4", "query": "q", "status": "skipped:long(400)"},
            {"query_id": "syn_5", "doc_id": "D5", "query": "q", "units": []},
        ],
    )
    out = tmp_path / "labeled.jsonl"
    labeler = FakeLabeler()
    _run(_args(pairs, out), labeler)

    rows = {r["doc_id"]: r for r in (json.loads(l) for l in out.open())}
    assert set(rows) == {"D1", "D2", "D3"}
    assert labeler.calls == 3

    ok = rows["D1"]
    assert ok["relevant_unit_ids"] == [0]
    assert ok["labeler_provider"] == "fake"
    assert ok["labeler_model"] == "fake-model"
    assert ok["prompt_version"] == "v2"
    assert ok["prompt_hash"] == PROMPTS["v2"].prompt_hash()
    assert ok["origin"] == "tavily"
    assert ok["n_units"] == 2

    # failures are recorded (status + null label), never silently dropped
    failed = rows["D3"]
    assert failed["status"] == "content_filter"
    assert failed["relevant_unit_ids"] is None


def test_rerun_is_a_noop_and_mixed_qid_types_dedupe(tmp_path):
    pairs = _pairs(
        tmp_path / "pairs.jsonl",
        [
            {"query_id": "syn_1", "doc_id": "D1", "query": "q1"},
            {"query_id": 2, "doc_id": "D2", "query": "q2"},  # int qid on purpose
        ],
    )
    out = tmp_path / "labeled.jsonl"

    labeler = FakeLabeler()
    _run(_args(pairs, out), labeler)
    assert labeler.calls == 2
    n_rows = sum(1 for _ in out.open())

    labeler2 = FakeLabeler()
    _run(_args(pairs, out), labeler2)
    assert labeler2.calls == 0, "resume must skip already-labeled pairs"
    assert sum(1 for _ in out.open()) == n_rows


def test_different_labeler_identity_relabels(tmp_path):
    pairs = _pairs(
        tmp_path / "pairs.jsonl",
        [
            {"query_id": "syn_1", "doc_id": "D1", "query": "q1"},
        ],
    )
    out = tmp_path / "labeled.jsonl"
    _run(_args(pairs, out), FakeLabeler())

    other = FakeLabeler()
    other.deployment = "other-model"
    _run(_args(pairs, out), other)
    assert other.calls == 1, "a new labeler model is a new label pass, not a dupe"
    assert sum(1 for _ in out.open()) == 2
