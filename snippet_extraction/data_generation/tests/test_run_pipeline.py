"""run_pipeline orchestrator: config parsing, stage slicing, end-to-end wiring.

The three network-bound stages (scrape, synth, label) are stubbed; the
deterministic stages (corpus merge, segment, join, aggregate merge) run for
real, so this exercises the actual wiring + manifest + fail-loud behavior.
"""

import asyncio
import json

import pytest
from data_generation.pipelines import run_pipeline as rp


def _write(path, rows):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _read(path):
    return [json.loads(l) for l in path.open()]


def _cfg(tmp_path):
    queries = _write(tmp_path / "queries.jsonl", [{"qid": "q1", "query": "hello"}])
    return rp.Config(queries=queries, run_dir=tmp_path / "run")


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------


def test_config_from_yaml_resolves_paths_and_rejects_unknown(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "queries: data/queries/seed.jsonl\n"
        "run_dir: data/labels/run_x\n"
        "retriever: tavily\n"
        "extra_inputs:\n  - data/labels/extra.jsonl\n"
    )
    cfg = rp.Config.from_yaml(cfg_path)
    assert cfg.queries.is_absolute() and cfg.queries.name == "seed.jsonl"
    assert cfg.run_dir.name == "run_x"
    assert cfg.extra_inputs[0].name == "extra.jsonl"

    bad = tmp_path / "bad.yaml"
    bad.write_text("queries: a\nrun_dir: b\nnope: 1\n")
    with pytest.raises(SystemExit):
        rp.Config.from_yaml(bad)

    missing = tmp_path / "missing.yaml"
    missing.write_text("retriever: tavily\n")
    with pytest.raises(SystemExit):
        rp.Config.from_yaml(missing)


# ---------------------------------------------------------------------------
# stage construction + slicing
# ---------------------------------------------------------------------------


def test_build_stages_derives_numbered_paths(tmp_path):
    stages = rp.build_stages(_cfg(tmp_path))
    assert [s.key for s in stages] == rp.STAGE_ORDER
    names = [s.out.name for s in stages]
    assert names == [
        "01_scrape.jsonl",
        "02_corpus.jsonl",
        "03_segmented.jsonl",
        "04_synth_queries.jsonl",
        "05_pairs.jsonl",
        "06_labeled.jsonl",
        "07_aggregate.jsonl",
    ]
    # aggregate folds in extra_inputs alongside the labeled output
    cfg = _cfg(tmp_path)
    cfg.extra_inputs = [tmp_path / "ms.jsonl"]
    agg = rp.build_stages(cfg)[-1]
    assert agg.inputs[0].name == "06_labeled.jsonl"
    assert agg.inputs[1].name == "ms.jsonl"


def test_selected_slices_inclusive_and_validates_order(tmp_path):
    stages = rp.build_stages(_cfg(tmp_path))
    sel = rp._selected(stages, "segment", "label")
    assert [s.key for s in sel] == ["segment", "synth", "join", "label"]
    with pytest.raises(SystemExit):
        rp._selected(stages, "label", "segment")


# ---------------------------------------------------------------------------
# end-to-end (network stages stubbed, deterministic stages real)
# ---------------------------------------------------------------------------


def _doc_row(doc_id, body):
    return {
        "doc_id": doc_id,
        "origin": "tavily",
        "query": "seed q",
        "query_id": "q1",
        "topic": None,
        "intent": "procedural",
        "url": f"https://x/{doc_id}",
        "title": "t",
        "description": None,
        "body": body,
        "position": 1,
    }


def _patch_network_stages(monkeypatch):
    async def fake_scrape(*, out_path, **kw):
        _write(
            out_path,
            [
                _doc_row("D1", "First sentence about cats. Second sentence about cats."),
                _doc_row("D1", "DUPLICATE doc, must lose"),  # dedupe in corpus
                _doc_row("D2", "Dogs are loyal animals. They bark a lot."),
                _doc_row("D3", ""),  # no body -> dropped
            ],
        )
        return {"written": 4}

    async def fake_synth(*, in_path, out_path, **kw):
        rows = []
        for doc in _read(in_path):
            rows.append(
                {
                    "qid": f"syn_{doc['doc_id']}",
                    "query": f"what about {doc['doc_id']}",
                    "topic": None,
                    "persona": "synth",
                    "intent": "definitional",
                    "source_query_id": doc["query_id"],
                    "source_doc_id": doc["doc_id"],
                }
            )
        _write(out_path, rows)
        return {"queries": len(rows)}

    async def fake_label(*, in_path, out_path, **kw):
        rows = _read(in_path)
        for r in rows:
            r["relevant_unit_ids"] = [0]
        _write(out_path, rows)
        return {"labeled": len(rows)}

    monkeypatch.setattr(rp.scrape, "scrape_queries", fake_scrape)
    monkeypatch.setattr(rp.synth_queries_per_doc, "synthesize_queries_per_doc", fake_synth)
    monkeypatch.setattr(rp.label_units, "label_pairs", fake_label)


def test_pipeline_runs_all_stages_and_writes_manifest(tmp_path, monkeypatch):
    _patch_network_stages(monkeypatch)
    cfg = _cfg(tmp_path)
    manifest = asyncio.run(rp.run_pipeline(cfg))

    assert set(manifest) == set(rp.STAGE_ORDER)

    # corpus deduped D1 and dropped body-less D3 -> D1, D2
    corpus = _read(cfg.run_dir / "02_corpus.jsonl")
    assert {r["doc_id"] for r in corpus} == {"D1", "D2"}

    # aggregate must not carry bodies
    agg = _read(cfg.run_dir / "07_aggregate.jsonl")
    assert agg and all("body" not in r for r in agg)
    assert all(r.get("relevant_unit_ids") == [0] for r in agg)

    # manifest records row counts per stage
    assert manifest["corpus"]["rows"] == 2
    assert (cfg.run_dir / "manifest.json").exists()


def test_pipeline_plan_mode_runs_nothing(tmp_path, monkeypatch):
    _patch_network_stages(monkeypatch)
    cfg = _cfg(tmp_path)
    out = asyncio.run(rp.run_pipeline(cfg, plan=True))
    assert out == {}
    assert not (cfg.run_dir / "01_scrape.jsonl").exists()


def test_pipeline_fails_loud_on_empty_stage(tmp_path, monkeypatch):
    _patch_network_stages(monkeypatch)

    async def empty_scrape(*, out_path, **kw):
        out_path.write_text("")  # zero rows
        return {"written": 0}

    monkeypatch.setattr(rp.scrape, "scrape_queries", empty_scrape)
    cfg = _cfg(tmp_path)
    with pytest.raises(SystemExit, match="0 rows"):
        asyncio.run(rp.run_pipeline(cfg))


def test_pipeline_missing_upstream_input_fails(tmp_path, monkeypatch):
    _patch_network_stages(monkeypatch)
    cfg = _cfg(tmp_path)
    # start at 'segment' without producing the corpus it needs
    with pytest.raises(SystemExit, match="missing input"):
        asyncio.run(rp.run_pipeline(cfg, from_stage="segment"))
