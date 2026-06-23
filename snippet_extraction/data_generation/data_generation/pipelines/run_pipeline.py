"""Orchestrate the web-corpus pipeline end-to-end from one YAML config.

The seven stages (scrape -> corpus -> segment -> synth -> join -> label ->
aggregate) already exist as independent, resume-safe CLI tools. This driver
wires them together: it derives every intermediate filename from a single
``run_dir`` (no hand-threading paths), runs the stages in order, fails loud if
a stage produces no rows, and records a ``manifest.json`` of what ran.

Each stage is still individually runnable via its own module; this only
chains their importable core functions.

Usage:
    python -m data_generation.pipelines.run_pipeline --config run.yaml
    python -m data_generation.pipelines.run_pipeline --config run.yaml --plan
    python -m data_generation.pipelines.run_pipeline --config run.yaml \
        --from segment --to label        # rerun a slice
    python -m data_generation.pipelines.run_pipeline --config run.yaml --force

Config (YAML)::

    queries: data/queries/synth_v1_part3_unused.jsonl   # seed queries (stage 1 input)
    run_dir: data/labels/run_fc3                         # all artifacts land here

    # stage knobs (all optional; defaults match the per-stage CLIs)
    retriever: tavily          # scrape
    scrape_limit: 200
    num_results: 5
    extract_depth: advanced
    rpm: null

    synth_n: 2                 # synth_queries_per_doc
    synth_concurrency: 8
    synth_limit: 0

    provider: bedrock          # label_units
    prompt_version: v2
    label_concurrency: 10
    label_limit: 0

    # fold dormant branches (MS MARCO / QASPER label files) into the final merge
    extra_inputs:
      - data/labels/msmarco_labels.jsonl
      - data/labels/qasper_labels.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..core.paths import COMPONENT_ROOT
from . import (
    join_queries_units,
    label_units,
    safely_merge_jsons,
    scrape,
    segment_documents,
    synth_queries_per_doc,
)

# Ordered stage keys; also the canonical pipeline order.
STAGE_ORDER = ["scrape", "corpus", "segment", "synth", "join", "label", "aggregate"]

# Numbered output filename for each stage inside the run dir.
STAGE_OUT = {
    "scrape": "01_scrape.jsonl",
    "corpus": "02_corpus.jsonl",
    "segment": "03_segmented.jsonl",
    "synth": "04_synth_queries.jsonl",
    "join": "05_pairs.jsonl",
    "label": "06_labeled.jsonl",
    "aggregate": "07_aggregate.jsonl",
}


@dataclass
class Config:
    queries: Path
    run_dir: Path
    # scrape
    retriever: str = "tavily"
    scrape_limit: int = 200
    num_results: int = 5
    extract_depth: str = "advanced"
    rpm: int | None = None
    # synth
    synth_n: int = 2
    synth_concurrency: int = 8
    synth_limit: int = 0
    model_id: str | None = None
    # label
    provider: str = "bedrock"
    prompt_version: str = "v2"
    label_concurrency: int = 10
    label_limit: int = 0
    # aggregate
    extra_inputs: list[Path] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"unknown config keys: {sorted(unknown)}")
        if "queries" not in raw or "run_dir" not in raw:
            raise SystemExit("config must set 'queries' and 'run_dir'")

        # Resolve paths relative to the component root (where data/ lives).
        def _p(v: str) -> Path:
            p = Path(v)
            return p if p.is_absolute() else COMPONENT_ROOT / p

        raw["queries"] = _p(raw["queries"])
        raw["run_dir"] = _p(raw["run_dir"])
        raw["extra_inputs"] = [_p(x) for x in raw.get("extra_inputs", [])]
        return cls(**raw)


@dataclass
class Stage:
    key: str
    out: Path
    inputs: list[Path]
    run: Callable[[], Awaitable[dict] | dict]


def _count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


def build_stages(cfg: Config) -> list[Stage]:
    d = cfg.run_dir
    out = {k: d / v for k, v in STAGE_OUT.items()}

    return [
        Stage(
            key="scrape",
            out=out["scrape"],
            inputs=[cfg.queries],
            run=lambda: scrape.scrape_queries(
                retriever=cfg.retriever,
                queries_path=cfg.queries,
                out_path=out["scrape"],
                limit=cfg.scrape_limit,
                rpm=cfg.rpm,
                num_results=cfg.num_results,
                extract_depth=cfg.extract_depth,
            ),
        ),
        Stage(
            key="corpus",
            out=out["corpus"],
            inputs=[out["scrape"]],
            run=lambda: safely_merge_jsons.merge(
                [out["scrape"]],
                out["corpus"],
                dedupe_key="doc_id",
                require_field="body",
                keep_fields=safely_merge_jsons.DOC_FIELDS,
            ),
        ),
        Stage(
            key="segment",
            out=out["segment"],
            inputs=[out["corpus"]],
            run=lambda: segment_documents.segment_corpus(
                in_path=out["corpus"], out_path=out["segment"]
            ),
        ),
        Stage(
            key="synth",
            out=out["synth"],
            inputs=[out["segment"]],
            run=lambda: synth_queries_per_doc.synthesize_queries_per_doc(
                in_path=out["segment"],
                out_path=out["synth"],
                n=cfg.synth_n,
                concurrency=cfg.synth_concurrency,
                limit=cfg.synth_limit,
                model_id=cfg.model_id,
            ),
        ),
        Stage(
            key="join",
            out=out["join"],
            inputs=[out["synth"], out["segment"]],
            run=lambda: join_queries_units.join_queries_to_units(
                queries_path=out["synth"],
                corpus_path=out["segment"],
                out_path=out["join"],
            ),
        ),
        Stage(
            key="label",
            out=out["label"],
            inputs=[out["join"]],
            run=lambda: label_units.label_pairs(
                in_path=out["join"],
                out_path=out["label"],
                provider=cfg.provider,
                prompt_version=cfg.prompt_version,
                concurrency=cfg.label_concurrency,
                limit=cfg.label_limit,
            ),
        ),
        Stage(
            key="aggregate",
            out=out["aggregate"],
            inputs=[out["label"], *cfg.extra_inputs],
            run=lambda: safely_merge_jsons.merge(
                [out["label"], *cfg.extra_inputs],
                out["aggregate"],
                drop_fields=["body"],
            ),
        ),
    ]


def _selected(stages: list[Stage], from_stage: str | None, to_stage: str | None) -> list[Stage]:
    keys = [s.key for s in stages]
    lo = keys.index(from_stage) if from_stage else 0
    hi = keys.index(to_stage) if to_stage else len(keys) - 1
    if lo > hi:
        raise SystemExit(f"--from {from_stage} is after --to {to_stage}")
    return stages[lo : hi + 1]


async def run_pipeline(
    cfg: Config,
    *,
    from_stage: str | None = None,
    to_stage: str | None = None,
    force: bool = False,
    plan: bool = False,
) -> dict:
    stages = _selected(build_stages(cfg), from_stage, to_stage)

    print(f"run_dir : {cfg.run_dir}")
    print(f"stages  : {' -> '.join(s.key for s in stages)}")
    if plan:
        for s in stages:
            exists = "exists" if s.out.exists() else "missing"
            ins = ", ".join(p.name for p in s.inputs)
            print(f"  [{s.key:<9}] in=[{ins}]  out={s.out.name} ({exists})")
        return {}

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for s in stages:
        for ip in s.inputs:
            if not ip.exists():
                raise SystemExit(f"[{s.key}] missing input {ip} — run an earlier stage first")
        if force and s.out.exists():
            s.out.unlink()

        print(f"\n=== stage: {s.key} -> {s.out.name} ===")
        started = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        result = s.run()
        counts = await result if asyncio.iscoroutine(result) else result

        n_rows = _count_rows(s.out)
        if n_rows == 0:
            raise SystemExit(f"[{s.key}] produced 0 rows in {s.out} — stopping (fail loud)")

        manifest[s.key] = {
            "inputs": [str(p) for p in s.inputs],
            "output": str(s.out),
            "rows": n_rows,
            "counts": counts,
            "started": started,
            "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"--- {s.key}: {n_rows} rows -> {s.out.name}")

    print(f"\npipeline complete. manifest -> {manifest_path}")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument(
        "--from",
        dest="from_stage",
        choices=STAGE_ORDER,
        default=None,
        help="first stage to run (default: scrape)",
    )
    ap.add_argument(
        "--to",
        dest="to_stage",
        choices=STAGE_ORDER,
        default=None,
        help="last stage to run (default: aggregate)",
    )
    ap.add_argument(
        "--force", action="store_true", help="delete each stage's output before running it"
    )
    ap.add_argument(
        "--plan", action="store_true", help="print the resolved stage DAG and exit (no work)"
    )
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    asyncio.run(
        run_pipeline(
            cfg,
            from_stage=args.from_stage,
            to_stage=args.to_stage,
            force=args.force,
            plan=args.plan,
        )
    )


if __name__ == "__main__":
    main()
