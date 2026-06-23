"""Snippet-extraction eval as a 4-stage cached pipeline.

All expensive work is persisted to disk and resumable. Re-scoring with a
different metric or labeler prompt only re-runs the cheap ``score`` stage.

Layout (one directory per run):
    <run_dir>/
      provider_results/<provider>.jsonl   # stage 1: per (provider, query_id)
      bodies.jsonl                        # stage 2: per url (Firecrawl scrape)
      gold.jsonl                          # stage 3: per (query_id, url) Sonnet labeler
      doc_eval.jsonl                      # stage 4: per (provider, query_id, url) scored
      summary.json                        # aggregates over doc_eval

Usage:
    python -m data_generation.pipelines.eval_snippets \\
        --queries data_generation/data/queries/smoke_v1.jsonl \\
        --providers exa,tavily,brave \\
        --n 10 \\
        --labeler bedrock \\
        --run-dir data_generation/data/snippet_runs/v2_10q \\
        --fc-rpm 6

Re-score only (no provider/Firecrawl/labeler calls):
    python -m data_generation.pipelines.eval_snippets \\
        --queries ... --providers exa,tavily,brave --n 10 \\
        --run-dir data_generation/data/snippet_runs/v2_10q \\
        --score-only
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from ..core.labeling import make_async_labeler
from ..core.retrievers.firecrawl import FirecrawlSearch
from ..core.snippet_eval import stages
from ..core.snippet_eval.harness import format_summary, summarize
from ..core.snippet_eval.providers import make_provider


def _provider_paths(run_dir: Path, names: list[str]) -> list[Path]:
    return [run_dir / "provider_results" / f"{name}.jsonl" for name in names]


async def main_async(
    queries_path: Path,
    provider_names: list[str],
    n: int,
    labeler_provider: str,
    run_dir: Path,
    fc_rpm: int,
    score_only: bool,
) -> None:
    queries = [json.loads(l) for l in queries_path.open()][:n]
    run_dir.mkdir(parents=True, exist_ok=True)
    pr_paths = _provider_paths(run_dir, provider_names)
    bodies_path = run_dir / "bodies.jsonl"
    gold_path = run_dir / "gold.jsonl"
    eval_path = run_dir / "doc_eval.jsonl"
    summary_path = run_dir / "summary.json"

    if not score_only:
        # Stage 1: providers.
        for name, out in zip(provider_names, pr_paths):
            print(f"\n[1/4] collect provider={name} -> {out}")
            provider = make_provider(name)
            try:
                await stages.collect_provider_results(queries, provider, out)
            finally:
                await provider.aclose()

        # Stage 2: bodies (one Firecrawl client serialized via its limiter).
        print(f"\n[2/4] fetch bodies -> {bodies_path}")
        fetcher = FirecrawlSearch(rpm=fc_rpm)
        try:
            await stages.fetch_bodies(pr_paths, fetcher, bodies_path)
        finally:
            await fetcher.aclose()

        # Stage 3: labeler.
        print(f"\n[3/4] label bodies (provider={labeler_provider}) -> {gold_path}")
        labeler = make_async_labeler(labeler_provider)
        await stages.label_bodies(pr_paths, bodies_path, labeler, gold_path)

    # Stage 4: score (always; cheap and pure).
    print(f"\n[4/4] score -> {eval_path}")
    rows = stages.score(queries, pr_paths, bodies_path, gold_path, eval_path)

    summary = summarize(rows)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(format_summary(summary))
    print(f"\n[eval_snippets] doc_eval: {eval_path}")
    print(f"[eval_snippets] summary:  {summary_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, type=Path)
    ap.add_argument("--providers", default="exa,tavily,brave")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--labeler", default="bedrock", help="bedrock | azure | gemini")
    ap.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="Directory holding all four stage outputs. Re-runs resume from it.",
    )
    ap.add_argument("--fc-rpm", type=int, default=6, help="Firecrawl scrape RPM")
    ap.add_argument(
        "--score-only",
        action="store_true",
        help="Skip stages 1-3 and just (re)build doc_eval.jsonl + summary from caches.",
    )
    args = ap.parse_args()
    provider_names = [p.strip() for p in args.providers.split(",") if p.strip()]
    asyncio.run(
        main_async(
            args.queries,
            provider_names,
            args.n,
            args.labeler,
            args.run_dir,
            args.fc_rpm,
            args.score_only,
        )
    )


if __name__ == "__main__":
    main()
