"""Synthesize queries for the Exa pass.

Generates natural-language queries across a (topic x persona x intent) grid
using Claude Sonnet on Bedrock. Output is a JSONL of:
    {"qid": str, "query": str, "topic": str, "persona": str, "intent": str}

Usage:
    python -m data_generation.pipelines.synth_queries \
        --out data_generation/data/queries/synth_v1.jsonl \
        --per-cell 1

A "cell" is one (topic, persona, intent) combo. With 18 topics x 5 personas x
5 intents = 450 cells, --per-cell 1 yields ~450 queries; --per-cell 5 yields
~2250. The generator is asked for N queries per call to amortize prompt cost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

from ..core.paths import load_env

TOPICS: list[tuple[str, str]] = [
    (
        "backend_services",
        "backend services: FastAPI, Go services, Rust async, Node servers, microservices, request lifecycles",
    ),
    (
        "databases_perf",
        "databases and query performance: Postgres, MySQL, Redis, ClickHouse, DynamoDB, query planning, indexing",
    ),
    (
        "distributed_systems",
        "distributed systems and infra: kubernetes, Docker, service mesh, queues, consensus, leader election",
    ),
    (
        "cloud_platforms",
        "cloud platforms: AWS, GCP, Azure-specific services, IAM, networking, cost",
    ),
    (
        "frontend",
        "frontend frameworks: Next.js, React internals, Vue, Svelte, build tooling, SSR, hydration",
    ),
    (
        "ml_llm_eng",
        "ML / LLM engineering: training, RAG, inference, eval, vector DBs, fine-tuning, prompt engineering",
    ),
    (
        "devops_cicd",
        "DevOps and CI/CD: Terraform, GitHub Actions, GitLab CI, observability, deployment strategies",
    ),
    (
        "security",
        "security: OAuth/OIDC, JWT, secrets management, network policy, threat modeling, supply chain",
    ),
    (
        "language_internals",
        "language internals: lifetimes, GC, async runtimes, type systems, memory models",
    ),
    (
        "apis_protocols",
        "APIs and protocols: gRPC, GraphQL, WebSockets, REST design, OpenAPI, idempotency",
    ),
    ("ios_swift", "iOS and Swift development: SwiftUI, Combine, concurrency, lifecycle, App Store"),
    (
        "android_kotlin",
        "Android and Kotlin development: Jetpack Compose, coroutines, lifecycle, build system",
    ),
    (
        "data_engineering",
        "data engineering: Spark, Airflow, dbt, lakehouse, Iceberg, Kafka, ELT pipelines",
    ),
    (
        "data_analysis_sql",
        "data analysis and SQL patterns: window functions, CTEs, dashboards, metric layers",
    ),
    (
        "product_pm",
        "product / PM artifacts: RFCs, roadmaps, A/B test design, one-pagers, prioritization frameworks",
    ),
    ("compliance_legal", "compliance and legal basics: GDPR, SOC2, HIPAA, data residency, DPAs"),
    (
        "saas_metrics",
        "SaaS finance and metrics: NRR, GRR, CAC, LTV, unit economics, cohort analysis",
    ),
    (
        "research_workflows",
        "research workflows: literature review, paper summarization, eval design, reproducibility",
    ),
]

PERSONAS = [
    "a senior backend engineer who is debugging a problem in production",
    "an engineer learning a new technology and trying to understand the basics",
    "a tech lead comparing two architectures or libraries and weighing tradeoffs",
    "an engineer doing a task they have done before and just needs the command, snippet, or recipe",
    "a knowledge worker (PM, analyst, researcher) trying to understand a concept or produce an artifact",
]

INTENTS = [
    "definitional - what is X / how does X work conceptually",
    "procedural - how do I do X / steps to accomplish X",
    "comparative - X vs Y, when to use which",
    "debugging - my X is doing Y and I want it to do Z, or 'why does X happen'",
    "best-practice - what is the recommended way to do X / pitfalls of X",
]


SYSTEM_SYNTH = """You are generating realistic search queries that engineers and knowledge workers would actually type into a search engine or ask an AI assistant.

Hard rules:
- Every query must read like something a real person would type. Not academic, not formal, not over-specified.
- Mix lengths: some terse keyword queries (3-6 words), some natural-language questions (10-20 words). Avoid uniform length.
- No quotation marks, no markdown, no numbering, no leading verbs like "How to" on every query - vary the phrasing.
- Do not invent fake product names, fake company names, or fake error messages. If you reference a tool, use a real one.
- Each query must be answerable by an external web page (docs, blog post, SO answer, paper). Avoid queries that depend on private knowledge.
- Do NOT include the topic name, persona, or intent label in the query text.
- Output ONLY the queries, one per line, no bullets, no numbering, no commentary."""


def build_user(topic_label: str, topic_desc: str, persona: str, intent: str, n: int) -> str:
    return (
        f"Topic area: {topic_desc}\n"
        f"Persona: {persona}\n"
        f"Intent: {intent}\n\n"
        f"Generate {n} distinct queries that fit this cell. Vary phrasing and length. "
        f"Output one query per line, nothing else."
    )


async def synth_cell(
    client,
    model_id: str,
    topic_label: str,
    topic_desc: str,
    persona: str,
    intent: str,
    n: int,
) -> list[str]:
    msg = await client.messages.create(
        model=model_id,
        max_tokens=1024,
        system=SYSTEM_SYNTH,
        messages=[
            {"role": "user", "content": build_user(topic_label, topic_desc, persona, intent, n)}
        ],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Strip any accidental leading bullets / numbers.
    cleaned: list[str] = []
    for ln in lines:
        ln = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", ln)
        ln = ln.strip().strip('"').strip("'")
        if 3 <= len(ln.split()) <= 40:
            cleaned.append(ln)
    return cleaned[:n]


async def main_async(
    out_path: Path,
    per_cell: int,
    concurrency: int,
    model_id: str,
    max_cells: int | None = None,
    seed: int = 0,
) -> None:
    import random

    from anthropic import AsyncAnthropicBedrock

    load_env()
    client = AsyncAnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"))

    sem = asyncio.Semaphore(concurrency)
    cells: list[tuple[str, str, str, str]] = []
    for topic_label, topic_desc in TOPICS:
        for persona in PERSONAS:
            for intent in INTENTS:
                cells.append((topic_label, topic_desc, persona, intent))

    if max_cells is not None and max_cells < len(cells):
        rng = random.Random(seed)
        rng.shuffle(cells)
        cells = cells[:max_cells]
        print(f"[synth] sampled {len(cells)} cells (seed={seed})")

    print(f"[synth] {len(cells)} cells x {per_cell} per cell = ~{len(cells) * per_cell} queries")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    if out_path.exists():
        for ln in out_path.open():
            try:
                seen.add(json.loads(ln)["query"].lower())
            except Exception:
                pass
    print(f"[synth] {len(seen)} existing queries in {out_path}")

    written = 0
    t0 = time.perf_counter()

    async def worker(topic_label: str, topic_desc: str, persona: str, intent: str) -> list[dict]:
        async with sem:
            try:
                qs = await synth_cell(
                    client, model_id, topic_label, topic_desc, persona, intent, per_cell
                )
            except Exception as e:
                print(f"[synth] cell err {topic_label}/{intent[:20]}: {e}")
                return []
            rows = []
            for q in qs:
                if q.lower() in seen:
                    continue
                seen.add(q.lower())
                qid = f"syn_{abs(hash((topic_label, persona, intent, q))) % (10**12):012d}"
                rows.append(
                    {
                        "qid": qid,
                        "query": q,
                        "topic": topic_label,
                        "persona": persona.split(" who")[0]
                        .split(" trying")[0]
                        .split(" doing")[0]
                        .split(" comparing")[0]
                        .strip(),
                        "intent": intent.split(" -")[0],
                    }
                )
            return rows

    tasks = [worker(*c) for c in cells]
    with out_path.open("a") as f:
        for fut in asyncio.as_completed(tasks):
            rows = await fut
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                written += 1
            if written % 50 == 0 and written > 0:
                dt = time.perf_counter() - t0
                print(f"[synth] {written} queries, {dt:.1f}s, {written / dt * 60:.1f}/min")

    print(
        f"[synth] done. wrote {written} new queries in {time.perf_counter() - t0:.1f}s -> {out_path}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--per-cell", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="global.anthropic.claude-sonnet-4-6")
    ap.add_argument(
        "--max-cells", type=int, default=None, help="If set, randomly sample this many cells."
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(
        main_async(args.out, args.per_cell, args.concurrency, args.model, args.max_cells, args.seed)
    )


if __name__ == "__main__":
    main()
