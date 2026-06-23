"""Generate additional diverse-intent queries per Firecrawl-pulled document.

Takes a Firecrawl labels JSONL (rows of {query_id, doc_id, query, intent, body, ...})
and for each (query_id, doc_id) row asks Sonnet 4.6 to produce N additional
queries the doc plausibly answers, each in a *different* intent class than the
existing one.

Writes one row per generated query to ``--out``, matching the schema of
synth_queries.py: {qid, query, topic, persona, intent, source_doc_id,
source_query_id}.

The new qids can then be paired with the same doc_id at label time:
label_firecrawl-derived rows downstream join on (query_id, doc_id).

Usage:
    python -m data_generation.pipelines.synth_queries_per_doc \\
        --in   data_generation/data/labels/firecrawl_v1_part2.jsonl \\
        --out  data_generation/data/queries/synth_v1_per_doc.jsonl \\
        --n    2 \\
        --concurrency 8
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

INTENT_CLASSES = [
    ("definitional", "what is X / how does X work conceptually"),
    ("procedural", "how do I do X / steps to accomplish X"),
    ("comparative", "X vs Y, when to use which"),
    ("debugging", "my X is doing Y and I want it to do Z, or why does X happen"),
    ("best-practice", "what is the recommended way to do X / pitfalls of X"),
    ("enumeration", "checklist / gotchas / things to consider when X"),
]
INTENT_LABELS = [c[0] for c in INTENT_CLASSES]


SYSTEM = """You are generating additional realistic search queries that engineers and knowledge workers would type into a search engine or ask an AI assistant. You will be given (a) an existing query, (b) the intent class of that query, and (c) the document body that the existing query plausibly relates to.

Your job: produce N additional queries the SAME document plausibly answers, each in a DIFFERENT intent class from the existing query and from each other.

Hard rules:
- Each query must read like something a real person would type into Google or Copilot Chat. Not academic, not formal.
- 5 to 15 words. Single sentence or fragment. Match the terse-natural style of search queries.
- No quotation marks, no markdown, no numbering, no leading verbs like "How to" on every query - vary the phrasing.
- Do not invent fake product names, fake company names, or fake error messages. If you reference a tool, use a real one from the document.
- The query must be answerable by the provided document - it must cover a real aspect the doc discusses.
- Each of the N queries must hit a DIFFERENT intent class than the existing query and than the other generated queries. Use the intent labels provided.
- Where possible, cover a different aspect / section of the document than the existing query covers.
- Output exactly N lines, each in the format:
    INTENT_LABEL ||| query text
  Where INTENT_LABEL is one of the provided intent class labels.
- Output ONLY those N lines, no commentary, no preamble."""


def build_user(
    existing_query: str, existing_intent: str, allowed_intents: list[str], body: str, n: int
) -> str:
    body_excerpt = body[:6000]  # cap input tokens; first ~1500 words is plenty for query gen
    allowed = ", ".join(allowed_intents)
    return (
        f"Existing query: {existing_query}\n"
        f"Existing query's intent class: {existing_intent}\n"
        f"Allowed intent classes for new queries (pick {n} distinct ones, none equal to the existing): {allowed}\n\n"
        f"Document body (may be truncated):\n---\n{body_excerpt}\n---\n\n"
        f"Generate exactly {n} additional queries. Output {n} lines, each in the form 'INTENT_LABEL ||| query text'."
    )


def make_qid(source_qid: str, doc_id: str, query: str, intent: str) -> str:
    h = abs(hash((source_qid, doc_id, intent, query))) % (10**12)
    return f"syn_pd_{h:012d}"


async def gen_for_doc(
    client,
    model_id: str,
    existing_query: str,
    existing_intent: str,
    body: str,
    n: int,
) -> list[tuple[str, str]]:
    """Return list of (intent, query) pairs."""
    # Build the allowed intent list (everything except existing_intent)
    allowed = [c for c in INTENT_LABELS if c != existing_intent]
    msg = await client.messages.create(
        model=model_id,
        max_tokens=512,
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": build_user(existing_query, existing_intent, allowed, body, n),
            }
        ],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    out: list[tuple[str, str]] = []
    seen_intents: set[str] = set()
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln or "|||" not in ln:
            continue
        ln = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", ln)
        intent_part, _, query_part = ln.partition("|||")
        intent = intent_part.strip().lower()
        query = query_part.strip().strip('"').strip("'")
        if intent not in INTENT_LABELS:
            continue
        if intent == existing_intent or intent in seen_intents:
            continue
        wc = len(query.split())
        if not (3 <= wc <= 20):
            continue
        seen_intents.add(intent)
        out.append((intent, query))
        if len(out) >= n:
            break
    return out


async def synthesize_queries_per_doc(
    *,
    in_path: Path,
    out_path: Path,
    n: int = 2,
    concurrency: int = 8,
    limit: int = 0,
    model_id: str | None = None,
) -> dict:
    """Generate ``n`` diverse-intent queries per document into ``out_path``.

    Resume-safe on (source_query_id, source_doc_id). Returns a counts dict.
    """
    from anthropic import AsyncAnthropicBedrock

    load_env()
    if model_id is None:
        model_id = os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")

    in_rows = []
    seen_pairs: set[tuple[str, str]] = set()
    for line in in_path.open():
        r = json.loads(line)
        if r.get("status") != "ok":
            continue
        if not r.get("body") or not r.get("query"):
            continue
        key = (r["query_id"], r["doc_id"])
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        in_rows.append(r)

    print(f"[synth_pd] input rows ok: {len(in_rows)}")

    # Resume: skip source rows whose source_query_id already appears in out
    done_source_qids: set[str] = set()
    if out_path.exists():
        for ln in out_path.open():
            try:
                r = json.loads(ln)
                if r.get("source_query_id") and r.get("source_doc_id"):
                    done_source_qids.add((r["source_query_id"], r["source_doc_id"]))
            except Exception:
                pass
    pending = [r for r in in_rows if (r["query_id"], r["doc_id"]) not in done_source_qids]
    print(f"[synth_pd] already covered: {len(done_source_qids)}; pending: {len(pending)}")
    if limit:
        pending = pending[:limit]
        print(f"[synth_pd] limit applied: {len(pending)}")
    if not pending:
        print("[synth_pd] nothing to do")
        return {"done": 0, "queries": 0, "errors": 0}

    client = AsyncAnthropicBedrock(aws_region=os.environ.get("AWS_REGION", "us-east-1"))
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters = {"done": 0, "queries": 0, "errors": 0}
    t0 = time.perf_counter()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")

    async def worker(row: dict) -> None:
        async with sem:
            existing_intent = (row.get("intent") or "procedural").lower()
            # Normalize: existing intents in synth_v1 use values like "procedural", "comparative", etc.
            if existing_intent not in INTENT_LABELS:
                existing_intent = "procedural"
            try:
                pairs = await gen_for_doc(
                    client,
                    model_id,
                    row["query"],
                    existing_intent,
                    row["body"],
                    n,
                )
            except Exception as e:
                async with write_lock:
                    counters["errors"] += 1
                    counters["done"] += 1
                    if counters["done"] % 20 == 0 or counters["errors"] <= 5:
                        print(f"[synth_pd] err {row['query_id'][:16]}: {str(e)[:120]}")
                return

            async with write_lock:
                for intent, query in pairs:
                    qid = make_qid(row["query_id"], row["doc_id"], query, intent)
                    out_row = {
                        "qid": qid,
                        "query": query,
                        "topic": row.get("topic"),
                        "persona": "synthesized from doc",
                        "intent": intent,
                        "source_query_id": row["query_id"],
                        "source_doc_id": row["doc_id"],
                    }
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    counters["queries"] += 1
                counters["done"] += 1
                fout.flush()
                if counters["done"] % 25 == 0:
                    dt = time.perf_counter() - t0
                    rate = counters["done"] / dt * 60
                    eta = (len(pending) - counters["done"]) / max(1, counters["done"]) * dt / 60
                    print(
                        f"[synth_pd] {counters['done']}/{len(pending)}  "
                        f"queries={counters['queries']} errors={counters['errors']}  "
                        f"rate={rate:.1f}/min  eta={eta:.1f}min"
                    )

    tasks = [worker(r) for r in pending]
    await asyncio.gather(*tasks)
    fout.close()

    dt = time.perf_counter() - t0
    print(
        f"\n[synth_pd] DONE. docs_processed={counters['done']} queries_written={counters['queries']} "
        f"errors={counters['errors']} wall={dt:.1f}s"
    )
    return dict(counters)


async def main_async(args) -> None:
    await synthesize_queries_per_doc(
        in_path=args.in_path,
        out_path=args.out,
        n=args.n,
        concurrency=args.concurrency,
        limit=args.limit,
        model_id=args.model_id,
    )


def main() -> None:
    load_env()  # BEDROCK_MODEL_ID may come from .env (used in the argparse default)
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=2, help="queries to generate per input row")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="cap input rows (0 = all)")
    ap.add_argument(
        "--model-id",
        default=os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
    )
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
