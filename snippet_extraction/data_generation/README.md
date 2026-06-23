# data_generation — training-data generation

Generates labeled data for an **extractive context-compression** task: given a
(query, document) pair, which document *units* (sentences / table / code
blocks) are needed to answer the query. The labels train the snippet model in
`query-aware-snippets-training/`.

## Layout

```
data_generation/
  core/                 # library code (importable, no I/O at import time)
    segment.py          #   body -> units (block-aware: tables/code atomic, prose sentence-split)
    schema.py           #   canonical row shapes + JSONL store + resume/dedupe keys
    paths.py            #   all filesystem locations + load_env()
    ratelimit.py        #   shared async RPM limiter
    labeling/
      prompts.py        #   versioned labeler prompts (v2 strict+anchor, v3 +context)
      clients.py        #   provider transports (azure/bedrock/gemini/openrouter), shared retry
    retrievers/         #   tavily, firecrawl, exa(+align), brave HTTP clients
    msmarco/            #   MS MARCO doc/BM25 index build + random-access fetch
    snippet_eval/       #   provider snippet-quality eval harness (4 cached stages)
  pipelines/            # CLI entry points (thin wrappers over core)
  data/                 # generated artifacts, gitignored — see data/DATA.md
  reports/              # experiment write-ups (tracked)
  html_visualizers/     # label/AB browsers (serve with: python3 -m http.server)
  tests/                # pytest suite for the pure logic
```

## The web-corpus pipeline (current path)

Each stage is append-only and resume-safe; re-running skips finished work.
Run from the repo root with `uv run`.

```
                       (1) scrape                  (2) safely_merge_jsons (corpus)
queries/*.jsonl ──► scrape rows (tavily_v1.jsonl) ──► fc2_01_corpus.jsonl
                                                            │ (3) segment_documents
                                                            ▼
            (4) synth_queries_per_doc  ◄──────────  fc2_02_segmented.jsonl
                        │ corpus_synth_q3.jsonl             │
                        └──────────► (5) join_queries_units ┘
                                            │ fc2_03_pairs.jsonl
                                            ▼
                                  (6) label_units (LLM labeler)
                                            │ fc2_04_labeled_sonnet46.jsonl
                                            ▼
                                  (7) safely_merge_jsons (aggregate)
                                            │ all_labels_v2.jsonl  ──► training
```

### Run the whole thing (`run_pipeline`)

`run_pipeline` chains all seven stages from one YAML config. It derives every
intermediate filename from a single `run_dir` (numbered `01_scrape.jsonl` …
`07_aggregate.jsonl`), runs stages in order, **fails loud** if a stage produces
no rows, and writes a `manifest.json` recording each stage's inputs / output /
row counts / params. Each stage is still independently runnable (below); the
orchestrator only calls their importable core functions.

```bash
# full run from an example config (see pipelines/configs/web_corpus.example.yaml)
uv run python -m data_generation.pipelines.run_pipeline \
    --config data_generation/pipelines/configs/web_corpus.example.yaml

# preview the resolved stage DAG without running anything
uv run python -m data_generation.pipelines.run_pipeline --config <cfg> --plan

# rerun only a slice (cross-stage resume; upstream files must already exist)
uv run python -m data_generation.pipelines.run_pipeline --config <cfg> \
    --from segment --to label

# redo a run from scratch (deletes each stage's output before running)
uv run python -m data_generation.pipelines.run_pipeline --config <cfg> --force
```

The config sets `queries` + `run_dir` and per-stage knobs (`retriever`,
`provider`, `synth_n`, …); `extra_inputs` folds the dormant MS MARCO / QASPER
label files into the final aggregate merge. Paths are resolved relative to the
component root.

### Stages individually

```bash
# 1. search + scrape full bodies for a query set (tavily is the current default)
uv run python -m data_generation.pipelines.scrape --retriever tavily \
    --queries data_generation/data/queries/synth_v1_part3_unused.jsonl \
    --out     data_generation/data/labels/tavily_v1.jsonl --limit 200

# 2. one row per unique scraped document (base files first; first-seen doc_id wins)
uv run python -m data_generation.pipelines.safely_merge_jsons \
    --inputs data_generation/data/labels/firecrawl_v1.jsonl \
             data_generation/data/labels/tavily_v1.jsonl \
    --out    data_generation/data/labels/fc2_01_corpus.jsonl \
    --dedupe-key doc_id --require body \
    --keep doc_id origin query query_id topic intent url title \
           description body position

# 3. segment bodies with the current segmenter
uv run python -m data_generation.pipelines.segment_documents \
    --in  data_generation/data/labels/fc2_01_corpus.jsonl \
    --out data_generation/data/labels/fc2_02_segmented.jsonl

# 4. generate extra diverse-intent queries per document (reads the segmented
#    corpus: synth needs status="ok" rows, which segment_documents sets)
uv run python -m data_generation.pipelines.synth_queries_per_doc \
    --in data_generation/data/labels/fc2_02_segmented.jsonl \
    --out data_generation/data/queries/corpus_synth_q3.jsonl --n 2

# 5. pair generated queries with their doc's units
uv run python -m data_generation.pipelines.join_queries_units \
    --queries data_generation/data/queries/corpus_synth_q3.jsonl \
    --corpus  data_generation/data/labels/fc2_02_segmented.jsonl \
    --out     data_generation/data/labels/fc2_03_pairs.jsonl

# 6. label every (query, units) pair (Sonnet 4.6 on Bedrock by default)
uv run python -m data_generation.pipelines.label_units \
    --in  data_generation/data/labels/fc2_03_pairs.jsonl \
    --out data_generation/data/labels/fc2_04_labeled_sonnet46.jsonl \
    --provider bedrock --concurrency 10

# 7. merge different datasets into the training aggregate (drop heavy bodies)
uv run python -m data_generation.pipelines.safely_merge_jsons \
    --inputs data_generation/data/labels/msmarco_labels.jsonl \
             data_generation/data/labels/fc2_04_labeled_sonnet46.jsonl \
    --out    data_generation/data/labels/all_labels_v2.jsonl --drop body
```

## Labeling

- **Prompts** are versioned in `core/labeling/prompts.py`. v2 (strict answer +
  anchor) is production; v3 (strict + context superset) was A/B'd. Adding a
  prompt = adding a `LabelPrompt` entry — transports don't change.
- **Providers**: `--provider azure | bedrock | gemini | openrouter`
  (gpt-5.4 / Claude Sonnet 4.6 / Gemini / DeepSeek). API keys in
  `data_generation/.env`.
- **`prompt_hash`** fingerprints the prompt text + builder bytecode and is
  stored in every row. The v2/v3 hashes are pinned in `tests/test_prompts.py`;
  if you edit a prompt, bump the version instead of mutating it.

## Conventions

- All label/scrape files are **append-only JSONL**; the two row shapes are
  defined only in `core/schema.py`. Resume keys: scrape = `query_id`,
  label = `(query_id, doc_id, labeler_provider, labeler_model, prompt_version)`.
- `query_id` is compared **as a string** everywhere (sources disagree on type).
- Documents with > 300 units are recorded as `skipped:long(n)`, not labeled.

## Other tools

| command (`data_generation.pipelines.*`) | purpose |
|---|---|
| `run_pipeline` | chain all 7 web-corpus stages from one YAML config (run-dir + manifest) |
| `synth_queries` | seed queries over a (topic × persona × intent) grid |
| `label_msmarco` / `regression` | MS MARCO qrel labeling + curated regression diff |
| `ab_labeler` / `merge_ab` | prompt A/B on identical pairs + visualizer merge |
| `compare_labelers` | labeler-vs-labeler agreement / cost report (e.g. Haiku vs Sonnet) |
| `eval_snippets` | provider snippet-quality eval (exa/tavily/brave vs labeled gold) |
| `eval_tavily` | quick qualitative dump of Tavily results |
| `inspect_labels` | pretty-print labeled rows in the terminal |
| `label_exa` | LEGACY: exa-highlight alignment labels (pre-LLM-labeler) |

## Tests

```bash
uv run python -m pytest data_generation/tests -q
```

Covers the segmenter, prompt registry (pinned hashes), row schema + resume
keys, exa alignment, and the snippet-eval metrics.
