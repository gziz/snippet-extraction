# Snippet Extraction: Compressing Documents Before They Hit the LLM

A small, query-aware **extractive** context compressor for RAG. Given a `(query, document)`
pair it keeps only the units (sentences / tables / code blocks) that actually help answer the
query and drops the rest — typically a **10–20× token reduction** with no meaningful loss in
downstream answer quality. This repo holds everything needed to reproduce it end to end: data
generation, training, an inference runtime, and serverless serving.

It's an open reproduction of Perplexity's query-aware compression recipe (see
[training/README.md](snippet_extraction/training/README.md) for references).

## How it works

The model is a bidirectional encoder (default `answerdotai/ModernBERT-base`) with a linear
head that scores every token:

1. Segment the document into **units** — prose is sentence-split; tables, code fences, and
   headings stay atomic. This layout is the single source of truth shared by training and
   inference (`snippets-common`).
2. Feed `(query, body)` to the encoder and project each token's hidden state to one logit.
3. Sigmoid the logits, then **mean-pool** them per unit using the unit's char span.
4. Keep a unit when the fraction of its tokens above `token_threshold` clears
   `sentence_threshold` (the dual-threshold rule).
5. Concatenate surviving units in document order — that's the snippet.

Long documents are packed into encoder-sized windows (breaking only at unit boundaries) and
the per-window results are merged, so a 300-unit page works the same as a 5-unit one.

## Layout

```
snippet_extraction/
  common/           torch-free shared core: segmentation, windowing, char spans
  data_generation/  LLM-labeled training-data pipelines: retrieval, scraping, labeling, eval
  runtime/          torch inference runtime: model + compress() entrypoints
  training/         model training, evaluation, and prediction
  serving/          Modal-based serverless serving + benchmarks
```

Each package is independent, with its own `pyproject.toml` and `README.md`:

| Package | What it is | README |
| --- | --- | --- |
| `snippets-common` | The unit/body/span contract both sides import instead of reimplementing. No torch. | [common](snippet_extraction/common/README.md) |
| `data_generation` | Builds `(query, document, kept_units)` labels by prompting an LLM as judge. | [data_generation](snippet_extraction/data_generation/README.md) |
| `snippets-runtime` | `SentenceCompressor` model + `compress` / `compress_long`. Shared by training and serving. | [runtime](snippet_extraction/runtime/README.md) |
| `snippets-training` | Preprocessing, BCE training, and evaluation. | [training](snippet_extraction/training/README.md) |
| `snippets-serving` | Deploys a checkpoint to Modal (T4, scale-to-zero). | [serving](snippet_extraction/serving/README.md) |

## End-to-end flow

```
data_generation ──► labeled JSONL ──► training ──► checkpoint ──► serving (Modal endpoint)
   (LLM judge)      (all_labels.jsonl)   (BCE)      (run9/)        │
                                                                  runtime.compress()
                                                          (shared model + decision rule)
```

`snippets-common` sits under all of it: data generation labels the same units the runtime
later pools scores onto, so the model never sees a different unit layout at inference than it
did at train time.

## Getting started

Each package syncs on its own with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --package snippets-common      # shared core, no torch
uv sync --package snippets-runtime     # + model / inference
uv sync --inexact --package snippets-serving   # modal client
```

Then follow the package READMEs:

- **Generate data** → [data_generation](snippet_extraction/data_generation/README.md)
  (`run_pipeline` chains scrape → segment → synth queries → label → aggregate).
- **Train** → [training](snippet_extraction/training/README.md).
- **Run inference locally**:

  ```python
  from snippets_runtime.inference import compress, load_for_inference

  model, tok = load_for_inference("checkpoints/run9")
  result = compress(model, tok, query, document)   # document: str or list[str] units
  print(result.snippet, result.compression_ratio)
  ```

- **Deploy** → [serving](snippet_extraction/serving/README.md) (a warm request round-trips in
  ~0.5 s; an idle deployment costs $0).

## Not included

Evals code.
