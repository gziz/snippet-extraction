# snippet-extraction

Query-aware snippet extraction: training, data generation, runtime, and serving
code for producing short, query-relevant snippets from documents.

> This repository is a curated, code-only mirror. It is published for reading and
> reference; development happens in a private repository and is exported here.

## Layout

```
snippet_extraction/
  common/           shared library (segmentation, windowing, spans)
  data_generation/  data pipelines: retrieval, scraping, labeling, eval
  runtime/          inference runtime
  serving/          Modal-based serving + benchmarks
  training/         model training, evaluation, and prediction
```

Each package is independent and has its own `pyproject.toml` and `README.md`.

## Not included

To keep this a focused, code-only repo, generated datasets, model checkpoints,
private notes, and blog/plot image artifacts are intentionally omitted.
