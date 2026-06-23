# snippets-runtime

The torch inference runtime for the snippet compressor: the model and the `compress` entrypoints. It sits
between `snippets-training` and `snippets-serving` so neither imports the other — both depend on this
package for the model definition and the keep/drop decision rule. Segmentation and the unit/body span
contract come from [`snippets-common`](../common/README.md), so what the runtime pools scores onto is the
same unit layout the labels were built against.

## Modules

| Module | What it does |
| --- | --- |
| `model.py` | `SentenceCompressor` — encoder + per-sentence keep/drop head. |
| `inference.py` | `compress`, `compress_long`, `compress_long_batch`, `CompressResult`, `load_for_inference`. |

### `model.py`

`SentenceCompressor` is an `AutoModel` encoder (default `answerdotai/ModernBERT-base`) with a linear head
projecting each token's hidden state to one logit. Token logits are mean-pooled within each sentence span
(via `sentence_ids`) and trained with `binary_cross_entropy_with_logits` against the per-sentence relevance
label — this matches the eval metric and avoids the label noise of marking every stopword/punctuation token
positive. Supports class-imbalance `pos_weight` and a sigmoid-prior bias init on the head (`head_bias_init`).

### `inference.py`

`compress(model, tok, query, document, ...)` tokenizes the `(query, body)` pair, runs the encoder, sigmoids
the token logits, and pools them per unit using the `snippets-common` spans. A unit is kept when the
fraction of its tokens above `token_threshold` (default 0.5) clears `sentence_threshold` (default 0.5) —
the dual-threshold rule. The encoder forward runs under bf16 autocast on CUDA (`INFERENCE_DTYPE`), the same
precision the model was trained and validated under, so the thresholds stay calibrated; set it to `None` to
force fp32.

- `compress_long` — packs units into encoder-sized windows (breaking only at unit boundaries), compresses
  each with the same query, and merges the per-window results into one document-level result.
- `compress_long_batch` — same per-document behavior, but flattens every window of every `DocRequest` into
  one length-bucketed list and runs padded GPU minibatches, replacing `sum(windows)` serial forward passes
  with a handful of batched ones.
- `CompressResult` — `kept_indices`, `kept_sentences`, `snippet`, `sentence_scores`, `compression_ratio`,
  and `sentence_mean_probs` (a finer-grained ranking key than the saturating `sentence_scores`).
- `load_for_inference(ckpt_dir)` — build the model with SDPA attention, load `model.pt` weights, return
  `(model, tokenizer)`.

## Usage

```python
from snippets_runtime.inference import compress, load_for_inference

model, tok = load_for_inference("checkpoints/run9")   # -> (model, tokenizer)
result = compress(model, tok, query, document)         # document: str or list[str] units
print(result.snippet, result.compression_ratio)
```

A raw-string `document` is segmented with the canonical [`snippets-common`](../common/README.md)
segmenter, so it pools onto the same unit layout the labels were built against. Pass a
pre-segmented `list[str]` to skip segmentation entirely.

## Setup

```bash
uv sync --package snippets-runtime
```
