# snippets-common

The torch-free shared core. Training and serving must segment a document, assemble its body, and window
it against **identical** logic — otherwise the model sees a different unit layout at inference than it did
at train time, and every char span and pooled score drifts. This package is the single source of truth
for that contract; both sides import it instead of reimplementing it.

No torch dependency (just `markdown-it-py`, `syntok`, `tiktoken`), so the segmentation and
selection/rendering logic can be unit-tested and reused by eval harnesses without the model stack.

## Modules

| Module | What it does |
| --- | --- |
| `segment.py` | Block-aware markdown segmentation: document body → `list[str]` units. |
| `spans.py` | `build_body_and_spans`: units → one body string + half-open char spans. |
| `windowing.py` | `pack_windows`, `select_under_budget`, `render_snippet`. |

### `segment.py`

`segment(body)` parses structure **first** with a CommonMark/GFM parser (`markdown-it-py`), then sentence-
splits **only** prose paragraphs with `syntok`. Structured blocks — tables, code fences, headings, list
items — are kept atomic and sliced verbatim from the source, so a prose segmenter never mangles a markdown
table at its inline-link boundaries. Prose units are left alone at or under `MAX_UNIT_TOKENS` (250) and
otherwise word-windowed; atomic blocks stay whole up to `MAX_BLOCK_TOKENS` (500) before falling back to
line-boundary splitting. Token counts are real `cl100k_base` BPE, not a whitespace proxy.

### `spans.py`

`build_body_and_spans(units)` concatenates units with single spaces and returns `(body, spans)` where each
span is the half-open `[start, end)` char range of a unit in `body`. This is the shared unit/body contract:
the inference runtime pools per-token scores back onto units using exactly these spans.

### `windowing.py`

Pure helpers for long-document inference, all operating on unit lists:

- `pack_windows(unit_token_lens, capacity)` — pack consecutive units into windows that fit the encoder
  context, breaking only at unit boundaries.
- `select_under_budget(units, scores, budget_tokens, count_tokens, *, min_score=0.1)` — rank by score, fill
  greedily under a token budget, re-sort to document order.
- `render_snippet(units, kept_indices, *, gap="[...]")` — join kept units in document order, marking
  non-contiguous gaps so the consumer can tell the snippet is non-contiguous.

## Usage

```python
from snippets_common.segment import segment
from snippets_common.spans import build_body_and_spans

units = segment(document_markdown)        # list[str], the canonical unit definition
body, spans = build_body_and_spans(units) # body string + half-open char spans per unit
```

## Setup

```bash
uv sync --package snippets-common
```
