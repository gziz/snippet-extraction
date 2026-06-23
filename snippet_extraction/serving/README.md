# Serving

The trained compressor (`run9`, ModernBERT-base + per-sentence head, 596 MB fp32) deployed as a serverless endpoint on Modal. A T4 container boots on the first request, loads the checkpoint from a Volume, and scales back to zero when idle — so an idle deployment costs $0.

Measured on the live deployment:

| | |
| --- | --- |
| Cold start (boot + weight load) | ~16 s |
| Warm request, round trip | **0.5 s** |
| Idle cost | $0 (scale-to-zero, 120 s idle tail) |
| GPU | T4, ~$0.59/hr billed per second |

## Deploy

One-time setup (from the repo root):

```bash
uv sync --inexact --package snippets-serving   # modal client (+ snippets-training)
.venv/bin/modal token new                      # opens browser auth, writes ~/.modal.toml
```

Upload a checkpoint and deploy:

```bash
.venv/bin/modal volume create compressor-checkpoints
.venv/bin/modal volume put compressor-checkpoints \
    snippet_extraction/training/checkpoints/run9 /run9

cd snippet_extraction/serving
../../.venv/bin/modal deploy modal_app.py
```

The deploy prints the endpoint URL (yours: `https://rev-71693--query-aware-compressor-compressor-compress.modal.run`). First deploy takes ~100 s — it builds the image and bakes the ModernBERT base weights and the NLTK sentence tokenizer into it, so cold starts only read the fine-tuned weights from the Volume.

To serve a different checkpoint: upload it under a new name (`/run10`), change `CHECKPOINT_NAME` in `modal_app.py`, redeploy. Redeploys reuse the cached image and take seconds.

Smoke test (runs one real request through a T4, costs a fraction of a cent):

```bash
../../.venv/bin/modal run modal_app.py
```

## Making requests

`POST` JSON to the endpoint URL.

**Request:**

| field | type | default | meaning |
| --- | --- | --- | --- |
| `query` | string | required | the question the snippet should answer |
| `document` | string **or** list of strings | required | raw text, or pre-split sentence units |
| `token_threshold` | float | 0.5 | per-token keep-probability cutoff |
| `sentence_threshold` | float | 0.5 | fraction of a sentence's tokens that must clear `token_threshold` for the sentence to be kept |

If `document` is a raw string, the server splits it with NLTK. For behavior faithful to training, pass `units` yourself — split with `data_generation.core.segment.segment`, the exact unit definition the labels were built with.

Long documents are handled transparently: units are packed into 8192-token windows and the per-window results are merged, so a 300-sentence page works the same as a 5-sentence one.

**Response:**

| field | meaning |
| --- | --- |
| `kept_indices` | indices (into the unit list) the model kept |
| `kept_sentences` | the kept units, in document order |
| `snippet` | kept units joined into one string |
| `sentence_scores` | per unit: fraction of its tokens above `token_threshold` |
| `sentence_mean_probs` | per unit: mean token keep-probability (finer-grained ranking key) |
| `compression_ratio` | kept chars / total chars |

**curl:**

```bash
curl -X POST https://<url>.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what is the difference between oneOf anyOf and allOf in OpenAPI",
    "document": ["unit 0 text", "unit 1 text", "unit 2 text"]
  }'
```

**Python:**

```python
import httpx

URL = "https://<url>.modal.run"

out = httpx.post(URL, json={
    "query": "what is net churn rate in SaaS",
    "document": units,            # list[str] from segment(), or one raw string
}, timeout=60).json()             # generous timeout: first hit may cold-start

print(out["snippet"], out["compression_ratio"])
```

The model is conservative by design: on documents where nothing answers the query it keeps (almost) nothing. If you want output even on weak matches, lower the thresholds per request, or rank units yourself using `sentence_mean_probs` instead of relying on the threshold decision.

Spot check against 5 labeled corpus docs (Sonnet 4.6 gold labels, 77–307 units each): micro **P 0.88 / R 0.92 / F1 0.90**, compression ratios 4–10%. Training-corpus rows, so a sanity check, not a held-out eval — the real number is `search_evals`' accuracy-vs-tokens curve.

## Cost

On the $30/month Starter tier: a T4 costs ~$0.59/hr billed per second, and each burst of use pays for the compute plus the 120 s `scaledown_window` idle tail. That's roughly $0.02 per cold-start-plus-a-few-requests session. Keep `min_containers = 0`; pinning one container warm 24/7 would cost ~$425/month.
