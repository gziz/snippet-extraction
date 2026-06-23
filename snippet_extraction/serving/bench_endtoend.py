"""End-to-end validation of the optimized compress pipeline against fp32.

Calls the *production* ``compress_long_batch`` (now fp16 + numpy pooling +
batched length tokenization) and compares it to a forced-fp32 reference on the
real long-tail PDFs — both for wall time and for kept-snippet agreement
(Jaccard), the methodology-safety check.

    cd snippet_extraction/serving
    modal run bench_endtoend.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "runtime"))
sys.path.insert(0, str(_REPO / "common"))

APP_NAME = "query-aware-compressor-bench-e2e"
CHECKPOINT_NAME = "run9"
VOLUME_MOUNT = "/checkpoints"
BASE_MODEL = "answerdotai/ModernBERT-base"
try:
    BENCH_DOCS = Path(__file__).resolve().parents[2] / "search_evals" / "sandbox" / "bench_docs"
except IndexError:
    BENCH_DOCS = Path("/bench_docs")

DOCS = [
    ("who_142k", "traditional and complementary medicine WHO member states regulation"),
    ("un_289k", "world health report summary statistics"),
]


def _prefetch_base_assets() -> None:
    from transformers import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(BASE_MODEL)
    AutoTokenizer.from_pretrained(BASE_MODEL)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "fastapi[standard]",
        "syntok>=1.4",
        "markdown-it-py>=3.0",
        "tiktoken>=0.7",
        "numpy>=1.26",
    )
    .run_function(_prefetch_base_assets)
    .add_local_python_source("snippets_runtime")
    .add_local_python_source("snippets_common")
    .add_local_dir(str(BENCH_DOCS), remote_path="/bench_docs")
)

volume = modal.Volume.from_name("compressor-checkpoints", create_if_missing=True)
app = modal.App(APP_NAME)


@app.cls(
    image=image,
    gpu="L4",
    volumes={VOLUME_MOUNT: volume},
    scaledown_window=120,
    min_containers=0,
    timeout=1200,
)
class Bench:
    @modal.enter()
    def load(self):
        from snippets_runtime.inference import load_for_inference

        ckpt_dir = f"{VOLUME_MOUNT}/{CHECKPOINT_NAME}"
        self.model, self.tokenizer = load_for_inference(ckpt_dir, base=BASE_MODEL, device="cuda")

    @modal.method()
    def run(self, docs: list[tuple[str, str]]) -> list[dict]:
        import torch
        from snippets_common.segment import segment
        from snippets_runtime import inference
        from snippets_runtime.inference import DocRequest, compress_long_batch

        def score(units, query, dtype):
            inference.INFERENCE_DTYPE = dtype  # patch the module-level switch
            req = DocRequest(query=query, document=units)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = compress_long_batch(self.model, self.tokenizer, [req], device="cuda")
            torch.cuda.synchronize()
            return set(out[0].kept_indices), time.perf_counter() - t0

        results = []
        for name, query in docs:
            md = Path(f"/bench_docs/{name}.md").read_text()
            units = segment(md)
            # warm
            score(units, query, torch.float16)

            kept32, t32 = score(units, query, None)  # fp32 baseline
            kept16, t16 = score(units, query, torch.float16)  # production

            inter = len(kept16 & kept32)
            union = len(kept16 | kept32) or 1
            jacc = inter / union
            row = {
                "doc": name,
                "n_units": len(units),
                "fp32_s": round(t32, 2),
                "opt_s": round(t16, 2),
                "speedup": round(t32 / t16, 2),
                "nkept_fp32": len(kept32),
                "nkept_opt": len(kept16),
                "jaccard": round(jacc, 4),
            }
            print(
                f"[e2e] {name}: fp32={t32:.2f}s opt={t16:.2f}s "
                f"({t32 / t16:.2f}x) jaccard={jacc:.4f} "
                f"nkept {len(kept32)}->{len(kept16)}",
                flush=True,
            )
            results.append(row)
        return results


@app.local_entrypoint()
def main():
    rows = Bench().run.remote(DOCS)
    print("\n==== end-to-end: production (fp16+numpy+batched) vs fp32 ====")
    for r in rows:
        print(
            f"  {r['doc']:>10} ({r['n_units']:>6,} units): "
            f"fp32={r['fp32_s']:>6.2f}s  opt={r['opt_s']:>6.2f}s  "
            f"speedup={r['speedup']:>4.2f}x  jaccard={r['jaccard']}  "
            f"nkept {r['nkept_fp32']}->{r['nkept_opt']}"
        )
