"""Back-compat shim. Inference now lives in ``snippets_runtime.inference``.
Importing from here still works; prefer the new path in new code.
"""

from snippets_runtime.inference import (  # noqa: F401
    CompressResult,
    compress,
    compress_long,
    load_for_inference,
)
