"""Back-compat shim. The model now lives in ``snippets_runtime.model``.
Importing from here still works; prefer the new path in new code.
"""

from snippets_runtime.model import (  # noqa: F401
    IGNORE_INDEX,
    SentenceCompressor,
)
