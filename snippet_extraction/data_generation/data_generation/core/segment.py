"""Back-compat shim. The canonical segmenter now lives in
``snippets_common.segment`` (torch-free shared core), so training-label
generation and serving share one unit definition. Importing from here still
works; prefer ``from snippets_common.segment import ...`` in new code.
"""

from snippets_common.segment import (  # noqa: F401  # noqa: F401  - non-public helpers used by tests
    _CHROME_RE,
    _ENC,
    BLOCK_SPLIT_TARGET_TOKENS,
    MAX_BLOCK_TOKENS,
    MAX_UNIT_TOKENS,
    SPLIT_TARGET_TOKENS,
    _count_tokens,
    _matching_close,
    _normalize,
    _segment_prose,
    _slice,
    _split_block_lines,
    _split_oversized,
    render_unit,
    segment,
)
