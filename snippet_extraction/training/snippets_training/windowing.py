"""Back-compat shim. The real implementation now lives in
``snippets_common.windowing`` (torch-free shared core). Importing from here
still works for existing callers; prefer the new path in new code.
"""

from snippets_common.windowing import (  # noqa: F401
    pack_windows,
    render_snippet,
    select_under_budget,
)
