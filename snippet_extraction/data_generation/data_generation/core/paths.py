"""Canonical filesystem locations and environment loading.

Every pipeline and library module derives paths from here instead of
computing ``Path(__file__).resolve().parents[n]`` locally, so a file can be
moved without silently pointing at the wrong directory.

Layout::

    snippet_extraction/data_generation/   # component root
      .env                  # API keys (loaded via load_env())
      data_generation/      # the importable package
        core/               # library code
        pipelines/          # CLI entry points
      data/                 # all generated artifacts (gitignored)
        labels/             # JSONL label/corpus files
        queries/            # synthesized query sets
        indexes/            # MS MARCO byte-offset / BM25 pickles
        snippet_runs/       # snippet-eval run directories
      reports/              # human-readable experiment write-ups (tracked)
"""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]  # data_generation/ (the package)
COMPONENT_ROOT = PACKAGE_ROOT.parent  # snippet_extraction/data_generation/
REPO_ROOT = COMPONENT_ROOT.parents[1]

DATA_DIR = COMPONENT_ROOT / "data"
LABELS_DIR = DATA_DIR / "labels"
QUERIES_DIR = DATA_DIR / "queries"
INDEXES_DIR = DATA_DIR / "indexes"
SNIPPET_RUNS_DIR = DATA_DIR / "snippet_runs"
REGRESSION_FILE = DATA_DIR / "regression.jsonl"
REPORTS_DIR = COMPONENT_ROOT / "reports"

# MS MARCO source TSVs live in the sibling document_ranking component.
MSMARCO_DATA_DIR = REPO_ROOT / "document_ranking" / "data"

_env_loaded = False


def load_env() -> None:
    """Load the component root's ``.env`` into the process environment once.

    Called by clients that need API keys at construction time; importing a
    library module never touches the environment.
    """
    global _env_loaded
    if _env_loaded:
        return
    from dotenv import load_dotenv

    load_dotenv(COMPONENT_ROOT / ".env")
    _env_loaded = True
