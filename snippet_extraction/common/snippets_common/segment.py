"""Document body -> list of selectable units.

Segmentation is **block-aware**: we first partition the markdown into typed
blocks with a CommonMark/GFM parser (``markdown-it-py``) and only hand *prose*
paragraphs to the sentence segmenter (``syntok``). Structured blocks — tables,
code fences, headings, list items — are kept as atomic units sliced verbatim
from the source.

This is the architecture every RAG framework converges on (Unstructured,
LlamaIndex, LangChain): parse structure first, sentence-split last. It fixes the
prior bug where syntok, a prose segmenter, mangled markdown tables by breaking
at the inline-link ``)`` boundaries instead of the row newlines.

The unit schema is ``list[str]``; ``render_unit`` is the seam where v2 will
inline ``parent_heading``.
"""

from __future__ import annotations

import html
import re

import syntok.segmenter as syntok_segmenter
import tiktoken
from markdown_it import MarkdownIt

# A prose unit is left completely alone if it's at or under this size.
MAX_UNIT_TOKENS = 250
# The chunk size used once prose splitting kicks in. When a unit is over the max, it gets broken into pieces of roughly 100 tokens each.
SPLIT_TARGET_TOKENS = 100

# Atomic blocks (code fences, tables) get a higher cap than prose: splitting a
# code block or table mid-way hurts comprehension far more than splitting prose,
# so we keep them whole until they get genuinely large. Only past this cap do we
# fall back to line-boundary splitting.
MAX_BLOCK_TOKENS = 500
# Target chunk size when an oversized atomic block does have to be split.
BLOCK_SPLIT_TARGET_TOKENS = 400

# CommonMark + GFM tables. Tables are emitted as one atomic unit, so they must
# be recognized as a single block rather than a run of prose lines.
_MD = MarkdownIt("commonmark").enable("table")

# Real BPE token counts, not a whitespace-word proxy. cl100k_base matches the
# encoding used elsewhere in the repo (see data_augmentation/token_stats.py).
_ENC = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    # disallowed_special=() so literal "<|endoftext|>"-style strings in scraped
    # documents are counted as ordinary text instead of raising.
    return len(_ENC.encode(text, disallowed_special=()))


# Page-chrome substrings frequently glued onto real content by HTML
# extractors (Wikipedia, Q&A sites, listings). We strip them inline so
# syntok can segment the surrounding prose cleanly.
_CHROME_RE = re.compile(
    r"""(?ix)
    \[\s*edit\s*\]
    | \[\s*hide\s*\]
    | Contents\s*\[\s*hide\s*\]
    | From\s+Wikipedia,\s+the\s+free\s+encyclopedia
    | navigation\s+search
    | Jump\s+to:?\s+navigation
    | Retrieved\s+(\d{1,2}\s+\w+\s+\d{4}|on\s+\d{4}-\d{2}-\d{2})
    | All\s+rights\s+reserved\.?
    | Click\s+here
    | Privacy\s+Policy
    | Skip\s+to\s+(content|main)
    """
)


def _normalize(text: str) -> str:
    """Light cleanup that preserves block structure.

    Intra-line whitespace is deliberately *not* collapsed here — doing so would
    destroy code-fence indentation and table column alignment. Prose blocks
    collapse their own whitespace in ``_segment_prose``.
    """
    text = html.unescape(text)
    text = _CHROME_RE.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_oversized(unit: str) -> list[str]:
    """Word-window fallback for an over-long prose sentence."""
    if _count_tokens(unit) <= MAX_UNIT_TOKENS:
        return [unit]
    words = unit.split()
    out: list[str] = []
    cur: list[str] = []
    for word in words:
        cur.append(word)
        if _count_tokens(" ".join(cur)) >= SPLIT_TARGET_TOKENS:
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def _split_block_lines(text: str) -> list[str]:
    """Line-boundary fallback for an over-long atomic block.

    Atomic blocks (code fences, tables) stay whole up to ``MAX_BLOCK_TOKENS``.
    Past that they are split on whole lines (table rows, code lines) so we
    never cut a row in the middle the way a word/character splitter would.
    """
    if _count_tokens(text) <= MAX_BLOCK_TOKENS:
        return [text]
    out: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for line in text.split("\n"):
        w = _count_tokens(line)
        if cur and cur_tokens + w > BLOCK_SPLIT_TARGET_TOKENS:
            out.append("\n".join(cur))
            cur, cur_tokens = [], 0
        cur.append(line)
        cur_tokens += w
    if cur:
        out.append("\n".join(cur))
    return out


def _segment_prose(text: str) -> list[str]:
    """Sentence-segment a prose fragment with syntok."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    units: list[str] = []
    for paragraph in syntok_segmenter.process(text):
        for sentence in paragraph:
            sent = "".join(tok.spacing + tok.value for tok in sentence).strip()
            if sent:
                units.extend(_split_oversized(sent))
    return units


def _slice(lines: list[str], token_map: list[int] | None) -> str:
    """Verbatim source slice for a token's [start, end) line span."""
    if not token_map:
        return ""
    return "\n".join(lines[token_map[0] : token_map[1]]).strip()


def _matching_close(tokens: list, open_idx: int, open_type: str, close_type: str) -> int:
    """Index of the close token matching the opener at ``open_idx``."""
    depth = 0
    for j in range(open_idx, len(tokens)):
        if tokens[j].type == open_type:
            depth += 1
        elif tokens[j].type == close_type:
            depth -= 1
            if depth == 0:
                return j
    return len(tokens) - 1


def segment(body: str) -> list[str]:
    """Split a document body into a list of unit strings."""
    normalized = _normalize(body)
    lines = normalized.split("\n")
    tokens = _MD.parse(normalized)

    units: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        ttype = tok.type

        if ttype in ("fence", "code_block"):
            # Code block: atomic, verbatim (fence markers and indentation kept).
            units.extend(_split_block_lines(_slice(lines, tok.map)))
            i += 1

        elif ttype == "table_open":
            # Table: one atomic unit; only split (by rows) if oversized.
            units.extend(_split_block_lines(_slice(lines, tok.map)))
            i = _matching_close(tokens, i, "table_open", "table_close") + 1

        elif ttype == "heading_open":
            # Heading: atomic, verbatim (keeps the `#` markers).
            units.extend(_split_block_lines(_slice(lines, tok.map)))
            i = _matching_close(tokens, i, "heading_open", "heading_close") + 1

        elif ttype == "paragraph_open":
            # Prose: hand the inline text to syntok.
            inline = tokens[i + 1]
            units.extend(_segment_prose(inline.content))
            i = _matching_close(tokens, i, "paragraph_open", "paragraph_close") + 1

        elif ttype in ("bullet_list_open", "ordered_list_open"):
            close_type = ttype.replace("_open", "_close")
            list_end = _matching_close(tokens, i, ttype, close_type)
            k = i + 1
            while k < list_end:
                if tokens[k].type == "list_item_open":
                    units.extend(_split_block_lines(_slice(lines, tokens[k].map)))
                    k = _matching_close(tokens, k, "list_item_open", "list_item_close") + 1
                else:
                    k += 1
            i = list_end + 1

        elif ttype == "blockquote_open":
            # Strip the `>` markers, then treat the contents as prose.
            quoted = _slice(lines, tok.map)
            unquoted = re.sub(r"(?m)^\s*>\s?", "", quoted)
            units.extend(_segment_prose(unquoted))
            i = _matching_close(tokens, i, "blockquote_open", "blockquote_close") + 1

        elif ttype == "hr":
            i += 1

        elif tok.map is not None and tok.nesting >= 0:
            # Generic fallback (html_block, etc.): treat as prose.
            units.extend(_segment_prose(_slice(lines, tok.map)))
            if tok.nesting == 1:
                i = _matching_close(tokens, i, ttype, ttype.replace("_open", "_close")) + 1
            else:
                i += 1

        else:
            i += 1

    return units


def render_unit(unit: str) -> str:
    """Render a unit for the labeler prompt / training input.

    v1: identity. v2 will inline parent_heading here.
    """
    return unit
