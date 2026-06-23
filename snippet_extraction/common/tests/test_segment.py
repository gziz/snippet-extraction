"""Block-aware segmenter invariants for snippets_common.

The segmenter parses markdown structure first and only sentence-splits prose:
tables and code fences stay atomic and verbatim, headings keep their markers,
list items split, blockquotes are unquoted then segmented as prose. These tests
pin the real contracts (including the original table-mangling bug) and the
oversized-split fallbacks.
"""

from snippets_common.segment import (
    MAX_BLOCK_TOKENS,
    MAX_UNIT_TOKENS,
    _count_tokens,
    render_unit,
    segment,
)


def test_prose_splits_into_sentences():
    units = segment("First sentence here. Second sentence follows. Third one ends.")
    assert units == [
        "First sentence here.",
        "Second sentence follows.",
        "Third one ends.",
    ]


def test_code_fence_is_one_atomic_unit_verbatim():
    body = "Before.\n\n```python\ndef f(x):\n    return x  # indented\n```\n\nAfter."
    units = segment(body)
    code = [u for u in units if "def f" in u]
    assert len(code) == 1
    block = code[0]
    # Fence markers and indentation are preserved verbatim.
    assert block.startswith("```python")
    assert block.endswith("```")
    assert "    return x  # indented" in block
    assert units[0] == "Before."
    assert units[-1] == "After."


def test_table_is_single_unit_not_broken_at_inline_link():
    # The original bug: syntok broke GFM tables at the inline-link ``)``
    # boundary. The block-aware segmenter must keep the whole table as one unit.
    body = (
        "Intro paragraph.\n\n"
        "| Name | Link |\n"
        "|------|------|\n"
        "| A | [alpha](http://a.example) |\n"
        "| B | [beta](http://b.example) |\n\n"
        "Outro paragraph."
    )
    units = segment(body)
    tables = [u for u in units if "|" in u]
    assert len(tables) == 1
    table = tables[0]
    # Every row survives inside the single unit; the ``)`` did not split it.
    assert "[alpha](http://a.example)" in table
    assert "[beta](http://b.example)" in table
    assert table.count("\n") == 3
    assert units[0] == "Intro paragraph."
    assert units[-1] == "Outro paragraph."


def test_heading_keeps_hash_markers():
    units = segment("## Title here\n\nSome prose follows it.")
    assert units[0] == "## Title here"
    assert "Some prose follows it." in units


def test_list_items_are_separate_units():
    units = segment("- first item\n- second item\n- third item")
    assert units == ["- first item", "- second item", "- third item"]


def test_blockquote_marker_stripped_and_segmented_as_prose():
    units = segment("> Quoted line one. Quoted line two.")
    assert units == ["Quoted line one.", "Quoted line two."]
    assert all(not u.startswith(">") for u in units)


def test_oversized_prose_sentence_is_split():
    # One giant "sentence" with no sentence boundaries, way over the cap.
    long_sentence = ("word " * 600).strip()
    units = segment(long_sentence)
    assert len(units) > 1
    assert all(_count_tokens(u) <= MAX_UNIT_TOKENS for u in units)


def test_oversized_atomic_block_splits_on_line_boundaries():
    lines = "\n".join(f"line_{i} = compute_value({i})" for i in range(300))
    fence = f"```python\n{lines}\n```"
    assert _count_tokens(fence) > MAX_BLOCK_TOKENS
    units = segment(fence)
    assert len(units) > 1
    # Split on whole lines: no piece ends mid-line (every newline is preserved
    # as a row boundary, never cut inside a line).
    for piece in units:
        for line in piece.split("\n"):
            assert line == "" or line.startswith("```") or line.startswith("line_")


def test_html_entities_unescaped():
    units = segment("Tom &amp; Jerry are &lt;friends&gt;.")
    assert units == ["Tom & Jerry are <friends>."]


def test_page_chrome_substrings_stripped():
    units = segment(
        "From Wikipedia, the free encyclopedia Romansh is a language. [edit] "
        "It is spoken in Switzerland."
    )
    joined = " ".join(units)
    assert "[edit]" not in joined
    assert "From Wikipedia" not in joined
    assert "Romansh is a language." in units
    assert "It is spoken in Switzerland." in units


def test_empty_or_whitespace_returns_empty_list():
    assert segment("") == []
    assert segment("   \n\n  \t ") == []


def test_render_unit_is_identity_in_v1():
    assert render_unit("## A *heading*") == "## A *heading*"
