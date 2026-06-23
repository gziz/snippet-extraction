"""Block-aware segmenter invariants (tables/code atomic, prose sentence-split)."""

from data_generation.core.segment import (
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


def test_table_is_one_atomic_unit():
    body = (
        "Intro paragraph.\n\n"
        "| col a | col b |\n"
        "|-------|-------|\n"
        "| 1     | 2     |\n"
        "| 3     | 4     |\n\n"
        "Outro paragraph."
    )
    units = segment(body)
    tables = [u for u in units if "|" in u]
    assert len(tables) == 1
    assert "col a" in tables[0] and "| 3" in tables[0]
    assert units[0] == "Intro paragraph."
    assert units[-1] == "Outro paragraph."


def test_code_fence_kept_verbatim():
    body = "Before.\n\n```python\ndef f(x):\n    return x  # indented\n```\n\nAfter."
    units = segment(body)
    code = [u for u in units if "def f" in u]
    assert len(code) == 1
    assert "    return x" in code[0]  # indentation preserved
    assert code[0].startswith("```")


def test_heading_is_atomic_with_markers():
    units = segment("# Title here\n\nSome prose follows it.")
    assert units[0] == "# Title here"


def test_list_items_are_separate_units():
    units = segment("- first item\n- second item\n- third item")
    assert units == ["- first item", "- second item", "- third item"]


def test_chrome_stripped():
    units = segment("Romansh is a language. [edit] It is spoken in Switzerland.")
    joined = " ".join(units)
    assert "[edit]" not in joined


def test_oversized_prose_is_split():
    long_sentence = "word " * 600  # one giant "sentence", way over the cap
    units = segment(long_sentence.strip())
    assert len(units) > 1
    assert all(_count_tokens(u) <= MAX_UNIT_TOKENS for u in units)


def test_empty_body():
    assert segment("") == []
    assert segment("   \n\n  ") == []


def test_render_unit_is_identity_in_v1():
    assert render_unit("## A *heading*") == "## A *heading*"
