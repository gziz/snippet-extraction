"""Highlight-fragment -> unit alignment (legacy Exa path, still used by tests/QA)."""

from data_generation.core.retrievers.exa_align import align, split_fragments


def test_split_fragments_drops_tiny_noise():
    frags = split_fragments("A real fragment of text here. [...] tiny [...] Another real fragment.")
    assert frags == ["A real fragment of text here.", "Another real fragment."]


def test_align_matches_exact_units():
    units = [
        "Consumer groups let multiple consumers share topic partitions between them.",
        "Totally unrelated text about cooking pasta for dinner tonight with sauce.",
        "A rebalance is triggered whenever a consumer joins or leaves the group.",
    ]
    frags = [units[0], units[2]]
    stats = align(frags, units)
    assert stats.matched_unit_ids == [0, 2]
    assert stats.n_matched == 2
    assert stats.n_dropped == 0


def test_align_drops_unmatched_fragments():
    units = ["Completely different content that shares no tokens with the fragment text."]
    stats = align(
        ["The quick brown fox jumps over a lazy sleeping dog near the river bank."], units
    )
    assert stats.matched_unit_ids == []
    assert stats.n_dropped == 1
