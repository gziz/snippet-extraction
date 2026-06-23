"""Token-PRF metric conventions (multiset overlap + empty-set edge cases)."""

import math

from data_generation.core.snippet_eval.metrics import join_snippets, token_prf, tokenize


def test_multiset_overlap():
    prf = token_prf("the cat sat on the mat", "the cat sat there on a mat mat")
    assert prf.n_pred == 6 and prf.n_gold == 8 and prf.n_overlap == 5
    assert math.isclose(prf.precision, 5 / 6)
    assert math.isclose(prf.recall, 5 / 8)


def test_repeated_token_only_counts_to_gold_multiplicity():
    prf = token_prf("mat mat mat", "mat")
    assert prf.n_overlap == 1


def test_empty_conventions():
    assert (token_prf("", "").precision, token_prf("", "").recall, token_prf("", "").f1) == (
        1.0,
        1.0,
        1.0,
    )
    p = token_prf("", "gold text")
    assert (p.precision, p.recall, p.f1) == (1.0, 0.0, 0.0)
    p = token_prf("pred text", "")
    assert (p.precision, p.recall, p.f1) == (0.0, 1.0, 0.0)


def test_tokenize_normalizes_accents_and_case():
    assert tokenize("Café NOISE") == ["cafe", "noise"]


def test_join_snippets_drops_blanks():
    assert join_snippets(["a", "", "  ", "b"]) == "a\nb"
