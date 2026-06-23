"""Per-sentence mean pooling in ``SentenceCompressor.pool_sentence_logits``.

``pool_sentence_logits`` is a ``@staticmethod`` operating on plain tensors, so
it is tested WITHOUT constructing ``SentenceCompressor`` (that would download
ModernBERT and require the network). The contract: mean-pool token logits per
sentence id, route ids ``< 0`` or ``>= max_units`` to an ignore bucket, and
clamp empty-sentence counts to 1.0 so a sentence with no tokens yields 0.0.
"""

import torch
from snippets_runtime.model import SentenceCompressor


def test_mean_pool_per_sentence_is_correct():
    token_logits = torch.tensor([[1.0, 3.0, 5.0, 7.0]])
    sentence_ids = torch.tensor([[0, 0, 1, 1]])
    out = SentenceCompressor.pool_sentence_logits(token_logits, sentence_ids, max_units=2)
    assert out.shape == (1, 2)
    # unit 0 = mean(1, 3) = 2 ; unit 1 = mean(5, 7) = 6
    assert torch.allclose(out, torch.tensor([[2.0, 6.0]]))


def test_negative_and_overflow_ids_are_ignored():
    token_logits = torch.tensor([[1.0, 3.0, 5.0, 99.0, 99.0]])
    # id -1 (outside any sentence) and id 5 (>= max_units) must be dropped into
    # the ignore bucket and not contribute to any unit.
    sentence_ids = torch.tensor([[0, 0, 1, -1, 5]])
    out = SentenceCompressor.pool_sentence_logits(token_logits, sentence_ids, max_units=3)
    # unit 0 = mean(1, 3) = 2 ; unit 1 = mean(5) = 5 ; the 99.0 tokens excluded.
    assert torch.allclose(out[:, :2], torch.tensor([[2.0, 5.0]]))


def test_empty_sentence_yields_zero_via_count_clamp():
    token_logits = torch.tensor([[2.0, 4.0]])
    sentence_ids = torch.tensor([[0, 0]])
    # unit 1 receives no tokens: sum 0 / count clamped to 1.0 -> 0.0 (defined).
    out = SentenceCompressor.pool_sentence_logits(token_logits, sentence_ids, max_units=2)
    assert out.shape == (1, 2)
    assert out[0, 0].item() == 3.0
    assert out[0, 1].item() == 0.0


def test_batched_input_pools_independently_per_row():
    token_logits = torch.tensor(
        [
            [1.0, 3.0, 10.0, 10.0],  # row 0: unit0 from first two, rest ignored
            [4.0, 6.0, 8.0, 12.0],  # row 1: unit0 = mean(4,6), unit1 = mean(8,12)
        ]
    )
    sentence_ids = torch.tensor(
        [
            [0, 0, -1, -1],
            [0, 0, 1, 1],
        ]
    )
    out = SentenceCompressor.pool_sentence_logits(token_logits, sentence_ids, max_units=2)
    assert out.shape == (2, 2)
    assert torch.allclose(out[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(out[1], torch.tensor([5.0, 10.0]))
