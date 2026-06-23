"""Snippet-extraction evaluation harness.

Evaluates, per provider, how well its extracted snippets cover the gold
relevant text for a query — conditioned on the document that provider itself
surfaced. Retrieval (which doc was returned) is NOT compared across providers;
only extraction quality on each provider's own doc is scored.

Primary metric: token-level precision/recall/F1 (see ``metrics.py``).
"""
