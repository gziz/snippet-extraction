"""Prompt-registry invariants.

The pinned hashes are load-bearing: existing label files record them, and the
resume/dedupe logic keys on prompt_version. If a hash test fails, the prompt
text or user-prompt builder changed — either revert, or bump the version and
treat it as a new prompt.
"""

from data_generation.core.labeling.prompts import (
    PROMPTS,
    build_user_prompt,
    merge_anchor,
)

# prompt_hash() fingerprints the system text + the user-prompt builder's source,
# so these values are stable across Python interpreter versions. Bump them only
# when intentionally changing a prompt (and bump its version alongside).
PINNED_HASHES = {"v2": "2a6223ce155a", "v3": "de16e637da21"}


def test_prompt_hashes_pinned():
    for version, expected in PINNED_HASHES.items():
        assert PROMPTS[version].prompt_hash() == expected, (
            f"{version} prompt_hash changed — existing labels reference {expected}"
        )


def test_build_user_prompt_shape():
    out = build_user_prompt("q?", ["alpha", "beta"])
    assert out == "QUERY:\nq?\n\nDOCUMENT UNITS:\n[0] alpha\n[1] beta"


def test_merge_anchor_folds_anchor_into_answer():
    assert merge_anchor({"relevant_unit_ids": [3, 1], "anchor_unit_id": 0}, 10) == [0, 1, 3]


def test_merge_anchor_ignores_anchor_on_empty_answer():
    assert merge_anchor({"relevant_unit_ids": [], "anchor_unit_id": 0}, 10) == []


def test_merge_anchor_drops_out_of_range_ids():
    assert merge_anchor({"relevant_unit_ids": [2, 99, -1], "anchor_unit_id": 50}, 10) == [2]


def test_v2_parse_has_no_context_set():
    ids = PROMPTS["v2"].parse({"relevant_unit_ids": [1], "anchor_unit_id": 0}, 5)
    assert ids.relevant == [0, 1]
    assert ids.context is None


def test_v3_parse_enforces_superset():
    # Model "slipped": context missing a relevant id — code must repair it.
    ids = PROMPTS["v3"].parse({"relevant_unit_ids": [1, 3], "context_unit_ids": [3, 4]}, 10)
    assert ids.relevant == [1, 3]
    assert ids.context == [1, 3, 4]


def test_v3_parse_validates_ranges():
    ids = PROMPTS["v3"].parse({"relevant_unit_ids": [0, 99], "context_unit_ids": [0, 1, 99]}, 5)
    assert ids.relevant == [0]
    assert ids.context == [0, 1]


def test_schemas_require_expected_fields():
    v2 = PROMPTS["v2"].schema["schema"]
    assert v2["required"] == ["relevant_unit_ids", "anchor_unit_id"]
    v3 = PROMPTS["v3"].schema["schema"]
    assert v3["required"] == ["relevant_unit_ids", "context_unit_ids"]
