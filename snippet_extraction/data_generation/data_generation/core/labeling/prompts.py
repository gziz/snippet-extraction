"""Versioned labeler prompts: system text, output schema, and response parsing.

A :class:`LabelPrompt` bundles everything that defines one labeling prompt —
the system message, the structured-output schema (plus provider-specific
variants of it), and the parser that turns the model's JSON into validated
unit-id lists. Transports (``clients.py``) are prompt-agnostic: trying a new
prompt version means adding an entry to :data:`PROMPTS`, not a new module.

Versions:

- **v2** — strict answer set + required ``anchor_unit_id`` (folded into the
  stored label by ``merge_anchor``). The production prompt.
- **v3** — strict answer set + ``context_unit_ids`` superset (answer plus
  supporting context). Used in the v2-vs-v3 A/B (see ``pipelines/ab_labeler.py``).

``prompt_hash()`` fingerprints the system text + the user-prompt builder's
source so accidental edits are visible in stored rows. The v2/v3 hashes are
pinned in ``tests/test_prompts.py``; existing label files reference them.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

from ..segment import render_unit

DEFAULT_VERSION = "v2"

V2_SYSTEM = (
    "You are labeling data for an extractive context-compression task. "
    "You will be given a user query and a document split into numbered units. "
    "Return the IDs of the units whose text is directly needed to answer the query — "
    "the smallest set a reader would have to read to get the answer.\n\n"
    "Decision procedure:\n\n"
    "1. First decide whether the document actually answers the query.\n"
    '   - "Actually answers" means a reader could form a correct, specific answer '
    "using only the units you keep.\n"
    "   - A document that is only topically related — same domain, shares keywords, "
    "mentions the entity — but does not contain the answer is NOT a match. Return [] in that case.\n"
    '   - Pay attention to the precise meaning of the query. A query about "moving cost" '
    '(relocating a household) is not answered by a document about "moving-average cost" '
    '(accounting). A query about "what X feels like" is not answered by a document about '
    '"how to treat X". When in doubt about intent, return [].\n\n'
    "2. If the document does answer the query, return the minimal set of unit IDs that "
    "carries the answer.\n"
    "   - Keep every unit that contributes a distinct fact, number, name, definition, step, "
    'or reason that the query asks for. Do not drop corroborating facts for list / "why" / '
    '"how" / multi-part queries.\n'
    "   - Do not keep background, introductions, navigation, page chrome, reference/citation "
    "lines, link lists, advertisements, or repeated near-duplicate answers. If the same fact "
    "appears in multiple units, keep one (prefer the clearest and most complete).\n"
    "   - Do not keep a unit just because it contains keywords from the query. Keep it only "
    "if its content is part of the answer.\n\n"
    "3. Fill in anchor_unit_id (required field). The kept units are shown to a reader who has "
    "NEVER heard of this topic, sees ONLY these units, and does not have the document or the "
    "query. Do NOT use your own knowledge to fill the gap: even if you can infer the subject "
    'from jargon (e.g. "consumer group", "topics", "rebalance"), a naive reader or a '
    "keyword-matching system cannot. anchor_unit_id is the ID of the ONE unit that literally "
    'states the subject\'s proper name (e.g. the actual word "Kafka").\n'
    "   - If a unit in relevant_unit_ids already names the subject, use that unit's ID.\n"
    "   - Otherwise use the page title, a heading, or the first unit that names the subject — "
    "even if that unit is chrome you would never keep as answer, and even if it covers a "
    "DIFFERENT aspect of the subject than the query asks about. The anchor only needs to name "
    "the subject; it does not need to answer the query.\n"
    "   - anchor_unit_id is about naming only. Do not add it to or remove anything from "
    "relevant_unit_ids because of it.\n"
    "   - Set anchor_unit_id to null only when relevant_unit_ids is empty.\n\n"
    "4. If no unit is relevant, return an empty list. Returning [] is the correct answer on "
    "documents that don't address the query; do not feel obligated to keep something."
)

V2_SCHEMA = {
    "name": "relevant_units",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["relevant_unit_ids", "anchor_unit_id"],
        "properties": {
            "relevant_unit_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            },
            "anchor_unit_id": {
                "type": ["integer", "null"],
                "description": (
                    "ID of the one unit that names the answer's subject by its proper "
                    "name (see step 3). null only when relevant_unit_ids is empty."
                ),
            },
        },
    },
}

# Gemini's response_schema uses an OpenAPI subset that doesn't accept
# JSON-Schema-specific keys like "additionalProperties", so each prompt
# carries a hand-written Gemini variant.
V2_GEMINI_SCHEMA = {
    "type": "object",
    "required": ["relevant_unit_ids", "anchor_unit_id"],
    "properties": {
        "relevant_unit_ids": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "anchor_unit_id": {
            "type": "integer",
            "nullable": True,
        },
    },
}

# json_object mode (OpenRouter) doesn't enforce a schema, so spell the exact
# shape out. The literal word "json" must appear in the prompt for that mode.
V2_JSON_SHAPE_INSTRUCTION = (
    "Respond with a single JSON object of the form "
    '{"relevant_unit_ids": [<integer unit ids>], "anchor_unit_id": <integer or null>}. '
    "Use an empty array (and null anchor) when no unit is relevant. "
    "Output only that JSON object."
)

# v3: the entire v2 strict procedure, then a second (context-superset) task.
V3_SYSTEM = (
    "You are labeling data for an extractive context-compression task. "
    "You will be given a user query and a document split into numbered units. "
    "You will perform two tasks. The data for each is independent of one another.\n\n"
    "There's one thing to consider:\n"
    "- First decide whether the document actually answers the query.\n"
    '   - "Actually answers" means a reader could form a correct, specific answer '
    "using only the units you keep.\n"
    "   - A document that is only topically related — same domain, shares keywords, "
    "mentions the entity — but does not contain the answer is NOT a match. Return [] in that case.\n"
    '   - Pay attention to the precise meaning of the query. A query about "moving cost" '
    '(relocating a household) is not answered by a document about "moving-average cost" '
    '(accounting). A query about "what X feels like" is not answered by a document about '
    '"how to treat X". When in doubt about intent, return [].\n\n\n'
    "Task 1 -- `relevant_unit_ids` : If the document does answer the query, return the minimal set of unit IDs that "
    "carries the answer.\n"
    "   - Keep every unit that contributes a distinct fact, number, name, definition, step, "
    'or reason that the query asks for. Do not drop corroborating facts for list / "why" / '
    '"how" / multi-part queries.\n'
    "   - Ground the answer so the kept set is self-contained: a reader who sees ONLY these "
    "units (without the rest of the document) should be able "
    "to tell what the answer is ABOUT. If the answer-bearing "
    "units are generic — e.g. a bare list of steps, numbers, or settings that never names the "
    "subject, product, or scope — also include the minimal anchoring unit(s) that identify it. "
    "   - Do not keep background, introductions, navigation, page chrome, reference/citation "
    "lines, link lists, advertisements, or repeated near-duplicate answers. If the same fact "
    "appears in multiple units, keep one (prefer the clearest and most complete).\n"
    "   - Do not keep a unit just because it contains keywords from the query. Keep it only "
    "if its content is part of the answer.\n\n"
    "- If no unit is relevant, return an empty list. Returning [] is the correct answer on "
    "documents that don't address the query; do not feel obligated to keep something."
    "\n"
    "TASK 2 — `context_unit_ids` (answer + supporting context):\n"
    "   - This is a SEPARATE, more generous pass. Start from `relevant_unit_ids` and ADD units; "
    "you may only add, never remove. It MUST be a superset of `relevant_unit_ids`.\n"
    "   - Add units that the answer depends on or that materially aid understanding: the "
    "definition or setup a fact relies on, a caveat or condition that qualifies it, a concrete "
    "example that illustrates it, or the immediately surrounding explanation that makes the "
    "answer intelligible to someone who didn't already know it.\n"
    "   - Still drop pure noise: navigation, page chrome, ads, cookie/login banners, "
    "reference/citation lines, link lists, author/date bylines, related-article teasers, and "
    "near-duplicate repetitions. Noise is excluded from BOTH sets.\n"
    "   - Do NOT expand to the whole document. Include only context genuinely tied to this "
    "answer. Topically-related material that doesn't help understand THIS answer is not "
    "context — leave it out.\n\n"
    "If the document does not answer the query, BOTH sets are empty. The context set is never "
    "a consolation prize for an off-topic document.\n\n"
    "Return `relevant_unit_ids` first (Task 1, the strict answer — unaffected by Task 2), then "
    "`context_unit_ids` (Task 2, answer + supporting context)."
)

V3_SCHEMA = {
    "name": "relevant_and_context_units",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["relevant_unit_ids", "context_unit_ids"],
        "properties": {
            "relevant_unit_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            },
            "context_unit_ids": {
                "type": "array",
                "items": {"type": "integer", "minimum": 0},
            },
        },
    },
}

V3_GEMINI_SCHEMA = {
    "type": "object",
    "required": ["relevant_unit_ids", "context_unit_ids"],
    "properties": {
        "relevant_unit_ids": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "context_unit_ids": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
}

V3_JSON_SHAPE_INSTRUCTION = (
    "Respond with a single JSON object of the form "
    '{"relevant_unit_ids": [<integer unit ids>], "context_unit_ids": [<integer unit ids>]}. '
    "context_unit_ids must be a superset of relevant_unit_ids; both are empty arrays "
    "when no unit is relevant. Output only that JSON object."
)


def build_user_prompt(query: str, units: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {render_unit(u)}" for i, u in enumerate(units))
    return f"QUERY:\n{query}\n\nDOCUMENT UNITS:\n{numbered}"


def merge_anchor(parsed: dict, n_units: int) -> list[int]:
    """Validated answer ids ∪ anchor id.

    The labeler reports the grounding anchor in a separate required field so the
    relevance decision stays untouched; the anchor is folded into the stored
    label here. An anchor on an empty answer set is ignored (a no-match doc
    needs no grounding).
    """
    ids = [
        i for i in parsed.get("relevant_unit_ids", []) if isinstance(i, int) and 0 <= i < n_units
    ]
    anchor = parsed.get("anchor_unit_id")
    if ids and isinstance(anchor, int) and 0 <= anchor < n_units:
        ids.append(anchor)
    return sorted(set(ids))


@dataclass
class ParsedIds:
    relevant: list[int]
    context: list[int] | None = None


def _valid_ids(ids, n_units: int) -> list[int]:
    return [i for i in ids if isinstance(i, int) and 0 <= i < n_units]


def _parse_v2(parsed: dict, n_units: int) -> ParsedIds:
    return ParsedIds(relevant=merge_anchor(parsed, n_units))


def _parse_v3(parsed: dict, n_units: int) -> ParsedIds:
    relevant = set(_valid_ids(parsed.get("relevant_unit_ids", []), n_units))
    # Enforce the superset invariant even if the model slips: context always
    # includes the strict answer.
    context = set(_valid_ids(parsed.get("context_unit_ids", []), n_units)) | relevant
    return ParsedIds(relevant=sorted(relevant), context=sorted(context))


@dataclass(frozen=True)
class LabelPrompt:
    version: str
    system: str
    schema: dict  # OpenAI strict json_schema form
    gemini_schema: dict  # OpenAPI-subset variant for Gemini
    json_shape_instruction: str  # for json_object-mode providers (OpenRouter)
    tool_name: str  # for Anthropic tool-use forcing
    tool_description: str
    parse: Callable[[dict, int], ParsedIds] = field(repr=False)

    def prompt_hash(self) -> str:
        """Stable hash of system text + builder source; bumps on accidental edits.

        Hashes the user-prompt builder's *source text* rather than its compiled
        bytecode (``co_code``) so the fingerprint is stable across Python
        interpreter versions instead of only matching the one it was captured on.
        """
        h = hashlib.sha1()
        h.update(self.system.encode())
        h.update(inspect.getsource(build_user_prompt).encode())
        return h.hexdigest()[:12]


PROMPTS: dict[str, LabelPrompt] = {
    "v2": LabelPrompt(
        version="v2",
        system=V2_SYSTEM,
        schema=V2_SCHEMA,
        gemini_schema=V2_GEMINI_SCHEMA,
        json_shape_instruction=V2_JSON_SHAPE_INSTRUCTION,
        tool_name="return_relevant_units",
        tool_description="Return the IDs of units whose content is needed to answer the query.",
        parse=_parse_v2,
    ),
    "v3": LabelPrompt(
        version="v3",
        system=V3_SYSTEM,
        schema=V3_SCHEMA,
        gemini_schema=V3_GEMINI_SCHEMA,
        json_shape_instruction=V3_JSON_SHAPE_INSTRUCTION,
        tool_name="return_units",
        tool_description=(
            "Return the strict answer unit IDs and the broader "
            "answer-plus-context unit IDs (a superset of the answer)."
        ),
        parse=_parse_v3,
    ),
}
