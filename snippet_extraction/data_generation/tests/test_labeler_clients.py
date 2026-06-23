"""The labeler base loop and transport parsing — the code that writes labels.

No network: transports get fake SDK clients; the base retry loop gets a
scripted ``_call``. These tests pin the failure-mode contract (status strings,
retry behavior, truncation flagging) that downstream resume/audit logic
depends on.
"""

import asyncio
from types import SimpleNamespace

import pytest
from data_generation.core.labeling.clients import (
    AsyncBedrockClaudeLabeler,
    AsyncLabelerBase,
    AsyncOpenRouterLabeler,
    _Call,
    _retry_after_seconds,
)
from data_generation.core.labeling.prompts import PROMPTS


def bare(cls, prompt=PROMPTS["v2"]):
    """Instantiate a transport without env/credentials: skip its __init__."""
    obj = cls.__new__(cls)
    AsyncLabelerBase.__init__(obj, prompt)
    obj.deployment = "fake-model"
    return obj


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Base loop: retry / fail / parse
# ---------------------------------------------------------------------------


class ScriptedLabeler(AsyncLabelerBase):
    provider = "fake"

    def __init__(self, prompt, n_failures=0, classify=lambda e: ("retry", 0.001), call_result=None):
        super().__init__(prompt)
        self.deployment = "fake-model"
        self.n_failures = n_failures
        self.classify = classify
        self.calls = 0
        self.call_result = call_result or _Call(
            {"relevant_unit_ids": [1], "anchor_unit_id": 0}, "{}", 10, 20
        )

    async def _call(self, user_prompt):
        self.calls += 1
        if self.calls <= self.n_failures:
            raise RuntimeError("boom")
        return self.call_result

    def _classify_error(self, exc):
        return self.classify(exc)


UNITS = ["# Kafka", "Consumers share partitions.", "Unrelated."]


def test_ok_path_parses_v2_with_anchor():
    j = ScriptedLabeler(PROMPTS["v2"])
    res = run(j.label("q", UNITS))
    assert res.status == "ok"
    assert res.relevant_unit_ids == [0, 1]  # anchor 0 folded in
    assert res.context_unit_ids is None
    assert (res.tokens_in, res.tokens_out) == (10, 20)


def test_ok_path_parses_v3_superset():
    j = ScriptedLabeler(
        PROMPTS["v3"],
        call_result=_Call({"relevant_unit_ids": [1], "context_unit_ids": [2]}, "{}", 1, 1),
    )
    res = run(j.label("q", UNITS))
    assert res.relevant_unit_ids == [1]
    assert res.context_unit_ids == [1, 2]  # superset enforced


def test_retries_then_succeeds():
    j = ScriptedLabeler(PROMPTS["v2"], n_failures=2)
    res = run(j.label("q", UNITS))
    assert res.status == "ok"
    assert j.calls == 3


def test_retry_exhaustion_returns_rate_limit_status():
    j = ScriptedLabeler(PROMPTS["v2"], n_failures=99)
    res = run(j.label("q", UNITS))
    assert res.status == "error:rate_limit"
    assert res.relevant_unit_ids is None
    assert j.calls == j.MAX_RETRIES + 1
    assert "boom" in res.raw


def test_fail_classification_is_terminal():
    j = ScriptedLabeler(PROMPTS["v2"], n_failures=99, classify=lambda e: ("fail", "content_filter"))
    res = run(j.label("q", UNITS))
    assert res.status == "content_filter"
    assert res.relevant_unit_ids is None
    assert j.calls == 1  # no retries on terminal errors


def test_unclassified_errors_propagate():
    j = ScriptedLabeler(PROMPTS["v2"], n_failures=99, classify=lambda e: None)
    with pytest.raises(RuntimeError):
        run(j.label("q", UNITS))


def test_non_ok_call_passes_status_through():
    j = ScriptedLabeler(
        PROMPTS["v2"], call_result=_Call(None, "garbage", 5, 6, "error:invalid_json")
    )
    res = run(j.label("q", UNITS))
    assert res.status == "error:invalid_json"
    assert res.relevant_unit_ids is None
    assert (res.tokens_in, res.tokens_out) == (5, 6)


def test_retry_after_header_parsing():
    exc = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "7"}))
    assert _retry_after_seconds(exc) == 7.0
    assert _retry_after_seconds(SimpleNamespace(response=None)) is None
    exc = SimpleNamespace(response=SimpleNamespace(headers={"Retry-After": "junk"}))
    assert _retry_after_seconds(exc) is None


# ---------------------------------------------------------------------------
# OpenRouter transport parsing (fake SDK client)
# ---------------------------------------------------------------------------


def _openrouter_with_response(content, finish_reason="stop"):
    j = bare(AsyncOpenRouterLabeler)
    resp = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
    )

    async def create(**kwargs):
        return resp

    j.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return j


def test_openrouter_truncation_is_flagged_not_empty():
    # A length-cut reasoning model yields a syntactically valid [] — the
    # client must surface error:truncated instead of a wrong empty label.
    j = _openrouter_with_response('{"relevant_unit_ids": []}', finish_reason="length")
    res = run(j.label("q", UNITS))
    assert res.status == "error:truncated"
    assert res.relevant_unit_ids is None


def test_openrouter_accepts_bare_list():
    j = _openrouter_with_response("[0, 2]")
    res = run(j.label("q", UNITS))
    assert res.status == "ok"
    assert res.relevant_unit_ids == [0, 2]


def test_openrouter_invalid_json():
    j = _openrouter_with_response("I think the answer is units 1 and 2")
    res = run(j.label("q", UNITS))
    assert res.status == "error:invalid_json"


def test_openrouter_wrong_shape():
    j = _openrouter_with_response('"just a string"')
    res = run(j.label("q", UNITS))
    assert res.status == "error:schema_shape"


# ---------------------------------------------------------------------------
# Bedrock transport parsing (fake SDK client)
# ---------------------------------------------------------------------------


def _bedrock_with_content(content_blocks):
    j = bare(AsyncBedrockClaudeLabeler)
    resp = SimpleNamespace(
        content=content_blocks,
        usage=SimpleNamespace(input_tokens=33, output_tokens=44),
    )

    async def create(**kwargs):
        # The tool must be forced and derived from the prompt registry.
        assert kwargs["tool_choice"]["name"] == j.prompt.tool_name
        assert kwargs["tools"][0]["input_schema"] == j.prompt.schema["schema"]
        return resp

    j.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return j


def test_bedrock_tool_use_parsed():
    j = _bedrock_with_content(
        [
            SimpleNamespace(type="text", text="thinking..."),
            SimpleNamespace(type="tool_use", input={"relevant_unit_ids": [1], "anchor_unit_id": 0}),
        ]
    )
    res = run(j.label("q", UNITS))
    assert res.status == "ok"
    assert res.relevant_unit_ids == [0, 1]
    assert (res.tokens_in, res.tokens_out) == (33, 44)


def test_bedrock_missing_tool_use():
    j = _bedrock_with_content([SimpleNamespace(type="text", text="no tool call")])
    res = run(j.label("q", UNITS))
    assert res.status == "error:no_tool_use"
    assert res.relevant_unit_ids is None
