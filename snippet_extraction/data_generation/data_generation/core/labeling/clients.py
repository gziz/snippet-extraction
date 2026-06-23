"""Async labeler clients: one transport per provider, one shared retry loop.

Every client is an :class:`AsyncLabelerBase` subclass that implements two
things:

- ``_call(user_prompt) -> _Call`` — one structured-output API call, returning
  the parsed JSON (or a terminal non-ok status such as ``error:invalid_json``).
- ``_classify_error(exc)`` — map provider exceptions to ``("retry", delay)``
  for rate limits, ``("fail", status)`` for terminal errors, or ``None`` to
  re-raise.

The base class owns timing, rate-limit backoff (honoring Retry-After when the
provider sends it), and parsing the response through the prompt's parser. All
clients expose the same coroutine ``labeler(query, units) -> LabelResult`` plus
``provider`` / ``deployment`` attributes that storage and dedupe rely on.

Schema enforcement per provider:

- Azure OpenAI — native strict ``json_schema`` response format.
- Bedrock Anthropic — tool-use forcing (a single tool whose ``input_schema``
  matches the prompt schema; the tool ``input`` is the JSON we want).
- Gemini — ``response_mime_type`` + ``response_schema`` (grammar-constrained
  decoding, same hardness as Azure's json_schema mode).
- OpenRouter — ``json_object`` mode + an explicit shape instruction in the
  system prompt (see the class docstring for why strict mode isn't used).

``make_async_labeler(provider, version)`` is the factory the pipelines use.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncAzureOpenAI, AsyncOpenAI, BadRequestError, RateLimitError

from ..paths import load_env
from .prompts import DEFAULT_VERSION, PROMPTS, LabelPrompt, build_user_prompt


@dataclass
class LabelResult:
    relevant_unit_ids: list[int] | None
    tokens_in: int
    tokens_out: int
    latency_s: float
    raw: str
    status: str  # "ok" | "content_filter" | "error:<code>"
    # Only set by prompts with a context task (v3): superset of relevant ids.
    context_unit_ids: list[int] | None = None


class AsyncLabelerClient(Protocol):
    """Common interface implemented by all async labeler clients."""

    provider: str
    deployment: str

    async def label(self, query: str, units: list[str]) -> LabelResult: ...


@dataclass
class _Call:
    """Outcome of one API call: parsed JSON, or a terminal non-ok status."""

    parsed: dict | None
    raw: str
    tokens_in: int
    tokens_out: int
    status: str = "ok"


def _retry_after_seconds(exc, header: str = "Retry-After") -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is None:
        return None
    try:
        return float(headers.get(header, "") or 0) or None
    except (TypeError, ValueError):
        return None


class AsyncLabelerBase:
    provider: str = "?"
    deployment: str = "?"
    MAX_RETRIES = 5

    def __init__(self, prompt: LabelPrompt) -> None:
        self.prompt = prompt

    async def _call(self, user_prompt: str) -> _Call:
        raise NotImplementedError

    def _classify_error(self, exc: Exception) -> tuple[str, object] | None:
        raise NotImplementedError

    async def label(self, query: str, units: list[str]) -> LabelResult:
        user_prompt = build_user_prompt(query, units)
        t0 = time.perf_counter()
        attempts = 0
        while True:
            try:
                call = await self._call(user_prompt)
                break
            except Exception as e:
                outcome = self._classify_error(e)
                if outcome is None:
                    raise
                kind, value = outcome
                if kind == "fail":
                    return LabelResult(
                        relevant_unit_ids=None,
                        tokens_in=0,
                        tokens_out=0,
                        latency_s=time.perf_counter() - t0,
                        raw=str(e)[:500],
                        status=str(value),
                    )
                attempts += 1
                if attempts > self.MAX_RETRIES:
                    return LabelResult(
                        relevant_unit_ids=None,
                        tokens_in=0,
                        tokens_out=0,
                        latency_s=time.perf_counter() - t0,
                        raw=f"rate_limit_giveup: {e}"[:500],
                        status="error:rate_limit",
                    )
                await asyncio.sleep(value if value else min(2**attempts, 30))

        dt = time.perf_counter() - t0
        if call.status != "ok" or call.parsed is None:
            return LabelResult(
                relevant_unit_ids=None,
                tokens_in=call.tokens_in,
                tokens_out=call.tokens_out,
                latency_s=dt,
                raw=call.raw,
                status=call.status,
            )
        ids = self.prompt.parse(call.parsed, len(units))
        return LabelResult(
            relevant_unit_ids=ids.relevant,
            tokens_in=call.tokens_in,
            tokens_out=call.tokens_out,
            latency_s=dt,
            raw=call.raw,
            status="ok",
            context_unit_ids=ids.context,
        )


class AsyncAzureLabeler(AsyncLabelerBase):
    """Azure OpenAI (gpt-5.4 etc.) with native strict JSON-schema output."""

    provider = "azure"

    def __init__(self, prompt: LabelPrompt | None = None) -> None:
        super().__init__(prompt or PROMPTS[DEFAULT_VERSION])
        load_env()
        self.client = AsyncAzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        self.deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]

    async def _call(self, user_prompt: str) -> _Call:
        resp = await self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": self.prompt.system},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_schema", "json_schema": self.prompt.schema},
        )
        raw = resp.choices[0].message.content or ""
        return _Call(
            parsed=json.loads(raw),
            raw=raw,
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
        )

    def _classify_error(self, exc: Exception):
        if isinstance(exc, RateLimitError):
            return ("retry", _retry_after_seconds(exc))
        if isinstance(exc, BadRequestError):
            code = getattr(exc, "code", None) or "bad_request"
            return ("fail", "content_filter" if code == "content_filter" else f"error:{code}")
        return None


class AsyncBedrockClaudeLabeler(AsyncLabelerBase):
    """AWS Bedrock + Anthropic Claude, schema enforced via tool-use forcing."""

    provider = "bedrock"

    def __init__(
        self,
        prompt: LabelPrompt | None = None,
        model_id: str | None = None,
        region: str | None = None,
    ) -> None:
        try:
            from anthropic import AsyncAnthropicBedrock
        except ImportError as e:
            raise ImportError(
                "The 'bedrock' provider needs the Anthropic SDK. "
                "Install it with: pip install 'data-generation[bedrock]'"
            ) from e

        super().__init__(prompt or PROMPTS[DEFAULT_VERSION])
        load_env()
        self.deployment = model_id or os.environ.get(
            "BEDROCK_MODEL_ID",
            "global.anthropic.claude-sonnet-4-6",
        )
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.client = AsyncAnthropicBedrock(aws_region=self.region)

    async def _call(self, user_prompt: str) -> _Call:
        resp = await self.client.messages.create(
            model=self.deployment,
            max_tokens=1024,
            temperature=0,
            system=self.prompt.system,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[
                {
                    "name": self.prompt.tool_name,
                    "description": self.prompt.tool_description,
                    "input_schema": self.prompt.schema["schema"],
                }
            ],
            tool_choice={"type": "tool", "name": self.prompt.tool_name},
        )
        tokens_in = getattr(resp.usage, "input_tokens", 0)
        tokens_out = getattr(resp.usage, "output_tokens", 0)
        tool_block = next((b for b in resp.content if getattr(b, "type", None) == "tool_use"), None)
        if tool_block is None or not isinstance(tool_block.input, dict):
            return _Call(None, str(resp)[:500], tokens_in, tokens_out, "error:no_tool_use")
        return _Call(tool_block.input, json.dumps(tool_block.input), tokens_in, tokens_out)

    def _classify_error(self, exc: Exception):
        from anthropic import APIStatusError
        from anthropic import RateLimitError as AnthropicRateLimitError

        if isinstance(exc, AnthropicRateLimitError):
            return ("retry", _retry_after_seconds(exc, "retry-after"))
        if isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            msg = str(exc).lower()
            if status_code == 400 and ("guardrail" in msg or "blocked" in msg):
                return ("fail", "content_filter")
            return ("fail", f"error:{status_code or 'api'}")
        return None


class AsyncGeminiLabeler(AsyncLabelerBase):
    """Google Gemini via the official google-genai SDK (grammar-constrained JSON)."""

    provider = "gemini"

    def __init__(self, prompt: LabelPrompt | None = None, model_id: str | None = None) -> None:
        try:
            from google import genai
        except ImportError as e:
            raise ImportError(
                "The 'gemini' provider needs the Google GenAI SDK. "
                "Install it with: pip install 'data-generation[gemini]'"
            ) from e

        super().__init__(prompt or PROMPTS[DEFAULT_VERSION])
        load_env()
        self.deployment = model_id or os.environ.get("GEMINI_MODEL_ID", "gemini-3.5-flash")
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    async def _call(self, user_prompt: str) -> _Call:
        from google.genai import types as genai_types

        resp = await self.client.aio.models.generate_content(
            model=self.deployment,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.prompt.system,
                response_mime_type="application/json",
                response_schema=self.prompt.gemini_schema,
            ),
        )
        usage = resp.usage_metadata
        tokens_in = getattr(usage, "prompt_token_count", 0) or 0
        tokens_out = getattr(usage, "candidates_token_count", 0) or 0
        raw = resp.text or ""
        # A 200 with no .text is a post-hoc safety block (rare).
        if not raw:
            return _Call(None, str(resp)[:500], tokens_in, tokens_out, "content_filter")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return _Call(None, raw[:500], tokens_in, tokens_out, "error:invalid_json")
        return _Call(parsed, raw, tokens_in, tokens_out)

    def _classify_error(self, exc: Exception):
        from google.genai import errors as genai_errors

        if not isinstance(exc, genai_errors.APIError):
            return None
        code = getattr(exc, "code", None)
        status_str = (getattr(exc, "status", "") or "").upper()
        msg = (getattr(exc, "message", "") or str(exc)).lower()
        if code == 429 or "RESOURCE_EXHAUSTED" in status_str:
            return ("retry", None)
        if code == 400 and ("safety" in msg or "policy" in msg or "blocked" in msg):
            return ("fail", "content_filter")
        return ("fail", f"error:{code or 'api'}")


class AsyncOpenRouterLabeler(AsyncLabelerBase):
    """OpenRouter using the OpenAI-compatible API.

    Defaults to DeepSeek V4 Pro, which is a *reasoning* model: it emits a long
    chain-of-thought before the answer, so the call must (a) leave room in the
    completion budget for that trace plus a potentially long ID list, and
    (b) avoid backends that gut quality or the reasoning trace.

    Routing is left unconstrained: OpenRouter load-balances across all
    endpoints that serve the model. OpenRouter only ranks providers by
    price/throughput/latency — never answer quality — and pinning a single
    provider (e.g. ``order=['deepseek']``) failed with "no endpoints found" for
    some V4 variants, so we let the router pick whichever endpoint is available.

    Schema enforcement: not all endpoints support strict ``json_schema``
    (grammar-constrained structured outputs); forcing it filters most endpoints
    out. So we use ``response_format`` JSON *object* mode — broadly supported —
    and pin the output shape via the prompt's ``json_shape_instruction``. The
    parser tolerates both a bare list and the ``{"relevant_unit_ids": [...]}``
    object, then validates ids against range.
    """

    provider = "openrouter"

    # Reasoning trace + (possibly 100+ element) ID list must both fit here.
    # Without an explicit budget, providers apply small defaults and the answer
    # array gets truncated to [] on large documents.
    MAX_TOKENS = 16000

    def __init__(self, prompt: LabelPrompt | None = None, model_id: str | None = None) -> None:
        super().__init__(prompt or PROMPTS[DEFAULT_VERSION])
        load_env()
        self.deployment = model_id or os.environ.get(
            "OPENROUTER_MODEL_ID",
            "deepseek/deepseek-v4-pro",
        )
        api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ["OPEN_ROUTER"]
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    async def _call(self, user_prompt: str) -> _Call:
        resp = await self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": self.prompt.system + "\n\n" + self.prompt.json_shape_instruction,
                },
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.MAX_TOKENS,
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        tokens_in = getattr(resp.usage, "prompt_tokens", 0) or 0
        tokens_out = getattr(resp.usage, "completion_tokens", 0) or 0
        # A reasoning model that runs out of completion budget gets cut off with
        # finish_reason="length"; the grammar then closes the array early, so a
        # truncated answer looks like a legitimate []. Flag it instead of
        # silently writing a wrong empty label.
        if getattr(resp.choices[0], "finish_reason", None) == "length":
            return _Call(None, raw[:500], tokens_in, tokens_out, "error:truncated")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return _Call(None, raw[:500], tokens_in, tokens_out, "error:invalid_json")
        # Some sub-providers don't strictly honor the shape and return a bare
        # list instead of {"relevant_unit_ids": [...]}; accept both.
        if isinstance(parsed, list):
            parsed = {"relevant_unit_ids": parsed}
        if not isinstance(parsed, dict):
            return _Call(None, raw[:500], tokens_in, tokens_out, "error:schema_shape")
        return _Call(parsed, raw, tokens_in, tokens_out)

    def _classify_error(self, exc: Exception):
        if isinstance(exc, RateLimitError):
            return ("retry", _retry_after_seconds(exc))
        if isinstance(exc, BadRequestError):
            code = getattr(exc, "code", None) or "bad_request"
            return ("fail", "content_filter" if code == "content_filter" else f"error:{code}")
        return None


_CLIENTS = {
    "azure": AsyncAzureLabeler,
    "bedrock": AsyncBedrockClaudeLabeler,
    "gemini": AsyncGeminiLabeler,
    "openrouter": AsyncOpenRouterLabeler,
}


def make_async_labeler(provider: str, version: str = DEFAULT_VERSION) -> AsyncLabelerClient:
    """Factory used by the pipelines to pick a client by --provider."""
    p = provider.lower()
    if p not in _CLIENTS:
        raise ValueError(f"unknown provider: {provider!r} (expected one of {sorted(_CLIENTS)})")
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version: {version!r} (expected one of {sorted(PROMPTS)})")
    return _CLIENTS[p](prompt=PROMPTS[version])
