"""Thin async wrapper over the OpenAI client with metrics + retries."""
from __future__ import annotations

import json
import time
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.core.logging import get_logger
from app.core.metrics import LLM_LATENCY, LLM_TOKENS

log = get_logger("openai")


class LLMClient:
    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        s = get_settings()
        self._settings = s
        if client is not None:
            self._client = client
        else:
            kwargs: dict[str, Any] = {
                "timeout": s.llm_timeout_seconds,
                "max_retries": 0,
            }
            # Any OpenAI-shape endpoint (Groq, Together, Ollama, vLLM, …):
            # set LLM_BASE_URL and the SDK targets that host instead of OpenAI.
            if s.llm_base_url:
                kwargs["base_url"] = s.llm_base_url
                kwargs["api_key"] = s.llm_api_key or "sk-local-noop"
            else:
                kwargs["api_key"] = s.llm_api_key
            self._client = AsyncOpenAI(**kwargs)

    async def chat(
        self,
        *,
        purpose: str,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float = 0.2,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = 800,
    ) -> tuple[str, dict[str, int]]:
        s = self._settings
        model = model or s.llm_model_chat
        start = time.perf_counter()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(s.llm_max_retries + 1),
            # max bumped to 20s so a Groq 429 (TPM resets each minute) gets
            # backed off long enough to clear on the next attempt.
            wait=wait_exponential(min=0.5, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if response_format is not None:
                    kwargs["response_format"] = response_format
                resp = await self._client.chat.completions.create(**kwargs)

        text = resp.choices[0].message.content or ""
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }
        LLM_TOKENS.labels(model=model, kind="prompt").inc(usage["prompt_tokens"])
        LLM_TOKENS.labels(model=model, kind="completion").inc(usage["completion_tokens"])
        LLM_LATENCY.labels(model=model, purpose=purpose).observe(time.perf_counter() - start)
        return text, usage

    async def chat_with_tools(
        self,
        *,
        purpose: str,
        messages: list[ChatCompletionMessageParam],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = 600,
    ) -> tuple[Any, dict[str, int]]:
        """Chat-completions call with function/tool calling enabled.

        Returns the raw assistant *message* (so the caller can read both
        ``.content`` and ``.tool_calls``) plus token usage. Same retry +
        metrics wrapper as :meth:`chat`. Used by the MCP agent loop.
        """
        s = self._settings
        model = model or s.llm_model_chat
        start = time.perf_counter()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(s.llm_max_retries + 1),
            wait=wait_exponential(min=0.5, max=20),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt:
                resp = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                )

        message = resp.choices[0].message
        usage = {
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(resp.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(resp.usage, "total_tokens", 0) or 0,
        }
        LLM_TOKENS.labels(model=model, kind="prompt").inc(usage["prompt_tokens"])
        LLM_TOKENS.labels(model=model, kind="completion").inc(usage["completion_tokens"])
        LLM_LATENCY.labels(model=model, purpose=purpose).observe(time.perf_counter() - start)
        return message, usage

    async def json_chat(
        self,
        *,
        purpose: str,
        messages: list[ChatCompletionMessageParam],
        model: str | None = None,
        temperature: float = 0.0,
    ) -> tuple[dict[str, Any], dict[str, int]]:
        text, usage = await self.chat(
            purpose=purpose,
            messages=messages,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        try:
            return json.loads(text), usage
        except json.JSONDecodeError as exc:
            log.warning("json_parse_failed", purpose=purpose, raw=text[:300])
            raise exc
