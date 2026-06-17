"""OpenAI-compatible LLM provider — works with any vLLM / Ollama / custom endpoint."""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from genie.application.state import Message
from genie.llm.base import LLMProvider, LLMResponse
from genie.observability.logging import get_logger

logger = get_logger(__name__)


class OpenAICompatibleLLMProvider:
    """LLM provider that calls any OpenAI-compatible chat-completions endpoint.

    Suitable for on-premise deployments (vLLM, Ollama, LM Studio) or any server
    that implements POST /{prompting_path}/chat/completions.

    Falls back to an empty-string response if the server is unreachable so the
    pipeline degrades gracefully in development without a running LLM server.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "not-required",
        provider_name: str = "openai_compat",
        max_token_limit: int = 4096,
    ) -> None:
        from openai import AsyncOpenAI

        self._model = model_name
        self._provider_name = provider_name
        self._max_token_limit = max_token_limit
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        logger.info(
            "llm_provider_initialized",
            provider=provider_name,
            base_url=base_url,
            model=model_name,
        )

    @property
    def name(self) -> str:
        return self._provider_name

    def _to_openai_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        capped = min(max_tokens, self._max_token_limit)
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=self._to_openai_messages(messages),
                max_tokens=capped,
                temperature=temperature,
            )
            raw_content = response.choices[0].message.content
            if not raw_content:
                # Some vLLM builds return None when finish_reason is "stop" with
                # no tokens generated (e.g. tool_call fallback, empty generation).
                logger.warning(
                    "llm_empty_content",
                    provider=self._provider_name,
                    model=self._model,
                    finish_reason=getattr(response.choices[0], "finish_reason", "unknown"),
                )
            content = raw_content or ""
            usage = response.usage
            return LLMResponse(
                content=content,
                model=self._model,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as exc:
            logger.warning(
                "llm_complete_failed",
                provider=self._provider_name,
                error=str(exc),
            )
            return LLMResponse(
                content="[LLM unavailable — response could not be generated]",
                model=self._model,
            )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        capped = min(max_tokens, self._max_token_limit)
        try:
            async with self._client.chat.completions.stream(
                model=self._model,
                messages=self._to_openai_messages(messages),
                max_tokens=capped,
                temperature=temperature,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta
        except Exception as exc:
            logger.warning(
                "llm_stream_failed",
                provider=self._provider_name,
                error=str(exc),
            )
            yield "[LLM unavailable]"
