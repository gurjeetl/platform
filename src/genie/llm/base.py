"""LLM provider protocol and response types."""
from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel

from genie.application.state import Message


class LLMResponse(BaseModel):
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]: ...
