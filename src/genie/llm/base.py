"""LLM provider protocol and response types."""

from __future__ import annotations

from typing import Any, AsyncIterator, Protocol, runtime_checkable

from pydantic import BaseModel

from genie.application.state import Message


class LLMResponse(BaseModel):
    """A completed LLM generation plus token-usage accounting."""

    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@runtime_checkable
class LLMProvider(Protocol):
    """Structural protocol every LLM backend must satisfy (complete + stream)."""

    @property
    def name(self) -> str:
        """Provider identifier used for registry lookup and logging."""
        ...

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a single completion for *messages* and return it whole."""
        ...

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield the completion incrementally as text chunks."""
        ...
