"""Deterministic mock LLM provider for zero-dependency local development and testing."""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from genie.application.state import Message
from genie.llm.base import LLMProvider, LLMResponse


class MockLLMProvider:
    """Deterministic mock LLM for zero-dependency local development and testing."""

    def __init__(self, model: str = "mock-model", response_prefix: str = "") -> None:
        # response_prefix is prepended to every canned answer (test fixtures use it).
        self._model = model
        self._response_prefix = response_prefix

    @property
    def name(self) -> str:
        """Always ``"mock"``."""
        return "mock"

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Return a deterministic canned response with approximate token counts."""
        response_text = self._generate_response(messages)
        return LLMResponse(
            content=response_text,
            model=self._model,
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=len(response_text.split()),
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream the canned response word-by-word, yielding control between words."""
        response = await self.complete(messages, max_tokens=max_tokens, temperature=temperature)
        words = response.content.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0)  # yield control

    def _generate_response(self, messages: list[Message]) -> str:
        """Pick a canned reply by keyword-matching the last message.

        Handles router classification prompts specially (returns an intent label)
        and otherwise falls back to domain-flavoured stock answers.
        """
        last_content = messages[-1].content.lower() if messages else ""
        prefix = self._response_prefix

        # Classification responses (for router node).
        # Extract only the actual user message from "Message: <text>" at end of
        # the classification prompt so template words ("document", "retrieve")
        # don't pollute keyword matching.
        if "classify" in last_content and "category" in last_content:
            if "\nmessage: " in last_content:
                user_msg = last_content.split("\nmessage: ", 1)[-1].strip()
            else:
                user_msg = last_content
            if any(kw in user_msg for kw in ["meter", "availability", "market"]):
                return "agent_task"
            if any(kw in user_msg for kw in ["deal", "validate", "rules"]):
                return "agent_task"
            if any(
                kw in user_msg
                for kw in [
                    "conductor",
                    "acsr",
                    "aac",
                    "pylon",
                    "pylons",
                    "span",
                    "spans",
                    "tower",
                    "towers",
                ]
            ):
                return "agent_task"
            if any(
                kw in user_msg for kw in ["document", "search", "nerc", "standard", "specification"]
            ):
                return "rag_query"
            return "general_chat"

        # Domain-specific responses
        if any(kw in last_content for kw in ["meter", "availability", "ercot", "market"]):
            return (
                f"{prefix}Meter data availability analysis: ERCOT market shows 98.5% availability "
                "for the requested period. All metering systems operational."
            )
        if any(kw in last_content for kw in ["deal", "validate", "san jose"]):
            return (
                f"{prefix}Deal validation complete: 3 deals validated for San Jose, May 1-15. "
                "2 passed weather-based rules, 1 flagged for temperature threshold review."
            )
        if any(kw in last_content for kw in ["weather"]):
            return (
                f"{prefix}Weather data retrieved. Current conditions: temperature 72°F, "
                "wind speed 8 mph, no precipitation."
            )
        if any(kw in last_content for kw in ["hello", "hi ", "greet"]):
            return (
                f"{prefix}Hello! I'm Genie, your AI platform assistant. How can I help you today?"
            )

        return (
            f"{prefix}I've processed your request. Based on the available information and context, "
            "here is my analysis and response to your query."
        )
