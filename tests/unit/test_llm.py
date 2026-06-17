"""Unit tests for MockLLMProvider and LLMRegistry."""

import pytest
from genie.application.state import Message
from genie.llm.mock import MockLLMProvider
from genie.llm.registry import LLMRegistry
from genie.platform.errors import GenieError


async def test_mock_llm_returns_response() -> None:
    llm = MockLLMProvider()
    resp = await llm.complete([Message(role="user", content="hello")])
    assert resp.content != ""
    assert resp.model == "mock-model"
    assert resp.prompt_tokens > 0


async def test_mock_llm_stream() -> None:
    llm = MockLLMProvider()
    tokens: list[str] = []
    async for token in llm.stream([Message(role="user", content="hi")]):
        tokens.append(token)
    assert "".join(tokens) != ""


async def test_mock_llm_classifies_meter_query() -> None:
    llm = MockLLMProvider()
    prompt = (
        "Classify the following user message into exactly one of: "
        "rag_query, agent_task, domain_query, general_chat. "
        "Return only the category.\n\nMessage: check meter availability ercot market"
    )
    resp = await llm.complete([Message(role="user", content=prompt)], temperature=0.0)
    assert resp.content.strip().lower() == "agent_task"


def test_llm_registry_get_registered() -> None:
    registry = LLMRegistry()
    mock = MockLLMProvider()
    registry.register("mock", mock)
    assert registry.get("mock") is mock


def test_llm_registry_raises_for_missing() -> None:
    registry = LLMRegistry()
    with pytest.raises(GenieError):
        registry.get("unknown")
