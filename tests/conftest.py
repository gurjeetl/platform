"""Shared test fixtures for the distributed-agent platform.

Agents are remote services in production; in tests we use an in-process ``FakeAgent``
(satisfying the ``BaseAgent`` protocol) and a ``StubLLM`` returning canned content so
the pipeline can be exercised deterministically without a network or a real model.
"""

from __future__ import annotations

from typing import Any, Callable

import pytest

from genie.agents import AgentRegistry
from genie.agents.base import AgentInfo, AgentResult, AgentTask, CapabilitySpec
from genie.application.state import GraphState, Message
from genie.llm.mock import MockLLMProvider
from genie.platform.config import Settings
from genie.platform.event_bus import EventBus


class _LLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.model = "stub"
        self.prompt_tokens = 0
        self.completion_tokens = 0


class StubLLM:
    """Async LLM stub. ``responder`` maps (system_prompt, user_prompt) → content.

    Defaults to echoing a generic answer; pass ``responder`` to return canned JSON
    for the router/planner or prose for the synthesizer.
    """

    def __init__(self, responder: Callable[[str, str], str] | None = None) -> None:
        self._responder = responder
        self.calls: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "stub"

    async def complete(self, messages: list[Message], **kwargs: Any) -> _LLMResponse:
        self.calls.append(list(messages))
        system = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        text = self._responder(system, user) if self._responder else "stub answer"
        return _LLMResponse(text)


class FakeAgent:
    """Minimal in-process agent satisfying ``genie.agents.base.BaseAgent``."""

    def __init__(
        self,
        agent_id: str,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        handler: Callable[[AgentTask, dict], AgentResult] | None = None,
        routing_keywords: list[str] | None = None,
        sla_ms: int = 10000,
    ) -> None:
        self._id = agent_id
        self._description = description
        self._input_schema = input_schema or {}
        self._output_schema = output_schema or {}
        self._tags = tags or []
        self._handler = handler
        self._routing_keywords = routing_keywords or []
        self._sla_ms = sla_ms
        self._enabled = True

    @property
    def agent_id(self) -> str:
        return self._id

    @property
    def name(self) -> str:
        return self._id

    @property
    def description(self) -> str:
        return self._description

    @property
    def capabilities(self) -> list[str]:
        return [self._id]

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> AgentResult:
        if self._handler is not None:
            return self._handler(task, context)
        args = (task.context or {}).get("args", {})
        return AgentResult(
            task_id=task.task_id,
            agent_id=self._id,
            success=True,
            output=f"{self._id} ran with {args}",
        )

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id=self._id,
            name=self._id,
            description=self._description,
            version="1.0.0",
            enabled=self._enabled,
            capability_specs=[CapabilitySpec(id=self._id, routing_keywords=self._routing_keywords)],
            input_schema=self._input_schema,
            output_schema=self._output_schema,
            tags=self._tags,
            sla_ms=self._sla_ms,
        )

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    async def health_check(self) -> str:
        return "healthy"


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        debug=True,
        rag_mode="local",
        enable_hitl=False,
        hitl_auto_approve=True,
        enable_tracking=False,
        enable_rag=False,
        enable_guards=False,
        agent_mode="local",
    )


@pytest.fixture()
def mock_llm() -> MockLLMProvider:
    return MockLLMProvider()


@pytest.fixture()
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture()
def agent_registry() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(
        FakeAgent(
            "weather",
            description="Weather report for a city",
            input_schema={"location": {"type": "string", "required": True}},
            tags=["weather", "forecast"],
            handler=lambda task, ctx: AgentResult(
                task_id=task.task_id,
                agent_id="weather",
                success=True,
                output=f"Weather in {ctx.get('args', {}).get('location', '?')}: 20C",
                data={"view": {"temp_c": 20}},
            ),
        )
    )
    registry.register(
        FakeAgent(
            "outage",
            description="Grid outage list / detail",
            input_schema={"outage_id": {"type": "integer", "required": False}},
            tags=["outage", "grid"],
            handler=lambda task, ctx: AgentResult(
                task_id=task.task_id,
                agent_id="outage",
                success=True,
                output="Top outages: #1, #2",
                data={"view": {"items": [{"id": 1}, {"id": 2}]}},
            ),
        )
    )
    return registry


@pytest.fixture()
def base_state() -> GraphState:
    return GraphState(
        conversation_id="test-conv",
        user_id="test-user",
        messages=[Message(role="user", content="Hello")],
    )
