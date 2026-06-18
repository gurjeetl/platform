"""enable_router toggles whether the router step runs in the compiled graph."""

import pytest

from genie.application.graph import build_graph
from genie.application.state import GraphState, Message
from tests.conftest import StubLLM

_PLAN_ONE = (
    '{"subtasks":[{"id":"t1","agent_id":"weather","args":{"location":"paris"},"depends_on":[]}]}'
)


def _state() -> dict:
    return GraphState(
        conversation_id="c", messages=[Message(role="user", content="weather in paris")]
    ).model_dump()


@pytest.mark.asyncio
async def test_router_disabled_skips_router_llm(settings, agent_registry):
    # The router prompt (system text contains "ROUTER") must never be sent.
    def responder(system: str, user: str) -> str:
        assert "ROUTER" not in system, "router LLM was called even though enable_router=False"
        if "planning agent" in system:
            return _PLAN_ONE
        return "passthrough"

    s = settings.model_copy(update={"enable_router": False})
    graph, _ = build_graph(
        llm_provider=StubLLM(responder), agent_registry=agent_registry, settings=s
    )
    out = await graph.ainvoke(_state(), config={"configurable": {"thread_id": "c"}})
    # planner ran directly → weather agent → single-view passthrough
    assert "paris" in (out["final_response"] or "").lower()


@pytest.mark.asyncio
async def test_router_enabled_consults_router(settings, agent_registry):
    seen = {"router": False}

    def responder(system: str, user: str) -> str:
        if "ROUTER" in system:
            seen["router"] = True
            return '{"route":"plan","agent_id":null,"args":{},"confidence":0.0}'
        if "planning agent" in system:
            return _PLAN_ONE
        return "passthrough"

    graph, _ = build_graph(
        llm_provider=StubLLM(responder), agent_registry=agent_registry, settings=settings
    )
    await graph.ainvoke(_state(), config={"configurable": {"thread_id": "c2"}})
    assert seen["router"] is True
