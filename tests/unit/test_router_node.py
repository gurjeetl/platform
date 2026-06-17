"""RouterNode triage: fast / chitchat / plan."""

import pytest

from genie.application.nodes.router import RouterNode
from genie.application.state import GraphState, Message
from tests.conftest import StubLLM


def _state(text: str) -> GraphState:
    return GraphState(conversation_id="c", messages=[Message(role="user", content=text)])


@pytest.mark.asyncio
async def test_fast_route_builds_one_task_plan(settings, agent_registry):
    llm = StubLLM(
        lambda s, u: (
            '{"route":"fast","agent_id":"weather","args":{"location":"paris"},"confidence":0.95}'
        )
    )
    node = RouterNode(llm_provider=llm, agent_registry=agent_registry, settings=settings)
    out = await node(_state("weather in paris"))
    assert out["route"] == "fast"
    assert out["plan"]["subtasks"][0]["agent_id"] == "weather"
    assert out["waves"] == [["t1"]]


@pytest.mark.asyncio
async def test_chitchat_route_empty_plan(settings, agent_registry):
    llm = StubLLM(lambda s, u: '{"route":"chitchat","agent_id":null,"args":{},"confidence":0.0}')
    node = RouterNode(llm_provider=llm, agent_registry=agent_registry, settings=settings)
    out = await node(_state("hello"))
    assert out["route"] == "chitchat"
    assert out["plan"] == {"subtasks": []}


@pytest.mark.asyncio
async def test_multi_intent_regex_forces_plan_without_llm(settings, agent_registry):
    llm = StubLLM(lambda s, u: pytest.fail("LLM should not be called for clear multi-intent"))
    node = RouterNode(llm_provider=llm, agent_registry=agent_registry, settings=settings)
    out = await node(_state("weather in tokyo; also the top outages"))
    assert out["route"] == "plan"


@pytest.mark.asyncio
async def test_low_confidence_fast_downgrades_to_plan(settings, agent_registry):
    llm = StubLLM(
        lambda s, u: (
            '{"route":"fast","agent_id":"weather","args":{"location":"x"},"confidence":0.2}'
        )
    )
    node = RouterNode(llm_provider=llm, agent_registry=agent_registry, settings=settings)
    out = await node(_state("weather"))
    assert out["route"] == "plan"
