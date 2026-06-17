"""PlannerNode DAG construction + validation."""

import pytest

from genie.application.nodes.planner import PlannerNode
from genie.application.state import GraphState, Message
from tests.conftest import StubLLM


def _state(text: str) -> GraphState:
    return GraphState(conversation_id="c", messages=[Message(role="user", content=text)])


@pytest.mark.asyncio
async def test_builds_multi_task_plan(settings, agent_registry):
    plan_json = (
        '{"subtasks":['
        '{"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},'
        '{"id":"t2","agent_id":"outage","args":{},"depends_on":[]}]}'
    )
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
    )
    out = await node(_state("weather in tokyo and top outages"))
    ids = [t["agent_id"] for t in out["plan"]["subtasks"]]
    assert ids == ["weather", "outage"]
    assert out["agent_versions"] == {"t1": "1.0.0", "t2": "1.0.0"}


@pytest.mark.asyncio
async def test_drops_unknown_agent_and_invalid_args(settings, agent_registry):
    plan_json = (
        '{"subtasks":['
        '{"id":"t1","agent_id":"ghost","args":{},"depends_on":[]},'
        '{"id":"t2","agent_id":"weather","args":{},"depends_on":[]}]}'  # missing required location
    )
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
    )
    out = await node(_state("bad plan"))
    assert out["plan"]["subtasks"] == []


@pytest.mark.asyncio
async def test_unparseable_response_yields_error(settings, agent_registry):
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: "not json"),
    )
    out = await node(_state("hi"))
    assert out["plan"]["subtasks"] == []
    assert "error" in out
