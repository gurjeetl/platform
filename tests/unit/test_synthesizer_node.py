"""SynthesizerNode: clarification, single-view passthrough, multi-task merge."""

import pytest

from genie.application.dag import Plan, Subtask
from genie.application.nodes.synthesizer import SynthesizerNode
from genie.application.state import GraphState, Message
from tests.conftest import StubLLM


@pytest.mark.asyncio
async def test_empty_plan_returns_clarification(settings):
    state = GraphState(
        conversation_id="c", plan={"subtasks": []}, messages=[Message(role="user", content="??")]
    )
    out = await SynthesizerNode(llm_provider=StubLLM(), settings=settings)(state)
    assert "weather" in out["final_response"].lower()
    assert out["is_complete"] is True


@pytest.mark.asyncio
async def test_single_view_passthrough_skips_llm(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather")])
    state = GraphState(
        conversation_id="c",
        plan=plan.model_dump(),
        messages=[Message(role="user", content="weather")],
        blackboard={"t1": {"agent_id": "weather", "text": "20C", "view": {"temp_c": 20}}},
    )
    llm = StubLLM(lambda s, u: pytest.fail("LLM should not be called for single-view passthrough"))
    out = await SynthesizerNode(llm_provider=llm, settings=settings)(state)
    assert out["final_response"] == "20C"
    assert out["view"] == {"temp_c": 20}


@pytest.mark.asyncio
async def test_multi_task_merges_via_llm(settings):
    plan = Plan(
        subtasks=[Subtask(id="t1", agent_id="weather"), Subtask(id="t2", agent_id="outage")]
    )
    state = GraphState(
        conversation_id="c",
        plan=plan.model_dump(),
        messages=[Message(role="user", content="both")],
        blackboard={
            "t1": {"agent_id": "weather", "text": "sunny"},
            "t2": {"agent_id": "outage", "text": "two outages"},
        },
    )
    out = await SynthesizerNode(
        llm_provider=StubLLM(lambda s, u: "Merged answer."), settings=settings
    )(state)
    assert out["final_response"] == "Merged answer."
