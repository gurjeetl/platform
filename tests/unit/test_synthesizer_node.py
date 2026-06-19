"""SynthesizerNode: clarification, single-view passthrough, multi-task merge."""

import pytest

from genie.application.dag import Plan, Subtask
from genie.application.nodes.synthesizer import _CLARIFICATION, SynthesizerNode
from genie.application.state import GraphState, Message
from tests.conftest import StubLLM


@pytest.mark.asyncio
async def test_empty_plan_returns_clarification(settings):
    state = GraphState(
        conversation_id="c", plan={"subtasks": []}, messages=[Message(role="user", content="??")]
    )
    out = await SynthesizerNode(llm_provider=StubLLM(), settings=settings)(state)
    # Domain-agnostic fallback — no app-specific domains baked into the platform.
    assert out["final_response"] == _CLARIFICATION
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


class _StubMemory:
    """Memory whose writeback records the call and returns canned trace db_ops."""

    def __init__(self):
        self.calls = 0

    async def writeback(self, state, blackboard, text):
        self.calls += 1
        return [{"store": "mongodb", "op": "write", "node": "synthesizer", "detail": "commit t1"}]


@pytest.mark.asyncio
async def test_writeback_db_ops_surface_on_node_output(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather")])
    state = GraphState(
        conversation_id="c",
        plan=plan.model_dump(),
        messages=[Message(role="user", content="weather")],
        blackboard={"t1": {"agent_id": "weather", "text": "20C", "view": {"temp_c": 20}}},
    )
    memory = _StubMemory()
    out = await SynthesizerNode(llm_provider=StubLLM(), settings=settings, memory=memory)(state)
    assert memory.calls == 1
    assert out["db_ops"] == [
        {"store": "mongodb", "op": "write", "node": "synthesizer", "detail": "commit t1"}
    ]


@pytest.mark.asyncio
async def test_partial_answer_skips_writeback(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather")])
    state = GraphState(
        conversation_id="c",
        plan=plan.model_dump(),
        partial=True,
        messages=[Message(role="user", content="weather")],
        blackboard={"t1": {"agent_id": "weather", "text": "20C", "view": {"temp_c": 20}}},
    )
    memory = _StubMemory()
    out = await SynthesizerNode(llm_provider=StubLLM(), settings=settings, memory=memory)(state)
    assert memory.calls == 0
    assert "db_ops" not in out


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
