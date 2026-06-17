"""CompletionGateNode: replan vs synthesize."""

import pytest

from genie.application.dag import Plan, Subtask
from genie.application.nodes.completion_gate import CompletionGateNode
from genie.application.state import GraphState


def _state(plan: Plan, blackboard: dict, replan_count: int = 0) -> GraphState:
    return GraphState(
        conversation_id="c",
        plan=plan.model_dump(),
        blackboard=blackboard,
        replan_count=replan_count,
    )


@pytest.mark.asyncio
async def test_all_present_synthesizes(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather")])
    out = await CompletionGateNode(settings=settings)(
        _state(plan, {"t1": {"agent_id": "weather", "text": "ok"}})
    )
    assert out["metadata"]["gate_action"] == "synthesize"
    assert out["partial"] is False


@pytest.mark.asyncio
async def test_errored_task_triggers_replan(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather")])
    out = await CompletionGateNode(settings=settings)(_state(plan, {"t1": {"error": "boom"}}))
    assert out["metadata"]["gate_action"] == "replan"
    assert out["replan_count"] == 1
    assert out["partial"] is True


@pytest.mark.asyncio
async def test_replan_budget_exhausted_synthesizes_partial(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather")])
    out = await CompletionGateNode(settings=settings)(
        _state(plan, {"t1": {"error": "boom"}}, replan_count=3)
    )
    assert out["metadata"]["gate_action"] == "synthesize"
    assert out["partial"] is True


@pytest.mark.asyncio
async def test_empty_plan_synthesizes(settings):
    out = await CompletionGateNode(settings=settings)(_state(Plan(subtasks=[]), {}))
    assert out["metadata"]["gate_action"] == "synthesize"
