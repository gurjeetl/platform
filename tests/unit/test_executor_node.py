"""ExecutorNode wave execution, blackboard, ref resolution, error capture."""

import pytest

from genie.agents import AgentRegistry
from genie.agents.base import AgentResult
from genie.application.dag import Plan, Subtask
from genie.application.nodes.executor import ExecutorNode
from genie.application.state import GraphState, Message
from tests.conftest import FakeAgent


def _state(plan: Plan, waves: list[list[str]]) -> GraphState:
    return GraphState(
        conversation_id="c",
        run_id="r",
        messages=[Message(role="user", content="go")],
        plan=plan.model_dump(),
        waves=waves,
    )


@pytest.mark.asyncio
async def test_runs_tasks_and_writes_blackboard(settings, agent_registry):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="weather", args={"location": "paris"})])
    node = ExecutorNode(agent_registry=agent_registry, settings=settings)
    out = await node(_state(plan, [["t1"]]))
    assert "paris" in out["blackboard"]["t1"]["text"].lower()
    assert out["blackboard"]["t1"]["view"] == {"temp_c": 20}


@pytest.mark.asyncio
async def test_missing_agent_writes_error(settings):
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="ghost")])
    node = ExecutorNode(agent_registry=AgentRegistry(), settings=settings)
    out = await node(_state(plan, [["t1"]]))
    assert "error" in out["blackboard"]["t1"]


@pytest.mark.asyncio
async def test_failed_agent_captured_as_error(settings):
    reg = AgentRegistry()
    reg.register(
        FakeAgent(
            "boom",
            handler=lambda t, c: AgentResult(
                task_id=t.task_id, agent_id="boom", success=False, output="", error="kaboom"
            ),
        )
    )
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="boom")])
    node = ExecutorNode(agent_registry=reg, settings=settings)
    out = await node(_state(plan, [["t1"]]))
    assert out["blackboard"]["t1"]["error"] == "kaboom"


@pytest.mark.asyncio
async def test_resolves_upstream_reference(settings):
    reg = AgentRegistry()
    reg.register(
        FakeAgent(
            "lister",
            handler=lambda t, c: AgentResult(
                task_id=t.task_id,
                agent_id="lister",
                success=True,
                output="list",
                data={"view": {"items": [{"id": 42}]}},
            ),
        )
    )
    captured = {}

    def detail_handler(task, ctx):
        captured["outage_id"] = ctx.get("args", {}).get("outage_id")
        return AgentResult(task_id=task.task_id, agent_id="detail", success=True, output="detail")

    reg.register(FakeAgent("detail", handler=detail_handler))
    plan = Plan(
        subtasks=[
            Subtask(id="t1", agent_id="lister"),
            Subtask(
                id="t2",
                agent_id="detail",
                args={"outage_id": "${t1.view.items.0.id}"},
                depends_on=["t1"],
            ),
        ]
    )
    node = ExecutorNode(agent_registry=reg, settings=settings)
    await node(_state(plan, [["t1"], ["t2"]]))
    assert captured["outage_id"] == 42
