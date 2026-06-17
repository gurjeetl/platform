"""DAG wave decomposition (Kahn's algorithm)."""

import pytest

from genie.application.dag import DAGCycleError, Plan, Subtask


def test_independent_tasks_single_wave():
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="a"), Subtask(id="t2", agent_id="b")])
    waves = plan.waves()
    assert len(waves) == 1
    assert {t.id for t in waves[0]} == {"t1", "t2"}


def test_dependency_orders_waves():
    plan = Plan(
        subtasks=[
            Subtask(id="t1", agent_id="a"),
            Subtask(id="t2", agent_id="b", depends_on=["t1"]),
        ]
    )
    waves = plan.waves()
    assert [t.id for t in waves[0]] == ["t1"]
    assert [t.id for t in waves[1]] == ["t2"]


def test_cycle_detected():
    plan = Plan(
        subtasks=[
            Subtask(id="t1", agent_id="a", depends_on=["t2"]),
            Subtask(id="t2", agent_id="b", depends_on=["t1"]),
        ]
    )
    with pytest.raises(DAGCycleError):
        plan.waves()


def test_unknown_dependency_raises():
    plan = Plan(subtasks=[Subtask(id="t1", agent_id="a", depends_on=["ghost"])])
    with pytest.raises(ValueError):
        plan.waves()
