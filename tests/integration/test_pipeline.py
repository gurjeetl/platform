"""End-to-end pipeline over the compiled LangGraph with stub LLM + fake agents."""

import pytest

from genie.application.graph import build_graph
from genie.application.state import GraphState, Message


def _router_then_planner(system: str, user: str) -> str:
    # The router prompt mentions "ROUTER"; the planner prompt mentions "planning agent".
    if "ROUTER" in system:
        return '{"route":"plan","agent_id":null,"args":{},"confidence":0.0}'
    if "planning agent" in system:
        return (
            '{"subtasks":['
            '{"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},'
            '{"id":"t2","agent_id":"outage","args":{},"depends_on":[]}]}'
        )
    return "Final merged answer combining weather and outages."


@pytest.mark.asyncio
async def test_plan_path_runs_agents_and_synthesizes(settings, agent_registry):
    from tests.conftest import StubLLM

    graph, _ = build_graph(
        llm_provider=StubLLM(_router_then_planner), agent_registry=agent_registry, settings=settings
    )
    st = GraphState(
        conversation_id="c1",
        messages=[Message(role="user", content="weather in tokyo and top outages")],
    )
    out = await graph.ainvoke(st.model_dump(), config={"configurable": {"thread_id": "c1"}})
    assert out["final_response"] == "Final merged answer combining weather and outages."
    assert set(out["blackboard"].keys()) == {"t1", "t2"}
    assert out["is_complete"] is True


@pytest.mark.asyncio
async def test_fast_path_single_agent(settings, agent_registry):
    from tests.conftest import StubLLM

    fast = '{"route":"fast","agent_id":"weather","args":{"location":"paris"},"confidence":0.95}'
    graph, _ = build_graph(
        llm_provider=StubLLM(lambda s, u: fast), agent_registry=agent_registry, settings=settings
    )
    st = GraphState(
        conversation_id="c2", messages=[Message(role="user", content="weather in paris")]
    )
    out = await graph.ainvoke(st.model_dump(), config={"configurable": {"thread_id": "c2"}})
    assert "paris" in out["final_response"].lower()
    assert out["view"] == {"temp_c": 20}
