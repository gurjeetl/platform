"""In-process application agents (weather / outage / rag) + provider wiring."""

import pytest

from applications.providers import AGENT_PROVIDERS
from genie.agents import AgentRegistry
from genie.agents.base import AgentTask


def _build_all():
    return [p(tool_gateway=None, settings=None) for p in AGENT_PROVIDERS]


def test_providers_register_three_agents():
    reg = AgentRegistry()
    for agent in _build_all():
        reg.register(agent)
    assert {a.agent_id for a in reg.list_all()} == {"weather", "outage", "rag"}


def test_agent_info_schemas():
    by_id = {a.agent_id: a for a in _build_all()}
    assert by_id["weather"].get_info().input_schema["location"]["required"] is True
    assert by_id["outage"].get_info().input_schema["outage_id"]["required"] is False
    assert "rag" in by_id["rag"].get_info().capabilities


def _task(agent_id: str, args: dict) -> AgentTask:
    return AgentTask(
        agent_id=agent_id,
        conversation_id="c",
        instruction="x",
        context={"args": args, "task_id": "t1"},
    )


@pytest.mark.asyncio
async def test_weather_execute():
    agent = next(a for a in _build_all() if a.agent_id == "weather")
    res = await agent.execute(_task("weather", {"location": "paris"}), {})
    assert res.success and "Paris" in res.output
    assert res.data["view"]["type"] == "weather"


@pytest.mark.asyncio
async def test_outage_list_and_detail():
    agent = next(a for a in _build_all() if a.agent_id == "outage")
    listed = await agent.execute(_task("outage", {}), {})
    assert listed.data["view"]["type"] == "outage_list"
    assert len(listed.data["view"]["items"]) == 5  # default top-5
    five = await agent.execute(_task("outage", {"limit": 5}), {})
    assert len(five.data["view"]["items"]) == 5
    detail = await agent.execute(_task("outage", {"outage_id": 18645677}), {})
    assert detail.data["view"]["type"] == "outage_detail"
    assert detail.data["view"]["outage_id"] == 18645677


@pytest.mark.asyncio
async def test_rag_execute_returns_citation():
    agent = next(a for a in _build_all() if a.agent_id == "rag")
    res = await agent.execute(_task("rag", {"query": "what is the a2a protocol"}), {})
    assert res.success and "[1]" in res.output
    assert res.data["view"]["sources"]
