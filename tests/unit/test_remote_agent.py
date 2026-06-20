"""RemoteAgent A2A dispatch + AgentInfo mapping (A2A transport mocked)."""

import pytest

from genie.agents.base import AgentTask
from genie.agents.remote import RemoteAgent
from genie.discovery.agent_meta import AgentMeta, FieldSpec


def _meta(**kw) -> AgentMeta:
    base = dict(
        agent_id="weather",
        description="city weather",
        endpoint="http://localhost:2010",
        capability_tags=["weather"],
        sla_ms=5000,
        input_schema={"location": FieldSpec(type="string", required=True)},
    )
    base.update(kw)
    return AgentMeta(**base)


def _rpc_reply(text: str, view: dict | None = None) -> dict:
    parts = [{"kind": "text", "text": text}]
    if view is not None:
        parts.append({"kind": "data", "data": {"view": view}})
    return {
        "jsonrpc": "2.0",
        "id": "t1",
        "result": {"kind": "message", "role": "agent", "messageId": "m1", "parts": parts},
    }


def test_get_info_maps_meta_to_agent_info():
    info = RemoteAgent(_meta()).get_info()
    assert info.agent_id == "weather"
    assert info.sla_ms == 5000
    assert info.input_schema["location"]["required"] is True
    assert "weather" in info.tags


@pytest.mark.asyncio
async def test_execute_dispatches_a2a_and_maps_reply(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:2010/a2a", json=_rpc_reply("Sunny, 20C", {"temp_c": 20})
    )
    agent = RemoteAgent(_meta())
    task = AgentTask(
        agent_id="weather",
        conversation_id="c",
        instruction="weather",
        context={"args": {"location": "paris"}, "task_id": "t1"},
    )
    result = await agent.execute(task)
    assert result.success is True
    assert result.output == "Sunny, 20C"
    assert result.data["view"] == {"temp_c": 20}


@pytest.mark.asyncio
async def test_execute_maps_jsonrpc_error_to_failure(httpx_mock):
    httpx_mock.add_response(
        url="http://localhost:2010/a2a",
        json={"jsonrpc": "2.0", "id": "t1", "error": {"code": -32001, "message": "agent boom"}},
    )
    agent = RemoteAgent(_meta())
    task = AgentTask(
        agent_id="weather",
        conversation_id="c",
        instruction="x",
        context={"args": {}, "task_id": "t1"},
    )
    result = await agent.execute(task)
    assert result.success is False
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_execute_missing_endpoint_fails_gracefully():
    agent = RemoteAgent(_meta(endpoint=None))
    task = AgentTask(
        agent_id="weather",
        conversation_id="c",
        instruction="x",
        context={"args": {}, "task_id": "t1"},
    )
    result = await agent.execute(task)
    assert result.success is False
