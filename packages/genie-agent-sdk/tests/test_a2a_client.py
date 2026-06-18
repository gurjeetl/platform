"""A2A send-client + discovery tests using an injected fake httpx client.

``resolve_endpoint`` / ``A2AClient.send`` / ``call_agent`` all accept an
``http=`` client, so we exercise the full resolve→send flow without real
networking or monkeypatching.
"""
import asyncio

import pytest

from genie_agent_sdk.a2a import JsonRpcResponse, Message, get_text, text_part
from genie_agent_sdk.a2a_client import A2AClient, A2AError, call_agent, resolve_endpoint


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Routes GET → registry agents list, POST → an A2A reply."""

    def __init__(self, agents: list[dict], reply: dict) -> None:
        self._agents = agents
        self._reply = reply
        self.posted_to: str | None = None

    async def get(self, url, headers=None):
        return _FakeResponse({"agents": self._agents})

    async def post(self, url, json=None, headers=None, timeout=None):
        self.posted_to = url
        return _FakeResponse(self._reply)


def _reply_envelope(text: str) -> dict:
    """A valid JSON-RPC response whose result is an agent Message."""
    msg = Message(role="agent", messageId="m1", parts=[text_part(text)])
    return JsonRpcResponse(id="t1", result=msg.model_dump(mode="json")).model_dump(mode="json")


def test_resolve_endpoint_returns_active_match():
    agents = [
        {"agent_id": "outage", "status": "active", "endpoint": "http://h:2011"},
        {"agent_id": "weather", "status": "active", "endpoint": "http://h:2010"},
    ]
    client = _FakeClient(agents, {})
    ep = asyncio.run(resolve_endpoint("weather", http=client))
    assert ep == "http://h:2010"


def test_resolve_endpoint_skips_inactive_and_missing():
    agents = [{"agent_id": "weather", "status": "deprecated", "endpoint": "http://h:2010"}]
    client = _FakeClient(agents, {})
    assert asyncio.run(resolve_endpoint("weather", http=client)) is None
    assert asyncio.run(resolve_endpoint("nope", http=client)) is None


def test_send_parses_reply_message():
    client = _FakeClient([], _reply_envelope("hello from peer"))
    reply = asyncio.run(
        A2AClient().send("http://h:2010", "weather", {"location": "paris"}, {}, sla_ms=4000, http=client)
    )
    assert isinstance(reply, Message)
    assert get_text(reply) == "hello from peer"
    assert client.posted_to == "http://h:2010/a2a"


def test_call_agent_resolves_then_sends():
    agents = [{"agent_id": "weather", "status": "active", "endpoint": "http://h:2010"}]
    client = _FakeClient(agents, _reply_envelope("sunny"))
    reply = asyncio.run(call_agent("weather", {"location": "paris"}, {}, http=client))
    assert get_text(reply) == "sunny"


def test_call_agent_raises_when_not_discovered():
    client = _FakeClient([], {})
    with pytest.raises(A2AError, match="not discovered"):
        asyncio.run(call_agent("ghost", {}, {}, http=client))


def test_send_raises_on_jsonrpc_error():
    err = {"jsonrpc": "2.0", "id": "t1", "error": {"code": -32001, "message": "agent boom"}}
    client = _FakeClient([], err)
    with pytest.raises(A2AError, match="agent boom"):
        asyncio.run(A2AClient().send("http://h:2010", "weather", {}, {}, sla_ms=1000, http=client))
