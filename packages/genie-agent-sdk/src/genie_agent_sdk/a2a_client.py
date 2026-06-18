"""A2A send client + registry discovery for agent-to-agent (peer) delegation.

The SDK already ships the A2A *types* (:mod:`genie_agent_sdk.a2a`) and the serve
side (:mod:`genie_agent_sdk.server`). This module adds the missing *caller* side
so an agent can fan work out to a peer mid-run (the "agents talk to each other"
half of A2A Hybrid):

  * :func:`resolve_endpoint` — discover a peer's endpoint via the Registry,
  * :class:`A2AClient` / :meth:`A2AClient.send` — POST a ``message/send`` to it,
  * :func:`call_agent` — resolve then send, returning the reply :class:`Message`.

Endpoint discovery is centralized in the Registry; transport is JSON-RPC. Env:
``REGISTRY_URL`` / ``REGISTRY_AUTH_TOKEN`` (discovery) and ``AGENT_INVOKE_TOKEN``
(A2A bearer), matching :mod:`genie_agent_sdk.server`.
"""
from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from genie_agent_sdk.a2a import (
    METHOD_MESSAGE_SEND,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    a2a_url,
    data_part,
)


class A2AError(RuntimeError):
    """Raised on transport failure, discovery miss, or a JSON-RPC/agent error."""

    def __init__(self, message: str, code: int | None = None) -> None:
        """Capture the message and an optional JSON-RPC error code."""
        super().__init__(message)
        self.code = code


def _registry_url() -> str:
    """Registry base URL from env (matches the serve-side default)."""
    return os.getenv("REGISTRY_URL", "http://127.0.0.1:2005").rstrip("/")


def _registry_headers() -> dict:
    """Bearer auth header for registry calls, or empty when no token is configured."""
    token = os.getenv("REGISTRY_AUTH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _invoke_headers() -> dict:
    """Bearer auth header for the A2A call from ``AGENT_INVOKE_TOKEN`` (empty when unset)."""
    token = os.getenv("AGENT_INVOKE_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


async def resolve_endpoint(
    agent_id: str, *, http: httpx.AsyncClient | None = None
) -> str | None:
    """Return the active endpoint advertised for ``agent_id``, or None if not found.

    Queries the Registry's ``GET /agents`` list and picks the first active record
    matching ``agent_id``. The list shape is tolerated defensively (``agents`` key
    or a bare list).
    """
    url = f"{_registry_url()}/agents"

    async def _get(client: httpx.AsyncClient) -> str | None:
        resp = await client.get(url, headers=_registry_headers())
        resp.raise_for_status()
        data = resp.json()
        records = data.get("agents", data) if isinstance(data, dict) else data
        for rec in records or []:
            if rec.get("agent_id") != agent_id:
                continue
            if rec.get("status", "active") != "active":
                continue
            endpoint = rec.get("endpoint")
            if endpoint:
                return endpoint
        return None

    if http is not None:
        return await _get(http)
    async with httpx.AsyncClient(timeout=float(os.getenv("REGISTRY_TIMEOUT_S", "3"))) as client:
        return await _get(client)


class A2AClient:
    """Stateless A2A JSON-RPC client; the target endpoint is passed per ``send``."""

    @staticmethod
    def _build_request(
        agent_id: str, args: dict | None, context: dict, sla_ms: int
    ) -> JsonRpcRequest:
        """Wrap args + routing context into a spec-shaped ``message/send`` request."""
        ctx = dict(context or {})
        message = Message(
            role="user",
            messageId=uuid.uuid4().hex,
            taskId=ctx.get("task_id"),
            contextId=ctx.get("thread_id"),
            parts=[data_part({"args": args or {}})],
            metadata={
                "agent_id": agent_id,
                "task_id": ctx.get("task_id"),
                "run_id": ctx.get("run_id"),
                "thread_id": ctx.get("thread_id"),
                "correlation_id": ctx.get("correlation_id") or uuid.uuid4().hex,
                "blackboard": ctx.get("blackboard") or {},
                "sla_ms": sla_ms,
            },
        )
        return JsonRpcRequest(
            id=ctx.get("task_id") or uuid.uuid4().hex,
            method=METHOD_MESSAGE_SEND,
            params={"message": message.model_dump(mode="json")},
        )

    @staticmethod
    def _parse_response(data: Any) -> Message:
        """Validate the JSON-RPC envelope; raise A2AError on error or empty result."""
        rpc = JsonRpcResponse.model_validate(data)
        if rpc.error is not None:
            raise A2AError(rpc.error.message, code=rpc.error.code)
        if not rpc.result:
            raise A2AError("A2A response had neither result nor error")
        return Message.model_validate(rpc.result)

    async def send(
        self,
        endpoint: str,
        agent_id: str,
        args: dict | None,
        context: dict,
        *,
        sla_ms: int,
        http: httpx.AsyncClient | None = None,
    ) -> Message:
        """POST a JSON-RPC ``message/send`` to ``endpoint``'s A2A URL; return the reply.

        Raises :class:`A2AError` on any transport, JSON-RPC, or agent error.
        """
        if not endpoint:
            raise A2AError(f"agent '{agent_id}' has no endpoint")
        url = a2a_url(endpoint)
        payload = self._build_request(agent_id, args, context, sla_ms).model_dump(mode="json")
        timeout = httpx.Timeout(sla_ms / 1000.0)

        async def _post(client: httpx.AsyncClient) -> Message:
            resp = await client.post(url, json=payload, headers=_invoke_headers(), timeout=timeout)
            resp.raise_for_status()
            return self._parse_response(resp.json())

        if http is not None:
            return await _post(http)
        async with httpx.AsyncClient() as client:
            return await _post(client)


async def call_agent(
    agent_id: str,
    args: dict | None,
    context: dict | None = None,
    *,
    sla_ms: int = 10000,
    http: httpx.AsyncClient | None = None,
) -> Message:
    """Discover ``agent_id`` via the Registry, then send it an A2A message.

    Raises :class:`A2AError` if the agent cannot be discovered or the call fails.
    """
    endpoint = await resolve_endpoint(agent_id, http=http)
    if not endpoint:
        raise A2AError(f"agent '{agent_id}' not discovered in registry")
    return await A2AClient().send(
        endpoint, agent_id, args, context or {}, sla_ms=sla_ms, http=http
    )
