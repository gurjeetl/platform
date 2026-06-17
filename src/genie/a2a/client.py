"""A2A client: send a JSON-RPC ``message/send`` to a distributed agent.

Ported from BaseAgentFramework ``a2a/client.py``, but endpoint-driven: the
endpoint is resolved by the discovery layer (``genie.discovery``) and passed in,
so the client itself has no registry dependency — it just builds the JSON-RPC
envelope and posts it.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from genie.a2a.agent_card import a2a_url
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    data_part,
)


class A2AError(RuntimeError):
    """Raised on transport failure or a JSON-RPC/agent error response."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.code = code


class A2AClient:
    @staticmethod
    def _headers() -> dict:
        token = os.getenv("AGENT_INVOKE_TOKEN")
        return {"Authorization": f"Bearer {token}"} if token else {}

    @staticmethod
    def _build_request(
        agent_id: str, args: dict | None, context: dict, sla_ms: int
    ) -> JsonRpcRequest:
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
            resp = await client.post(url, json=payload, headers=self._headers(), timeout=timeout)
            resp.raise_for_status()
            return self._parse_response(resp.json())

        if http is not None:
            return await _post(http)
        async with httpx.AsyncClient() as client:
            return await _post(client)
