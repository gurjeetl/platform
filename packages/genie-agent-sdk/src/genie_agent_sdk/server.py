"""Reusable harness that turns a BaseAgent into a self-registering remote service.

``serve_agent(agent, agent_meta=..., ...)`` builds a FastAPI app that:

  * exposes the formal A2A surface — ``POST /a2a`` (JSON-RPC ``message/send``)
    and ``GET /.well-known/agent.json`` (the Agent Card),
  * self-registers its :class:`AgentMeta` with the Registry Service on startup,
  * heartbeats on an interval so the registry keeps it "live" (TTL),
  * re-registers automatically if the registry swept/restarted,
  * deregisters on shutdown.

The /a2a wire contract matches the platform's A2A client:
  request  params.message has parts=[{kind:"data", data:{args:{...}}}] and
           metadata{task_id, run_id, thread_id, blackboard, sla_ms};
  response result is a Message role="agent" with parts=[TextPart, optional
           DataPart{data:{view:...}}].
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from genie_agent_sdk.a2a import (
    ERR_AGENT_EXECUTION,
    ERR_INVALID_PARAMS,
    ERR_METHOD_NOT_FOUND,
    METHOD_MESSAGE_SEND,
    JsonRpcError,
    JsonRpcResponse,
    Message,
    data_part,
    get_data,
    text_part,
    to_agent_card,
)
from genie_agent_sdk.agent_meta import AgentMeta
from genie_agent_sdk.base_agent import build_task_state

load_dotenv()
_log = logging.getLogger("genie_agent_sdk.server")


# --- Config helpers ---------------------------------------------------------
def _registry_headers() -> dict:
    token = os.getenv("REGISTRY_AUTH_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def _require_a2a_auth(authorization: str | None = Header(None)) -> None:
    token = os.getenv("AGENT_INVOKE_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid A2A token")


# --- Registry interactions --------------------------------------------------
async def _register(client: httpx.AsyncClient, registry_url: str, meta: AgentMeta) -> int | None:
    """Register; return the heartbeat interval (seconds) or None on failure."""
    try:
        resp = await client.post(
            f"{registry_url}/register",
            json={"meta": meta.model_dump(mode="json")},
            headers=_registry_headers(),
        )
        resp.raise_for_status()
        data = resp.json()
        _log.info("agent.registered agent_id=%s instance_id=%s", meta.agent_id, meta.instance_id)
        return int(data.get("heartbeat_interval_seconds") or 0) or None
    except Exception as e:
        _log.warning("agent.register_failed error=%s", e)
        return None


async def _heartbeat_loop(
    client: httpx.AsyncClient, registry_url: str, meta: AgentMeta, interval: int
) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            resp = await client.post(
                f"{registry_url}/heartbeat/{meta.instance_id}",
                json={"instance_id": meta.instance_id, "status": meta.status},
                headers=_registry_headers(),
            )
            resp.raise_for_status()
            if not resp.json().get("known", False):
                _log.info("agent.reregister_unknown agent_id=%s", meta.agent_id)
                await _register(client, registry_url, meta)
        except Exception as e:
            _log.warning("agent.heartbeat_failed error=%s", e)
            await _register(client, registry_url, meta)  # self-heal


# --- App factory ------------------------------------------------------------
def build_agent_app(
    agent,
    *,
    agent_meta: AgentMeta,
    advertised_endpoint: str,
    registry_url: str,
    default_heartbeat: int = 30,
) -> FastAPI:
    """Build the FastAPI app for one agent instance (no server started)."""
    meta = agent_meta.model_copy(
        update={"endpoint": advertised_endpoint, "instance_id": uuid.uuid4().hex}
    )
    registry_url = registry_url.rstrip("/")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        client = httpx.AsyncClient(timeout=float(os.getenv("REGISTRY_TIMEOUT_S", "3")))
        interval = await _register(client, registry_url, meta) or default_heartbeat
        hb_task = asyncio.create_task(_heartbeat_loop(client, registry_url, meta, interval))
        try:
            yield
        finally:
            hb_task.cancel()
            try:
                await client.post(
                    f"{registry_url}/deregister",
                    json={"instance_id": meta.instance_id},
                    headers=_registry_headers(),
                )
            except Exception:
                pass
            await client.aclose()

    app = FastAPI(title=f"agent:{meta.agent_id}", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "agent_id": meta.agent_id, "instance_id": meta.instance_id}

    @app.get("/.well-known/agent.json")
    async def agent_card() -> dict:
        return to_agent_card(meta).model_dump(mode="json")

    @app.post("/a2a", dependencies=[Depends(_require_a2a_auth)])
    async def a2a(body: dict) -> dict:
        """A2A JSON-RPC 2.0 endpoint. Handles ``message/send``.

        Args travel in a request DataPart (``{"args": {...}}``); invocation
        context (task_id, run_id, thread_id, blackboard, sla_ms) travels in the
        message ``metadata``. The reply is an agent-role Message: a TextPart with
        the answer and an optional DataPart carrying a structured ``view``.
        """
        rpc_id = body.get("id")

        def _err(code: int, message: str) -> dict:
            return JsonRpcResponse(
                id=rpc_id, error=JsonRpcError(code=code, message=message)
            ).model_dump(mode="json")

        if body.get("method") != METHOD_MESSAGE_SEND:
            return _err(ERR_METHOD_NOT_FOUND, f"unsupported method '{body.get('method')}'")
        try:
            in_msg = Message.model_validate((body.get("params") or {}).get("message") or {})
        except Exception as e:  # malformed message payload
            return _err(ERR_INVALID_PARAMS, f"invalid message: {e}")

        meta_in = in_msg.metadata or {}
        args = (get_data(in_msg) or {}).get("args") or {}
        state = build_task_state(
            task_id=meta_in.get("task_id") or in_msg.taskId or "",
            agent_id=meta_in.get("agent_id") or meta.agent_id,
            args=args,
            thread_id=meta_in.get("thread_id") or in_msg.contextId or "",
            run_id=meta_in.get("run_id") or "",
            blackboard=meta_in.get("blackboard") or {},
        )
        result_state = await asyncio.to_thread(agent.run, state)
        if result_state.get("error"):
            return _err(ERR_AGENT_EXECUTION, str(result_state["error"]))

        parts = [text_part(result_state.get("final_output") or "")]
        view = result_state.get("view")
        if view:
            parts.append(data_part({"view": view}))
        out_msg = Message(
            role="agent",
            messageId=uuid.uuid4().hex,
            taskId=meta_in.get("task_id"),
            contextId=meta_in.get("thread_id"),
            parts=parts,
            metadata={"agent_id": meta.agent_id},
        )
        return JsonRpcResponse(id=rpc_id, result=out_msg.model_dump(mode="json")).model_dump(mode="json")

    return app


def serve_agent(
    agent,
    *,
    agent_meta: AgentMeta,
    host: str | None = None,
    port: int | None = None,
    registry_url: str | None = None,
) -> None:
    """Run ``agent`` as a self-registering A2A service (blocking).

    ``agent`` is a ready BaseAgent instance. ``agent_meta`` is its advertised
    :class:`AgentMeta`. host/port/registry_url fall back to env
    (AGENT_HOST, AGENT_PORT, REGISTRY_URL) then to sane defaults.
    """
    host = host or os.getenv("AGENT_HOST", "127.0.0.1")
    port = port or int(os.getenv("AGENT_PORT", "8010"))
    registry_url = registry_url or os.getenv("REGISTRY_URL", "http://127.0.0.1:2005")

    advertise_host = os.getenv("AGENT_ADVERTISE_HOST") or host
    advertise_port = os.getenv("AGENT_ADVERTISE_PORT") or str(port)
    advertised_endpoint = f"http://{advertise_host}:{advertise_port}"

    app = build_agent_app(
        agent,
        agent_meta=agent_meta,
        advertised_endpoint=advertised_endpoint,
        registry_url=registry_url,
        default_heartbeat=int(os.getenv("REGISTRY_HEARTBEAT_SECONDS", "30")),
    )
    uvicorn.run(app, host=host, port=port)


# Convenience alias matching the framework's name.
AgentServer = build_agent_app
