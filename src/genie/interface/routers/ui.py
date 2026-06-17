"""UI-support endpoints for the bundled chat + trace visualizer frontends.

These mirror the shapes the ported React UIs expect (BaseAgentFramework):
  POST /api/v1/chat/ui        → {response, view}
  POST /api/v1/chat/trace     → step-by-step pipeline trace
  GET  /api/v1/registry       → discovered agents (live discovery or in-process)
  GET  /api/v1/conversations  → durable conversation list (Mongo; empty otherwise)
  GET  /api/v1/conversations/{id} / DELETE
  GET  /api/v1/state/{id}     → LangGraph checkpoint snapshot
"""

from __future__ import annotations

import contextlib
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from genie.application.checkpointer import get_thread_config
from genie.application.state import GraphState, Message
from genie.observability.logging import get_logger

router = APIRouter(prefix="/api/v1", tags=["ui"])
logger = get_logger(__name__)

# Nodes that actually touch a datastore (keep their db_ops in the trace).
_DB_OP_PRODUCERS = {"planner", "executor", "synthesizer", "input_guard", "output_guard"}
_DROP_FIELDS = {"messages", "tool_calls", "tool_results", "rag_context"}


class UIChatRequest(BaseModel):
    message: str
    thread_id: str


def _slim_update(update: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in update.items():
        if k in _DROP_FIELDS:
            continue
        if isinstance(v, str) and len(v) > 2000:
            out[k] = v[:2000] + "...[truncated]"
        else:
            out[k] = v
    return out


def _initial_state(message: str, thread_id: str) -> GraphState:
    return GraphState(
        conversation_id=thread_id,
        run_id=uuid.uuid4().hex,
        messages=[Message(role="user", content=message)],
    )


async def _save_turn(memory: Any, thread_id: str, role: str, content: str) -> None:
    if memory is None or getattr(memory, "mongo", None) is None:
        return
    with contextlib.suppress(Exception):
        await memory.mongo.save_turn(thread_id, role, content)


@router.post("/chat/ui", summary="Chat (UI shape: {response, view})")
async def chat_ui(body: UIChatRequest, request: Request) -> dict[str, Any]:
    graph = request.app.state.graph
    memory = getattr(request.app.state, "memory", None)
    config = get_thread_config(body.thread_id)
    await _save_turn(memory, body.thread_id, "user", body.message)
    result = await graph.ainvoke(
        _initial_state(body.message, body.thread_id).model_dump(), config=config
    )
    response = result.get("final_response") or result.get("error") or "Sorry, something went wrong."
    await _save_turn(memory, body.thread_id, "assistant", response)
    return {"response": response, "view": result.get("view")}


@router.post("/chat/trace", summary="Chat with a step-by-step pipeline trace")
async def chat_trace(body: UIChatRequest, request: Request) -> dict[str, Any]:
    graph = request.app.state.graph
    memory = getattr(request.app.state, "memory", None)
    run_id = uuid.uuid4().hex
    config = get_thread_config(f"{body.thread_id}:trace:{run_id}")

    steps: list[dict] = []
    cumulative: dict = {}
    t0 = time.perf_counter()
    state = _initial_state(body.message, body.thread_id)
    try:
        async for chunk in graph.astream(state.model_dump(), config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                cumulative.update(update)
                slim = _slim_update(update)
                if node not in _DB_OP_PRODUCERS:
                    slim.pop("db_ops", None)
                steps.append(
                    {
                        "node": node,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "update": slim,
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.error("chat_trace_failed", error=str(exc))
        return {"error": str(exc), "steps": steps}

    final_text = cumulative.get("final_response") or cumulative.get("error") or ""
    await _save_turn(memory, body.thread_id, "user", body.message)
    if final_text:
        await _save_turn(memory, body.thread_id, "assistant", final_text)

    # Synthetic "final" step: the run is done — clear this run's Redis blackboard
    # mirror (best-effort; the 1h TTL is the fallback). Surfaced so the trace shows
    # cleanup after the output guard. Only the executor writes the blackboard, so a
    # run that never reached it has nothing to clear (honest no-op).
    redis = getattr(memory, "redis", None) if memory is not None else None
    redis_enabled = bool(getattr(redis, "enabled", False))
    wrote_blackboard = bool(cumulative.get("blackboard"))
    if wrote_blackboard and redis_enabled and hasattr(redis, "delete_run"):
        with contextlib.suppress(Exception):
            await redis.delete_run(body.thread_id, run_id)
        final_op = {"store": "redis", "op": "delete", "node": "final",
                    "detail": "blackboard cleared (1h TTL is the fallback)",
                    "code": f"DEL bb:{body.thread_id}:{run_id}:*", "enabled": True}
    else:
        final_op = {"store": "redis", "op": "delete", "node": "final",
                    "detail": ("no blackboard written this run — nothing to clear (no-op)"
                               if not wrote_blackboard else "Redis disabled — TTL/no-op"),
                    "code": f"DEL bb:{body.thread_id}:{run_id}:*  → 0 keys", "enabled": redis_enabled}
    steps.append({"node": "final", "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                  "update": {"db_ops": [final_op]}})

    return {
        "user_input": body.message,
        "thread_id": body.thread_id,
        "run_id": run_id,
        "steps": steps,
        "session_loaded": {"turns": 0, "preview": "", "facts": []},
        "final": {
            "response": final_text,
            "view": cumulative.get("view"),
            "partial": bool(cumulative.get("partial")),
        },
    }


@router.get("/registry", summary="Discovered agents (for the trace UI)")
async def registry_dump(request: Request) -> dict[str, Any]:
    discovery = getattr(request.app.state, "discovery_client", None)
    if discovery is not None:
        try:
            metas = await discovery.list_active()
            return {
                "agents": [
                    {
                        "agent_id": m.agent_id,
                        "version": m.version,
                        "capability_tags": m.capability_tags,
                        "description": m.description,
                        "input_schema": {k: v.model_dump() for k, v in m.input_schema.items()},
                        "output_schema": {k: v.model_dump() for k, v in m.output_schema.items()},
                        "sla_ms": m.sla_ms,
                        "transport": m.transport,
                        "status": m.status,
                        "endpoint": m.endpoint,
                        "instance_id": m.instance_id,
                        "last_heartbeat": m.last_heartbeat.isoformat()
                        if m.last_heartbeat
                        else None,
                    }
                    for m in metas
                ]
            }
        except Exception as exc:  # noqa: BLE001
            return {"agents": [], "error": str(exc)}
    # Fallback: in-process registry (local/hybrid mode without a registry service).
    registry = request.app.state.agent_registry
    return {
        "agents": [
            {
                "agent_id": i.agent_id,
                "version": i.version,
                "capability_tags": i.tags,
                "description": i.description,
                "input_schema": i.input_schema,
                "output_schema": i.output_schema,
                "sla_ms": i.sla_ms,
                "transport": "in-process",
                "status": "active" if i.enabled else "deprecated",
                "endpoint": None,
                "instance_id": None,
                "last_heartbeat": None,
            }
            for i in registry.list_all()
        ]
    }


@router.get("/conversations", summary="List durable conversations (sidebar)")
async def list_conversations(request: Request, limit: int = 50) -> dict[str, Any]:
    memory = getattr(request.app.state, "memory", None)
    if memory is None or getattr(memory, "mongo", None) is None:
        return {"conversations": []}
    with contextlib.suppress(Exception):
        return {"conversations": await memory.mongo.list_conversations(limit=limit)}
    return {"conversations": []}


@router.get("/conversations/{thread_id}", summary="Full history of one conversation")
async def get_conversation(thread_id: str, request: Request) -> dict[str, Any]:
    memory = getattr(request.app.state, "memory", None)
    turns: list = []
    if memory is not None and getattr(memory, "mongo", None) is not None:
        with contextlib.suppress(Exception):
            turns = await memory.mongo.get_conversation(thread_id)
    return {"thread_id": thread_id, "turns": turns}


@router.delete("/conversations/{thread_id}", summary="Delete a conversation")
async def delete_conversation(thread_id: str, request: Request) -> dict[str, Any]:
    memory = getattr(request.app.state, "memory", None)
    if memory is not None and getattr(memory, "mongo", None) is not None:
        with contextlib.suppress(Exception):
            await memory.mongo.delete_conversation(thread_id)
    return {"deleted": thread_id}


@router.get("/state/{thread_id}", summary="LangGraph checkpoint snapshot")
async def get_state(thread_id: str, request: Request) -> dict[str, Any]:
    graph = request.app.state.graph
    snapshot = graph.get_state(get_thread_config(thread_id))
    return snapshot.values if snapshot and snapshot.values else {}
