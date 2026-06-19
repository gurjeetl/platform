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
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
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
    """Chat body used by the bundled UIs (message + thread id)."""

    message: str
    thread_id: str


def _slim_update(update: dict) -> dict:
    """Drop heavy fields and truncate long strings so trace payloads stay small."""
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
    """Build the starting GraphState for one user turn."""
    return GraphState(
        conversation_id=thread_id,
        run_id=uuid.uuid4().hex,
        messages=[Message(role="user", content=message)],
    )


async def _save_turn(memory: Any, thread_id: str, role: str, content: str) -> None:
    """Persist one conversation turn to Mongo; best-effort no-op without it."""
    if memory is None or getattr(memory, "mongo", None) is None:
        return
    with contextlib.suppress(Exception):
        await memory.mongo.save_turn(thread_id, role, content)


@router.post("/chat/ui", summary="Chat (UI shape: {response, view})")
async def chat_ui(body: UIChatRequest, request: Request) -> dict[str, Any]:
    """Run one turn and return ``{response, view}`` for the chat UI.

    Saves both the user and assistant turns to durable memory when available.
    """
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


async def _trace_events(
    graph: Any, memory: Any, message: str, thread_id: str
) -> AsyncIterator[dict]:
    """Run the pipeline and yield trace events as each node completes.

    Emits, in order: one ``meta`` event, one ``step`` event per node update as it
    streams from the graph, a synthetic ``final`` cleanup ``step``, then a ``done``
    event with the final answer. On failure, an ``error`` event. Both the buffered
    (``/chat/trace``) and streaming (``/chat/trace/stream``) endpoints consume this.
    """
    # Derive run_id from the state so the cleanup DEL below targets the SAME
    # bb:{thread}:{run}:* keys the executor actually wrote (state.run_id).
    state = _initial_state(message, thread_id)
    run_id = state.run_id
    config = get_thread_config(f"{thread_id}:trace:{run_id}")
    cumulative: dict = {}
    t0 = time.perf_counter()

    yield {"type": "meta", "run_id": run_id, "user_input": message, "thread_id": thread_id}

    try:
        async for chunk in graph.astream(state.model_dump(), config=config, stream_mode="updates"):
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                cumulative.update(update)
                slim = _slim_update(update)
                if node not in _DB_OP_PRODUCERS:
                    slim.pop("db_ops", None)
                yield {
                    "type": "step",
                    "node": node,
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                    "update": slim,
                }
    except Exception as exc:  # noqa: BLE001
        logger.error("chat_trace_failed", error=str(exc))
        yield {"type": "error", "error": str(exc)}
        return

    final_text = cumulative.get("final_response") or cumulative.get("error") or ""
    await _save_turn(memory, thread_id, "user", message)
    if final_text:
        await _save_turn(memory, thread_id, "assistant", final_text)

    # Synthetic "final" step: the run is done — clear this run's Redis blackboard
    # mirror (best-effort; the 1h TTL is the fallback). Honest no-op when nothing ran.
    redis = getattr(memory, "redis", None) if memory is not None else None
    redis_enabled = bool(getattr(redis, "enabled", False))
    wrote_blackboard = bool(cumulative.get("blackboard"))
    if wrote_blackboard and redis_enabled and hasattr(redis, "delete_run"):
        removed = 0
        with contextlib.suppress(Exception):
            removed = await redis.delete_run(thread_id, run_id)
        plural = "" if removed == 1 else "s"
        final_op = {
            "store": "redis",
            "op": "delete",
            "node": "final",
            "detail": f"blackboard cleared — {removed} key{plural} removed (1h TTL is the fallback)",
            "code": f"DEL bb:{thread_id}:{run_id}:*  → {removed} key{plural}",
            "enabled": True,
        }
    else:
        final_op = {
            "store": "redis",
            "op": "delete",
            "node": "final",
            "detail": (
                "no blackboard written this run — nothing to clear (no-op)"
                if not wrote_blackboard
                else "Redis disabled — TTL/no-op"
            ),
            "code": f"DEL bb:{thread_id}:{run_id}:*  → 0 keys",
            "enabled": redis_enabled,
        }
    yield {
        "type": "step",
        "node": "final",
        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
        "update": {"db_ops": [final_op]},
    }

    yield {
        "type": "done",
        "session_loaded": {"turns": 0, "preview": "", "facts": []},
        "final": {
            "response": final_text,
            "view": cumulative.get("view"),
            "partial": bool(cumulative.get("partial")),
        },
    }


@router.post("/chat/trace", summary="Chat with a step-by-step pipeline trace (buffered)")
async def chat_trace(body: UIChatRequest, request: Request) -> dict[str, Any]:
    """Run the pipeline and return the full trace (meta + steps + final) at once."""
    graph = request.app.state.graph
    memory = getattr(request.app.state, "memory", None)
    meta: dict = {}
    steps: list[dict] = []
    final: dict = {}
    session: dict = {"turns": 0, "preview": "", "facts": []}
    async for ev in _trace_events(graph, memory, body.message, body.thread_id):
        kind = ev["type"]
        if kind == "meta":
            meta = ev
        elif kind == "step":
            steps.append(
                {"node": ev["node"], "elapsed_ms": ev["elapsed_ms"], "update": ev["update"]}
            )
        elif kind == "error":
            return {"error": ev["error"], "steps": steps}
        elif kind == "done":
            final, session = ev["final"], ev["session_loaded"]
    return {
        "user_input": meta.get("user_input"),
        "thread_id": meta.get("thread_id"),
        "run_id": meta.get("run_id"),
        "steps": steps,
        "session_loaded": session,
        "final": final,
    }


@router.post("/chat/trace/stream", summary="Chat trace streamed live (NDJSON, one event per line)")
async def chat_trace_stream(body: UIChatRequest, request: Request) -> StreamingResponse:
    """Stream the trace as it executes — one JSON event per line (meta / step / done /
    error). The UI renders each node the moment it finishes instead of waiting for the
    whole pipeline, so input_guard → router → planner → … appear live."""
    graph = request.app.state.graph
    memory = getattr(request.app.state, "memory", None)

    async def _ndjson() -> AsyncIterator[str]:
        async for ev in _trace_events(graph, memory, body.message, body.thread_id):
            yield json.dumps(ev) + "\n"

    # no-transform/no-buffering hints so proxies flush each line immediately.
    return StreamingResponse(
        _ndjson(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@router.get("/registry", summary="Discovered agents (for the trace UI)")
async def registry_dump(request: Request) -> dict[str, Any]:
    """Return discovered agents for the trace UI.

    Prefers live discovery (distributed/hybrid); falls back to the in-process
    registry in local mode or when the registry service is unavailable.
    """
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
    """List durable conversations from Mongo; empty when memory is in-process."""
    memory = getattr(request.app.state, "memory", None)
    if memory is None or getattr(memory, "mongo", None) is None:
        return {"conversations": []}
    with contextlib.suppress(Exception):
        return {"conversations": await memory.mongo.list_conversations(limit=limit)}
    return {"conversations": []}


@router.get("/conversations/{thread_id}", summary="Full history of one conversation")
async def get_conversation(thread_id: str, request: Request) -> dict[str, Any]:
    """Return all stored turns for *thread_id* (empty without durable memory)."""
    memory = getattr(request.app.state, "memory", None)
    turns: list = []
    if memory is not None and getattr(memory, "mongo", None) is not None:
        with contextlib.suppress(Exception):
            turns = await memory.mongo.get_conversation(thread_id)
    return {"thread_id": thread_id, "turns": turns}


@router.delete("/conversations/{thread_id}", summary="Delete a conversation")
async def delete_conversation(thread_id: str, request: Request) -> dict[str, Any]:
    """Delete a conversation from durable memory (best-effort no-op without it)."""
    memory = getattr(request.app.state, "memory", None)
    if memory is not None and getattr(memory, "mongo", None) is not None:
        with contextlib.suppress(Exception):
            await memory.mongo.delete_conversation(thread_id)
    return {"deleted": thread_id}


@router.get("/state/{thread_id}", summary="LangGraph checkpoint snapshot")
async def get_state(thread_id: str, request: Request) -> dict[str, Any]:
    """Return the LangGraph checkpoint values for *thread_id* ({} if none)."""
    graph = request.app.state.graph
    snapshot = graph.get_state(get_thread_config(thread_id))
    return snapshot.values if snapshot and snapshot.values else {}
