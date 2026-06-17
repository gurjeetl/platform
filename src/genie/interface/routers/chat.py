"""Chat endpoint — submit a message and receive a pipeline response."""
from __future__ import annotations

import itertools
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from genie.application.checkpointer import get_thread_config
from genie.application.state import GraphState, Message
from genie.observability.correlation import get_correlation_id
from genie.observability.logging import get_logger
from genie.platform.errors import ErrorCode, GenieError, error_response
from genie.security.auth import sanitize_user_input

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])
logger = get_logger(__name__)

# In-memory counter for readable conversation IDs (Genie_0001, Genie_0002, …).
# Resets on server restart — intentional; useful for testing, not for production audit trails.
_conv_counter = itertools.count(1)


def _next_conversation_id() -> str:
    return f"Genie_{next(_conv_counter):04d}"


@asynccontextmanager
async def _null_context() -> AsyncIterator[None]:
    yield


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None   # auto-generates Genie_XXXX when omitted
    user_id: str = "PankajG"
    metadata: dict[str, Any] = {}


class ChatResponse(BaseModel):
    conversation_id: str
    response: str
    request_type: str | None = None
    agents_used: list[str] = []
    rag_used: bool = False
    correlation_id: str = ""


def _get_graph(request: Request) -> Any:
    graph = getattr(request.app.state, "graph", None)
    if graph is None:
        raise GenieError(ErrorCode.INTERNAL_ERROR, "Graph not initialised")
    return graph


def _get_tracker(request: Request) -> Any:
    return getattr(request.app.state, "tracker", None)


async def _invoke_with_trace(
    graph: Any,
    initial_state: Any,
    thread_config: dict[str, Any],
    conversation_id: str,
    message: str,
) -> dict[str, Any]:
    """Run graph.ainvoke wrapped in an MLflow trace span for pipeline visibility."""
    try:
        import mlflow
        with mlflow.start_span(name="genie_pipeline") as span:
            span.set_inputs({"message": message, "conversation_id": conversation_id})
            result = await graph.ainvoke(initial_state.model_dump(), config=thread_config)
            span.set_outputs({
                "request_type": result.get("request_type") or "",
                "agents_used": result.get("selected_agents", []),
                "response_length": len(result.get("final_response") or ""),
            })
            return result
    except Exception:
        # If MLflow tracing is unavailable, run without it
        return await graph.ainvoke(initial_state.model_dump(), config=thread_config)


@router.post("", response_model=ChatResponse, summary="Send a chat message")
async def chat(body: ChatRequest, request: Request) -> ChatResponse:
    graph = _get_graph(request)
    tracker = _get_tracker(request)

    try:
        clean_message = sanitize_user_input(body.message)
    except GenieError as exc:
        raise HTTPException(status_code=400, detail=error_response(exc).model_dump())

    conversation_id = body.conversation_id or _next_conversation_id()
    correlation_id = get_correlation_id() or str(uuid.uuid4())

    initial_state = GraphState(
        conversation_id=conversation_id,
        correlation_id=correlation_id,
        user_id=body.user_id,
        messages=[Message(role="user", content=clean_message)],
        metadata=body.metadata,
    )

    thread_config = get_thread_config(conversation_id)

    run_ctx = tracker.start_run(
        run_name=conversation_id,
        tags={"conversation_id": conversation_id, "user_id": body.user_id},
    ) if tracker else None

    async with (run_ctx if run_ctx else _null_context()):
        if run_ctx and tracker:
            tracker.log_params({
                "user_id": body.user_id,
                "conversation_id": conversation_id,
                "message_length": len(clean_message),
            })

        try:
            final_state = await _invoke_with_trace(
                graph, initial_state, thread_config, conversation_id, clean_message
            )
        except GenieError as exc:
            raise HTTPException(status_code=500, detail=error_response(exc).model_dump())
        except Exception as exc:
            logger.error("chat_pipeline_error", error=str(exc), conversation_id=conversation_id)
            raise HTTPException(status_code=500, detail={"message": str(exc)})

        if run_ctx and tracker:
            tracker.log_params({
                "request_type": final_state.get("request_type") or "",
                "agents_used": ",".join(final_state.get("selected_agents", [])),
                "rag_used": str(bool(final_state.get("rag_context"))),
            })
            tracker.log_metrics({
                "response_length": len(final_state.get("final_response") or ""),
                "agents_count": len(final_state.get("selected_agents", [])),
            })

    return ChatResponse(
        conversation_id=conversation_id,
        response=final_state.get("final_response") or "",
        request_type=final_state.get("request_type"),
        agents_used=final_state.get("selected_agents", []),
        rag_used=bool(final_state.get("rag_context")),
        correlation_id=correlation_id,
    )
