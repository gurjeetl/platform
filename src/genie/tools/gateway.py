"""Concrete ToolGateway — dispatches ToolCalls to registered tool callables."""
from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Awaitable, Callable

from genie.observability.logging import get_logger
from genie.platform.errors import ErrorCode, GenieError
from genie.tools.base import ToolCall, ToolResult

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]

logger = get_logger(__name__)


def _row_count(output: Any) -> int | None:
    """Best-effort row count from a tool output."""
    if isinstance(output, list):
        return len(output)
    if isinstance(output, str):
        with contextlib.suppress(Exception):
            parsed = json.loads(output)
            if isinstance(parsed, list):
                return len(parsed)
            if isinstance(parsed, dict) and "rows" in parsed:
                return len(parsed["rows"])
    if isinstance(output, dict):
        return len(output.get("rows", [output]))
    return None


class ConcreteToolGateway:
    """Routes ToolCall requests to registered async handler functions."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolHandler] = {}

    def register(self, tool_id: str, handler: ToolHandler) -> None:
        self._tools[tool_id] = handler
        logger.info("tool_registered", tool_id=tool_id)

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    async def execute(
        self,
        call: ToolCall,
        requesting_agent_id: str,
        user_id: str,
    ) -> ToolResult:
        handler = self._tools.get(call.tool_id)
        if handler is None:
            raise GenieError(
                ErrorCode.NOT_FOUND, f"Tool '{call.tool_id}' not registered"
            )

        t0 = time.perf_counter()
        try:
            output, elapsed_ms = await self._run_handler_with_span(
                call, requesting_agent_id, handler, t0
            )
            logger.info(
                "tool_executed",
                tool_id=call.tool_id,
                agent_id=requesting_agent_id,
                elapsed_ms=round(elapsed_ms, 2),
                row_count=_row_count(output),
            )
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=True,
                output=output,
                execution_time_ms=elapsed_ms,
            )
        except GenieError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "tool_error",
                tool_id=call.tool_id,
                agent_id=requesting_agent_id,
                error=str(exc),
            )
            self._log_error_span(call, requesting_agent_id, elapsed_ms, str(exc))
            return ToolResult(
                call_id=call.call_id,
                tool_id=call.tool_id,
                success=False,
                error=str(exc),
                execution_time_ms=elapsed_ms,
            )

    async def _run_handler_with_span(
        self,
        call: ToolCall,
        agent_id: str,
        handler: ToolHandler,
        t0: float,
    ) -> tuple[Any, float]:
        """Run the handler wrapped in an MLflow child span.

        The span is opened BEFORE the await so it becomes a child of
        whatever span the calling LangGraph node created via autolog —
        asyncio contextvars propagate the active span through awaits.

        Falls back to a plain handler call if MLflow is unavailable.
        """
        try:
            import mlflow
            with mlflow.start_span(
                name=f"tool:{call.tool_id}",
                span_type="TOOL",
            ) as span:
                with contextlib.suppress(Exception):
                    span.set_inputs({
                        "tool_id": call.tool_id,
                        "agent_id": agent_id,
                        "parameters": json.dumps(call.parameters, default=str),
                    })

                output = await handler(call.parameters)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                with contextlib.suppress(Exception):
                    out: dict[str, Any] = {
                        "success": True,
                        "elapsed_ms": round(elapsed_ms, 2),
                    }
                    count = _row_count(output)
                    if count is not None:
                        out["row_count"] = count
                    span.set_outputs(out)

                return output, elapsed_ms

        except ImportError:
            output = await handler(call.parameters)
            return output, (time.perf_counter() - t0) * 1000

    def _log_error_span(
        self,
        call: ToolCall,
        agent_id: str,
        elapsed_ms: float,
        error: str,
    ) -> None:
        with contextlib.suppress(Exception):
            import mlflow
            with mlflow.start_span(name=f"tool:{call.tool_id}", span_type="TOOL") as span:
                span.set_inputs({
                    "tool_id": call.tool_id,
                    "agent_id": agent_id,
                    "parameters": json.dumps(call.parameters, default=str),
                })
                span.set_outputs({
                    "success": False,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error": error,
                })
