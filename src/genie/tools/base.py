"""Tool base types — ToolCall, ToolResult, and ToolGateway protocol."""
from __future__ import annotations

import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    agent_id: str
    parameters: dict[str, Any] = {}


class ToolResult(BaseModel):
    call_id: str
    tool_id: str
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


@runtime_checkable
class ToolGateway(Protocol):
    async def execute(
        self,
        call: ToolCall,
        requesting_agent_id: str,
        user_id: str,
    ) -> ToolResult: ...
