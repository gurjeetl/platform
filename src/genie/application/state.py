"""LangGraph graph state definition for the Genie workflow."""

import uuid
from typing import Any, Literal

from genie_rag_contracts.retrieval import RetrievalResult
from pydantic import BaseModel, Field


class Message(BaseModel):
    """One chat-history turn (system / user / assistant / tool)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ToolCallRecord(BaseModel):
    """API-compat record of a dispatched agent/tool call (surfaced in the trace)."""

    call_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tool_id: str
    agent_id: str
    parameters: dict[str, Any] = {}
    timestamp: str = ""


class ToolResultRecord(BaseModel):
    """API-compat result for a ``ToolCallRecord`` — success/output or error."""

    call_id: str
    tool_id: str
    success: bool
    output: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0


class GraphState(BaseModel):
    """The single mutable state object threaded through every LangGraph node.

    Nodes return partial dicts that LangGraph merges into this model; later nodes
    read what earlier ones wrote. Fields fall into the legacy/API-compat block
    (top) and the DAG planner/orchestrator/gate/synthesizer block (below).
    """

    conversation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = "anonymous"
    messages: list[Message] = []
    request_type: Literal["rag_query", "agent_task", "domain_query", "general_chat"] | None = None
    selected_agents: list[str] = []
    tool_calls: list[ToolCallRecord] = []
    tool_results: list[ToolResultRecord] = []
    rag_context: list[RetrievalResult] = []
    final_response: str | None = None
    error: str | None = None
    requires_approval: bool = False
    approved: bool | None = None
    hitl_prompt: str | None = None
    metadata: dict[str, Any] = {}
    rag_unavailable: bool = False

    # ── DAG planner / wave orchestrator / replan gate / blackboard synthesizer ──
    # (ported from BaseAgentFramework — the distributed multi-agent pipeline)
    # Router decision: "plan" (full pipeline) | "fast" (skip to executor) |
    # "chitchat" (skip to synthesizer).
    route: str | None = None
    # Plan is Plan.model_dump(): {"subtasks": [{id, agent_id, args, depends_on, sla_ms}]}
    plan: dict[str, Any] | None = None
    agent_versions: dict[str, str] = {}
    # Orchestrator decomposition: task ids per dependency wave.
    waves: list[list[str]] | None = None
    plan_error: str | None = None
    # Shared blackboard: {task_id: {agent_id, text, view?} | {error: ...}}.
    blackboard: dict[str, Any] = {}
    blackboard_snapshot: dict[str, Any] | None = None
    replan_count: int = 0
    replan_reason: str | None = None
    # Completion-gate decision surfaced to the trace UI: "synthesize" | "replan".
    next_action: str | None = None
    partial: bool = False
    # Optional structured view passed through to the chat response.
    view: dict[str, Any] | None = None
    # Per-node real datastore operations, surfaced to the trace UI's Live DB panel.
    db_ops: list[dict[str, Any]] | None = None
    # Content-guard records (set by input/output guard nodes). guard_block is
    # truthy only when a blocking scanner fired (short-circuits to a safe refusal).
    guard_block: dict[str, Any] | None = None
    guard_input: dict[str, Any] | None = None
    guard_output: dict[str, Any] | None = None
    is_complete: bool = False
