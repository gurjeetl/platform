"""Base types for the agent framework: capabilities, tasks, results, and protocol."""

from __future__ import annotations  # enables | union syntax on Python 3.9

import uuid
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, model_validator


class AgentCapability(str):
    """Open string type for agent capability identifiers.

    NOT a closed Enum. Application agents use plain string literals and never
    need to import or subclass this type.

    Platform-owned well-known values:
      ``"general"``       — fallback for unrecognised capability hints
      ``"rag_retrieval"`` — RAG document-search agents

    All other capability strings (``"conductor_data"``, ``"meter_data_availability"``,
    etc.) are owned by the application and declared in each ``CapabilitySpec``.
    """

    def __new__(cls, value: str) -> AgentCapability:
        """Construct a capability from any string (no validation — it's open)."""
        return str.__new__(cls, value)

    GENERAL = "general"
    RAG_RETRIEVAL = "rag_retrieval"


class CapabilitySpec(BaseModel):
    """Single capability declaration — the agent's uniform, self-describing card entry.

    Bundles three things that were previously scattered across the class body and
    ``get_info()``:

    * ``id``               — the capability string the Planner matches against
    * ``routing_keywords`` — keywords the Router uses for heuristic fallback
    * ``input_schema``     — JSON Schema for ``AgentTask.context``; makes the
                             catalog machine-readable (like MCP ``tools/list``)

    Every agent declares one ``CapabilitySpec`` per capability it can handle.
    Having them together prevents the ``capabilities`` / ``routing_keywords``
    desync that was possible when they lived in separate lists.
    """

    id: str
    display_name: str = ""
    description: str = ""
    routing_keywords: list[str] = []
    input_schema: dict[str, Any] = {}


class AgentTask(BaseModel):
    """A unit of work dispatched to an agent: instruction + structured ``context``."""

    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    conversation_id: str = ""
    correlation_id: str = ""
    instruction: str
    context: dict[str, Any] = {}
    metadata: dict[str, Any] = {}


class AgentResult(BaseModel):
    """An agent's reply: ``output`` text plus optional structured ``data`` or ``error``."""

    task_id: str
    agent_id: str
    success: bool
    output: str
    data: dict[str, Any] = {}
    error: str | None = None
    execution_time_ms: float = 0.0


class AgentInfo(BaseModel):
    """The agent's self-description — the uniform card registered in ``AgentRegistry``.

    Agents populate ``capability_specs`` with one ``CapabilitySpec`` per capability.
    The flat ``capabilities`` and ``routing_keywords`` lists are derived automatically
    via a model validator so all existing platform code (Planner, Router, REST
    response) continues to work without changes.

    Backward compat: agents that still set ``capabilities`` / ``routing_keywords``
    directly (without ``capability_specs``) continue to work — the validator only
    auto-derives when ``capability_specs`` is non-empty.
    """

    agent_id: str
    name: str
    description: str
    version: str
    enabled: bool
    health: str = "healthy"  # managed by AgentRegistry; do NOT hardcode in agents
    capability_specs: list[CapabilitySpec] = []
    # Flat lists kept for backward compat — auto-derived from capability_specs when provided
    capabilities: list[str] = []
    routing_keywords: list[str] = []

    # ── Planner-facing metadata (ported from BaseAgentFramework AgentMeta) ──────
    # Agent-level schema/SLA/tags the DAG planner & router render into the
    # capability menu and validate generated args against. ``input_schema`` /
    # ``output_schema`` map field-name → {"type","required","description","persist"}.
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    sla_ms: int = 10000
    tags: list[str] = []

    @model_validator(mode="after")
    def _sync_from_specs(self) -> AgentInfo:
        """Derive flat lists from capability_specs so existing consumers need no changes."""
        if self.capability_specs:
            if not self.capabilities:
                self.capabilities = [s.id for s in self.capability_specs]
            if not self.routing_keywords:
                self.routing_keywords = [
                    kw for s in self.capability_specs for kw in s.routing_keywords
                ]
            # Adopt the first spec's input/output schema when not set at agent level.
            if not self.input_schema and self.capability_specs[0].input_schema:
                self.input_schema = self.capability_specs[0].input_schema
        return self

    def validate_args(self, args: dict[str, Any]) -> tuple[bool, str | None]:
        """Check that every required input field is present (mirrors AgentMeta).

        ``input_schema`` entries may be plain JSON-Schema-ish dicts; a field is
        required when its spec sets ``"required": true``. Returns ``(ok, error)``.
        """
        if not isinstance(args, dict):
            return False, "args must be an object"
        for name, spec in self.input_schema.items():
            required = bool(spec.get("required")) if isinstance(spec, dict) else False
            if required and name not in args:
                return False, f"missing required arg '{name}'"
        return True, None


@runtime_checkable
class BaseAgent(Protocol):
    """Protocol every agent must satisfy.

    Use structural typing — no inheritance required.  New methods ``enable()``,
    ``disable()``, and ``health_check()`` are part of the contract so the registry
    can manage lifecycle without accessing private attributes.
    """

    @property
    def agent_id(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    @property
    def capabilities(self) -> list[str]: ...

    @property
    def version(self) -> str: ...

    @property
    def enabled(self) -> bool: ...

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> AgentResult: ...

    def get_info(self) -> AgentInfo: ...

    def enable(self) -> None: ...

    def disable(self) -> None: ...

    async def health_check(self) -> str:
        """Return ``'healthy'``, ``'degraded'``, or ``'unhealthy'``."""
        ...
