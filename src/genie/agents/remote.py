"""RemoteAgent — a discovered distributed agent surfaced through the AgentProtocol.

This is the bridge that lets Genie keep its in-process pipeline (planner selects
from the ``AgentRegistry``; executor calls ``agent.execute``) while agents actually
run as separate services: ``RemoteAgent`` is built from a discovered ``AgentMeta``
and its ``execute`` dispatches an A2A JSON-RPC ``message/send`` to the agent's
endpoint, mapping the reply into an ``AgentResult``.
"""

from __future__ import annotations

from genie.a2a import A2AClient, A2AError, get_data, get_text
from genie.agents.base import AgentInfo, AgentResult, AgentTask, CapabilitySpec
from genie.discovery.agent_meta import AgentMeta
from genie.observability.logging import get_logger

logger = get_logger(__name__)


def agent_info_from_meta(meta: AgentMeta, *, enabled: bool | None = None) -> AgentInfo:
    """Project a discovered ``AgentMeta`` into the registry's ``AgentInfo`` card.

    Single source of truth for the meta → card mapping (was inline in
    ``RemoteAgent.get_info``). ``enabled`` defaults to the meta's active status.
    """
    input_schema = {k: v.model_dump() for k, v in meta.input_schema.items()}
    output_schema = {k: v.model_dump() for k, v in meta.output_schema.items()}
    spec = CapabilitySpec(
        id=meta.agent_id,
        display_name=meta.agent_id,
        description=meta.description,
        routing_keywords=list(meta.capability_tags),
        input_schema=input_schema,
    )
    return AgentInfo(
        agent_id=meta.agent_id,
        name=meta.agent_id,
        description=meta.description,
        version=meta.version,
        enabled=meta.status == "active" if enabled is None else enabled,
        capability_specs=[spec],
        input_schema=input_schema,
        output_schema=output_schema,
        sla_ms=meta.sla_ms,
        tags=list(meta.capability_tags),
    )


class RemoteAgent:
    """Adapter implementing ``genie.agents.base.AgentProtocol`` over A2A transport."""

    def __init__(self, meta: AgentMeta, client: A2AClient | None = None) -> None:
        """Wrap a discovered ``AgentMeta``; enabled iff its registry status is active."""
        self._meta = meta
        self._client = client or A2AClient()
        self._enabled = meta.status == "active"

    # ── AgentProtocol surface ─────────────────────────────────────────────────────
    @property
    def agent_id(self) -> str:
        return self._meta.agent_id

    @property
    def name(self) -> str:
        return self._meta.agent_id

    @property
    def description(self) -> str:
        return self._meta.description

    @property
    def capabilities(self) -> list[str]:
        return list(self._meta.capability_tags) or [self._meta.agent_id]

    @property
    def version(self) -> str:
        return self._meta.version

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    async def health_check(self) -> str:
        # Liveness is tracked by the registry's TTL; presence in discovery == healthy.
        return "healthy"

    def get_info(self) -> AgentInfo:
        """Project the discovered ``AgentMeta`` into the registry's ``AgentInfo`` card."""
        return agent_info_from_meta(self._meta, enabled=self._enabled)

    # ── Execution over A2A ──────────────────────────────────────────────────────
    def _failure(self, task: AgentTask, exc: Exception) -> AgentResult:
        """Log and wrap a dispatch failure as a failed ``AgentResult``."""
        event = "remote_agent_a2a_error" if isinstance(exc, A2AError) else "remote_agent_error"
        logger.warning(event, agent_id=self._meta.agent_id, error=str(exc))
        return AgentResult(
            task_id=task.task_id,
            agent_id=self._meta.agent_id,
            success=False,
            output="",
            error=str(exc),
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        """Dispatch the task over A2A; map the reply (or any error) into an AgentResult.

        Identity fields come from the typed ``task`` (``task_id``/``conversation_id``/
        ``correlation_id``); only the untyped routing extras (``args``/``run_id``/
        ``blackboard``) are read from ``task.context``.
        """
        ctx = task.context or {}
        try:
            reply = await self._client.send(
                self._meta.endpoint or "",
                self._meta.agent_id,
                ctx.get("args", {}),
                {
                    "task_id": task.task_id,
                    "run_id": ctx.get("run_id"),
                    "thread_id": task.conversation_id,
                    "correlation_id": task.correlation_id,
                    "blackboard": ctx.get("blackboard") or {},
                },
                sla_ms=self._meta.sla_ms,
            )
        except Exception as exc:  # noqa: BLE001 — all failures degrade to a failed result
            return self._failure(task, exc)

        view = (get_data(reply) or {}).get("view")
        return AgentResult(
            task_id=task.task_id,
            agent_id=self._meta.agent_id,
            success=True,
            output=get_text(reply),
            data={"view": view} if view else {},
        )
