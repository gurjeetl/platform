"""RemoteAgent — a discovered distributed agent surfaced through the BaseAgent protocol.

This is the bridge that lets Genie keep its in-process pipeline (planner selects
from the ``AgentRegistry``; executor calls ``agent.execute``) while agents actually
run as separate services: ``RemoteAgent`` is built from a discovered ``AgentMeta``
and its ``execute`` dispatches an A2A JSON-RPC ``message/send`` to the agent's
endpoint, mapping the reply into an ``AgentResult``.
"""

from __future__ import annotations

from typing import Any

from genie.a2a import A2AClient, A2AError, get_data, get_text
from genie.agents.base import AgentInfo, AgentResult, AgentTask, CapabilitySpec
from genie.discovery.agent_meta import AgentMeta
from genie.observability.logging import get_logger

logger = get_logger(__name__)


class RemoteAgent:
    """Adapter implementing ``genie.agents.base.BaseAgent`` over A2A transport."""

    def __init__(self, meta: AgentMeta, client: A2AClient | None = None) -> None:
        """Wrap a discovered ``AgentMeta``; enabled iff its registry status is active."""
        self._meta = meta
        self._client = client or A2AClient()
        self._enabled = meta.status == "active"

    # ── BaseAgent protocol ──────────────────────────────────────────────────────
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
        input_schema = {k: v.model_dump() for k, v in self._meta.input_schema.items()}
        output_schema = {k: v.model_dump() for k, v in self._meta.output_schema.items()}
        spec = CapabilitySpec(
            id=self._meta.agent_id,
            display_name=self._meta.agent_id,
            description=self._meta.description,
            routing_keywords=list(self._meta.capability_tags),
            input_schema=input_schema,
        )
        return AgentInfo(
            agent_id=self._meta.agent_id,
            name=self._meta.agent_id,
            description=self._meta.description,
            version=self._meta.version,
            enabled=self._enabled,
            capability_specs=[spec],
            input_schema=input_schema,
            output_schema=output_schema,
            sla_ms=self._meta.sla_ms,
            tags=list(self._meta.capability_tags),
        )

    # ── Execution over A2A ──────────────────────────────────────────────────────
    async def execute(self, task: AgentTask, context: dict[str, Any]) -> AgentResult:
        """Dispatch the task over A2A; map the reply (or any error) into an AgentResult."""
        args = (task.context or {}).get("args", {})
        try:
            reply = await self._client.send(
                self._meta.endpoint or "",
                self._meta.agent_id,
                args,
                {
                    "task_id": (task.context or {}).get("task_id") or task.task_id,
                    "run_id": (task.context or {}).get("run_id"),
                    "thread_id": (task.context or {}).get("thread_id") or task.conversation_id,
                    "correlation_id": task.correlation_id,
                    "blackboard": (task.context or {}).get("blackboard") or {},
                },
                sla_ms=self._meta.sla_ms,
            )
        except A2AError as exc:
            logger.warning("remote_agent_a2a_error", agent_id=self._meta.agent_id, error=str(exc))
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._meta.agent_id,
                success=False,
                output="",
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("remote_agent_error", agent_id=self._meta.agent_id, error=str(exc))
            return AgentResult(
                task_id=task.task_id,
                agent_id=self._meta.agent_id,
                success=False,
                output="",
                error=str(exc),
            )

        view = (get_data(reply) or {}).get("view")
        return AgentResult(
            task_id=task.task_id,
            agent_id=self._meta.agent_id,
            success=True,
            output=get_text(reply),
            data={"view": view} if view else {},
        )
