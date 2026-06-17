"""OrchestratorNode — decomposes the plan DAG into dependency waves.

Ported from BaseAgentFramework ``orchestrator/orchestrator.py``. Runs no agents —
it computes the execution waves (Kahn's algorithm) and hands the decomposition to
the Executor, so decomposition and execution are observable separately in the trace.

It also retains Genie's optional Human-in-the-Loop (HITL) gate: when ``enable_hitl``
is on (and not auto-approving), it flags ``requires_approval`` so the graph pauses at
the ``human_approval`` node. HITL is OFF by default, so the gate is transparent.
"""

from __future__ import annotations

import contextlib
from typing import Any

from genie.application.dag import Plan
from genie.application.state import GraphState
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)


class OrchestratorNode:
    def __init__(self, settings: Any) -> None:
        self._settings = settings

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        with node_span("orchestrator") as span:
            plan = Plan(**(state.plan or {}))

            if not plan.subtasks:
                logger.info("orchestrator_empty_plan")
                return {"waves": [], "plan_error": None, "requires_approval": False}

            try:
                waves = plan.waves()
                wave_ids = [[t.id for t in wave] for wave in waves]
                plan_error = None
            except Exception as exc:  # noqa: BLE001 — cycle / unknown dep
                logger.error("orchestrator_dag_invalid", error=str(exc))
                wave_ids = []
                plan_error = str(exc)

            # Optional HITL gate (off by default).
            requires_approval = (
                getattr(self._settings, "enable_hitl", False)
                and not getattr(self._settings, "hitl_auto_approve", True)
                and bool(plan.subtasks)
            )
            hitl_prompt: str | None = None
            if requires_approval:
                hitl_prompt = (
                    "Human approval required before executing this plan:\n"
                    f"agents: {[t.agent_id for t in plan.subtasks]}\n"
                    "Please approve or reject this action."
                )

            with contextlib.suppress(Exception):
                if span is not None:
                    span.set_outputs(
                        {
                            "wave_count": len(wave_ids),
                            "requires_approval": requires_approval,
                        }
                    )

            logger.info(
                "orchestrator_decomposed",
                wave_count=len(wave_ids),
                task_count=sum(len(w) for w in wave_ids),
            )
            return {
                "waves": wave_ids,
                "plan_error": plan_error,
                "requires_approval": requires_approval,
                "hitl_prompt": hitl_prompt,
            }
