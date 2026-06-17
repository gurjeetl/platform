"""CompletionGateNode — decides: synthesize the final answer, or re-plan?

Ported from BaseAgentFramework ``gate/completion_gate.py``. Inspects the blackboard
against the plan; if tasks are missing or errored and the re-plan budget is not
exhausted, it routes back to the Planner. Otherwise it proceeds to the Synthesizer
(possibly with a partial answer).
"""

from __future__ import annotations

import contextlib
from typing import Any

from genie.application.dag import Plan
from genie.application.state import GraphState
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)


class CompletionGateNode:
    def __init__(self, settings: Any | None = None) -> None:
        self._settings = settings
        self._max_replans = int(getattr(settings, "max_replans", 3)) if settings else 3

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        with node_span("completion_gate") as span:
            plan = Plan(**(state.plan or {}))
            blackboard = state.blackboard or {}
            replan_count = state.replan_count or 0
            max_replans = state.metadata.get("max_replans", self._max_replans)

            all_present = all(t.id in blackboard for t in plan.subtasks)
            error_keys = [
                tid
                for tid, entry in blackboard.items()
                if isinstance(entry, dict) and "error" in entry
            ]
            partial = bool(error_keys)

            budget_left = replan_count < max_replans
            empty_plan = len(plan.subtasks) == 0
            should_replan = (not empty_plan) and (not all_present or partial) and budget_left

            with contextlib.suppress(Exception):
                if span is not None:
                    span.set_outputs(
                        {
                            "action": "replan" if should_replan else "synthesize",
                            "error_count": len(error_keys),
                            "replan_count": replan_count,
                        }
                    )

            if should_replan:
                missing = [t.id for t in plan.subtasks if t.id not in blackboard]
                logger.info(
                    "gate_replan",
                    replan_count=replan_count + 1,
                    missing=missing,
                    errored=error_keys,
                )
                meta = dict(state.metadata)
                meta["gate_action"] = "replan"
                return {
                    "metadata": meta,
                    "replan_count": replan_count + 1,
                    "replan_reason": f"missing tasks: {missing}; errored tasks: {error_keys}",
                    "blackboard_snapshot": dict(blackboard),
                    "partial": partial,
                }

            meta = dict(state.metadata)
            meta["gate_action"] = "synthesize"
            logger.info("gate_synthesize", partial=partial)
            return {"metadata": meta, "partial": partial}
