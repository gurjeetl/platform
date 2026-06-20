"""ExecutorNode — runs the orchestrator's wave decomposition against agents.

Ported from BaseAgentFramework ``orchestrator/executor.py``. Consumes
``state.waves`` (task ids per wave) and runs each wave concurrently via
``asyncio.gather``; the next wave starts only after the current one's tasks have
written to the shared blackboard. Each task is dispatched through the in-process
``AgentRegistry`` — a ``RemoteAgent`` performs the actual A2A JSON-RPC call, so the
executor is identical for local and remote agents. Per-task failures are captured
on the blackboard; the gate decides whether to re-plan.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from typing import Any

from genie.agents import AgentRegistry, AgentTask
from genie.application.blackboard import Blackboard
from genie.application.dag import Plan, Subtask
from genie.application.state import GraphState, ToolCallRecord, ToolResultRecord
from genie.observability.correlation import get_correlation_id
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)

# Matches an upstream-output reference like ${t1.text} or ${t1.view.items.0.id}.
_REF_RE = re.compile(r"\$\{([^}]+)\}")


def _last_user_message(state: GraphState) -> str:
    """Return the most recent user message text (the instruction agents receive)."""
    for m in reversed(state.messages):
        if m.role == "user":
            return m.content
    return ""


class ExecutorNode:
    """Runs the orchestrator's waves against the registry, writing to the blackboard."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        settings: Any,
        event_bus: Any | None = None,
        redis_store: Any | None = None,
    ) -> None:
        self._registry = agent_registry
        self._settings = settings
        self._event_bus = event_bus
        self._redis = redis_store

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        """Execute all waves and emit the blackboard plus API-compat tool records."""
        with node_span("executor") as span:
            result = await self._execute(state)
            with contextlib.suppress(Exception):
                if span is not None:
                    bb = result.get("blackboard", {})
                    span.set_outputs(
                        {
                            "tasks": len(bb),
                            "errors": sum(
                                1 for e in bb.values() if isinstance(e, dict) and "error" in e
                            ),
                        }
                    )
            return result

    async def _execute(self, state: GraphState) -> dict[str, Any]:
        """Run waves in order; tasks within a wave go concurrently, waves serially."""
        plan = Plan(**(state.plan or {}))
        by_id = plan.by_id()
        bb = Blackboard(
            thread_id=state.conversation_id, run_id=state.run_id, redis_store=self._redis
        )

        # Seed from a previous attempt's snapshot so successful tasks don't re-run.
        snapshot = state.blackboard_snapshot or {}
        for tid, entry in snapshot.items():
            if isinstance(entry, dict) and "error" not in entry:
                bb._mem[tid] = entry

        if state.plan_error:
            return {"blackboard": {"_plan_error": {"error": state.plan_error}}}

        # Task ids mirrored to Redis this run (tid, ok) — drives the trace db_ops card.
        written: list[tuple[str, bool]] = []

        wave_ids = state.waves or []
        if not plan.subtasks or not wave_ids:
            logger.info("executor_nothing_to_run")
            return self._finalize(state, plan, bb, written)

        for wave_idx, ids in enumerate(wave_ids):
            wave = [by_id[tid] for tid in ids if tid in by_id]
            todo = [t for t in wave if bb.get(t.id) is None or "error" in (bb.get(t.id) or {})]
            if not todo:
                continue
            logger.info("executor_wave_start", wave=wave_idx, count=len(todo))
            await asyncio.gather(
                *(self._run_task(t, wave_idx, bb, state, written) for t in todo)
            )
            logger.info("executor_wave_done", wave=wave_idx)

        return self._finalize(state, plan, bb, written)

    def _blackboard_db_ops(
        self, state: GraphState, written: list[tuple[str, bool]]
    ) -> list[dict[str, Any]]:
        """One trace ``db_op`` summarizing this run's blackboard mirror to Redis.

        Every ``bb.write``/``bb.write_error`` mirrors the result to
        ``bb:{thread}:{run}:{task_id}`` with a 1h TTL. Reported with ``enabled`` so the
        card honestly shows ``disabled (no-op)`` when Redis isn't configured.
        """
        if not written:
            return []
        enabled = bool(getattr(self._redis, "enabled", False))
        n = len(written)
        return [
            {
                "store": "redis",
                "op": "write",
                "node": "executor",
                "detail": f"blackboard mirrored — {n} task result{'s' if n != 1 else ''} (1h TTL)",
                "code": f"SET bb:{state.conversation_id}:{state.run_id}:* EX 3600",
                "enabled": enabled,
                "hits": [
                    f"bb:{state.conversation_id}:{state.run_id}:{tid} → {'ok' if ok else 'error'}"
                    for tid, ok in written
                ],
            }
        ]

    def _finalize(
        self,
        state: GraphState,
        plan: Plan,
        bb: Blackboard,
        written: list[tuple[str, bool]] | None = None,
    ) -> dict[str, Any]:
        """Surface the blackboard plus API-compat tool records / selected agents."""
        snap = bb.snapshot()
        tool_calls: list[ToolCallRecord] = []
        tool_results: list[ToolResultRecord] = []
        for tid, entry in snap.items():
            sub = plan.by_id().get(tid)
            agent_id = (
                sub.agent_id if sub else (entry.get("agent_id") if isinstance(entry, dict) else tid)
            )
            is_err = isinstance(entry, dict) and "error" in entry
            tool_calls.append(
                ToolCallRecord(call_id=tid, tool_id=agent_id or tid, agent_id=agent_id or tid)
            )
            tool_results.append(
                ToolResultRecord(
                    call_id=tid,
                    tool_id=agent_id or tid,
                    success=not is_err,
                    output=(entry.get("text") if isinstance(entry, dict) and not is_err else None),
                    error=(entry.get("error") if is_err else None),
                )
            )
        selected = [t.agent_id for t in plan.subtasks]
        return {
            "blackboard": snap,
            "selected_agents": selected,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "db_ops": self._blackboard_db_ops(state, written or []),
        }

    # ── Runtime data-passing: resolve ${task_id.path} arg references ────────────
    def _lookup_ref(self, ref: str, bb: Blackboard) -> Any:
        """Walk a dotted ``task_id.path`` into the blackboard (list indices allowed)."""
        parts = ref.strip().split(".")
        cur: Any = bb.get(parts[0])
        for p in parts[1:]:
            if isinstance(cur, dict):
                cur = cur.get(p)
            elif isinstance(cur, list):
                try:
                    cur = cur[int(p)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        return cur

    def _resolve_args(self, value: Any, bb: Blackboard) -> Any:
        """Recursively substitute ``${...}`` references in args with blackboard values.

        A whole-string match (``"${t1.view}"``) is replaced by the raw value so types
        survive; an embedded reference (``"id=${t1.id}"``) is stringified inline.
        Unresolved references are left verbatim.
        """
        if isinstance(value, dict):
            return {k: self._resolve_args(v, bb) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_args(v, bb) for v in value]
        if not isinstance(value, str):
            return value
        # Whole-string reference → return the resolved value untouched (preserves type).
        full = _REF_RE.fullmatch(value.strip())
        if full:
            resolved = self._lookup_ref(full.group(1), bb)
            return value if resolved is None else resolved

        def _sub(m: re.Match) -> str:
            resolved = self._lookup_ref(m.group(1), bb)
            return m.group(0) if resolved is None else str(resolved)

        return _REF_RE.sub(_sub, value)

    async def _run_task(
        self,
        task: Subtask,
        wave_idx: int,
        bb: Blackboard,
        state: GraphState,
        written: list[tuple[str, bool]],
    ) -> None:
        """Dispatch one subtask through the registry with SLA timeout + one retry.

        Resolves arg references, builds the ``AgentTask``, and writes the result (or
        an error entry) to the blackboard. Records each write in ``written`` for the
        trace. Never raises — failures are captured so the gate can decide whether to
        re-plan.
        """
        agent = self._registry.get(task.agent_id)
        if agent is None:
            await bb.write_error(task.id, f"agent_id '{task.agent_id}' not in registry")
            written.append((task.id, False))
            return

        resolved_args = self._resolve_args(task.args or {}, bb)
        # Identity fields ride on the typed AgentTask; context carries only the
        # untyped routing extras (args/run_id/blackboard).
        context = {
            "run_id": state.run_id,
            "args": resolved_args,
            "blackboard": bb.snapshot(),
        }
        agent_task = AgentTask(
            task_id=task.id,
            agent_id=task.agent_id,
            conversation_id=state.conversation_id,
            correlation_id=get_correlation_id() or state.correlation_id,
            instruction=_last_user_message(state),
            context=context,
        )

        timeout_s = max(0.1, task.sla_ms / 1000.0)
        last_error: str | None = None
        for attempt in range(2):  # 1 retry
            try:
                result = await asyncio.wait_for(
                    agent.execute(agent_task), timeout=timeout_s
                )
                if not result.success:
                    last_error = result.error or "agent reported failure"
                    logger.warning(
                        "executor_task_retry", task=task.id, attempt=attempt, error=last_error
                    )
                    continue
                payload: dict[str, Any] = {"agent_id": task.agent_id, "text": result.output}
                view = (result.data or {}).get("view")
                if view:
                    payload["view"] = view
                await bb.write(task.id, payload)
                written.append((task.id, True))
                with contextlib.suppress(Exception):
                    if self._event_bus is not None:
                        from genie.platform.event_bus import TOPIC_AGENT_EXECUTED

                        await self._event_bus.publish(
                            TOPIC_AGENT_EXECUTED,
                            payload={
                                "agent_id": task.agent_id,
                                "task_id": task.id,
                                "success": True,
                                "conversation_id": state.conversation_id,
                            },
                            correlation_id=state.correlation_id,
                        )
                return
            except TimeoutError:
                last_error = f"agent timed out after {task.sla_ms}ms"
                logger.warning(
                    "executor_task_retry", task=task.id, attempt=attempt, error=last_error
                )
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                logger.warning(
                    "executor_task_retry", task=task.id, attempt=attempt, error=last_error
                )

        await bb.write_error(task.id, last_error or "unknown failure")
        written.append((task.id, False))
