"""PlannerNode — splits the user prompt into a DAG of subtasks.

Ported from BaseAgentFramework ``planner/planner_agent.py``. Pure LLM planning over
the live registry-derived capability menu (no MCP tools). Optionally enriches the
prompt with semantic recall + structured facts from the memory subsystem (wired in
Phase 3; no-ops when ``memory`` is None).
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from genie.agents.registry import AgentRegistry
from genie.application.dag import Plan, Subtask
from genie.application.parsing import extract_json, normalize_agent_id, render_capability_menu
from genie.application.prompts import (
    DEFAULT_SYSTEM_CONTEXT,
    DEFAULT_SYSTEM_PROMPT,
    PLAN_SCHEMA_HINT,
    PLANNER_PROMPT,
)
from genie.application.state import GraphState, Message
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)


def _last_user_message(state: GraphState) -> str:
    """Return the most recent user message text — the request being planned."""
    for m in reversed(state.messages):
        if m.role == "user":
            return m.content
    return ""


class PlannerNode:
    """LLM-plans the user request into a DAG of subtasks over the live agent menu."""

    def __init__(
        self,
        agent_registry: AgentRegistry,
        settings: Any,
        llm_provider: Any,
        memory: Any | None = None,
    ) -> None:
        self._registry = agent_registry
        self._settings = settings
        self._llm = llm_provider
        self._memory = memory
        self._max_facts = int(getattr(settings, "planner_max_facts", 40))

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        """Build the plan and surface its subtask count to the trace span."""
        with node_span("planner") as span:
            with contextlib.suppress(Exception):
                if span is not None:
                    span.set_inputs({"message": _last_user_message(state)[:200]})
            result = await self._plan(state)
            with contextlib.suppress(Exception):
                if span is not None:
                    plan = result.get("plan") or {}
                    span.set_outputs({"subtask_count": len(plan.get("subtasks", []))})
            return result

    def _agents(self) -> list:
        """The currently enabled agents the planner is allowed to assign work to."""
        return [a for a in self._registry.list_all() if a.enabled]

    def _build_system_prompt(
        self, state: GraphState, agents: list, recall: list[dict], facts: dict[str, str]
    ) -> str:
        """Assemble the planning prompt: capability menu, memory recall, replan context."""
        menu = render_capability_menu(agents)
        recall_block = ""
        if recall:
            lines = "\n".join(f"- {str(h.get('content', '')).strip()}" for h in recall)
            recall_block = (
                "\n\nRELEVANT PAST CONTEXT (semantic recall from long-term memory — "
                "use only if it helps; do not invent facts):\n" + lines
            )
        facts_block = ""
        if facts:
            lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
            facts_block = (
                "\n\nKNOWN FACTS (structured recall — use only if it helps; "
                "do not invent facts):\n" + lines
            )
        replan_block = ""
        snapshot = state.blackboard_snapshot
        reason = state.replan_reason
        if snapshot or reason:
            replan_block = (
                "\n\nRE-PLAN CONTEXT (previous attempt's blackboard + reason):\n"
                f"reason: {reason or '(none)'}\n"
                f"snapshot: {json.dumps(snapshot, default=str)[:2000]}\n"
                "Adjust the plan to recover from the errors above."
            )
        return PLANNER_PROMPT.safe_substitute(
            system_prompt=self._settings.app_system_prompt or DEFAULT_SYSTEM_PROMPT,
            system_context=self._settings.app_system_context or DEFAULT_SYSTEM_CONTEXT,
            capability_menu=menu,
            recall_block=recall_block,
            facts_block=facts_block,
            replan_block=replan_block,
            schema_hint=PLAN_SCHEMA_HINT,
        )

    def _build_plan(self, parsed: dict, agents: list) -> Plan:
        """Validate the LLM's raw subtasks into a Plan; drop unknown agents / bad args."""
        raw_subtasks: list[dict[str, Any]] = parsed.get("subtasks", []) or []
        infos = {a.agent_id: a for a in agents}
        known_ids = set(infos)
        clean: list[Subtask] = []
        for st in raw_subtasks:
            raw_id = st.get("agent_id")
            agent_id = normalize_agent_id(raw_id, known_ids)
            info = infos.get(agent_id) if agent_id else None
            if info is None:
                logger.warning("planner_unknown_agent_id", raw=str(raw_id))
                continue
            args = st.get("args") or {}
            ok, err = info.validate_args(args)
            if not ok:
                logger.warning("planner_invalid_args", agent_id=agent_id, error=err)
                continue
            clean.append(
                Subtask(
                    id=str(st.get("id") or f"t{len(clean) + 1}"),
                    agent_id=agent_id,
                    agent_version=info.version,
                    args=args,
                    depends_on=list(st.get("depends_on") or []),
                    sla_ms=int(st.get("sla_ms") or info.sla_ms),
                )
            )
        return Plan(subtasks=clean)

    def _recall_db_ops(
        self, recall: list[dict], facts: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Trace ``db_ops`` for the memory the planner read to enrich its prompt.

        One Milvus ``search`` (semantic recall) and one Mongo ``read`` (structured
        facts), each reported with ``enabled`` so the card honestly shows
        ``disabled (no-op)`` when that backend isn't configured.
        """
        vector = getattr(self._memory, "vector", None)
        vector_enabled = bool(getattr(vector, "enabled", False))
        mongo = getattr(self._memory, "mongo", None)
        mongo_enabled = bool(getattr(mongo, "enabled", False))
        n_recall, n_facts = len(recall), len(facts)
        return [
            {
                "store": "milvus",
                "op": "search",
                "node": "planner",
                "detail": (
                    f"semantic recall — {n_recall} hit{'s' if n_recall != 1 else ''}"
                    if vector_enabled
                    else "semantic recall (Milvus disabled — no-op)"
                ),
                "code": f"milvus.search(long_term_memory, top_k=5, filter=thread)",
                "enabled": vector_enabled,
                "hits": [str(h.get("content", "")).strip()[:80] for h in recall],
            },
            {
                "store": "mongodb",
                "op": "read",
                "node": "planner",
                "detail": (
                    f"facts recall — {n_facts} fact{'s' if n_facts != 1 else ''}"
                    if mongo_enabled
                    else "facts recall (Mongo disabled — no-op)"
                ),
                "code": "db.agent_facts.find({scope: {$in: [global, session]}})",
                "enabled": mongo_enabled,
                "hits": [f"{k}: {v}" for k, v in list(facts.items())[:10]],
            },
        ]

    async def _plan(self, state: GraphState) -> dict[str, Any]:
        """Recall memory, prompt the LLM, parse + validate, and return the plan dict."""
        user_msg = _last_user_message(state)
        agents = self._agents()

        recall: list[dict] = []
        facts: dict[str, str] = {}
        db_ops: list[dict[str, Any]] = []
        if self._memory is not None and user_msg:
            with contextlib.suppress(Exception):
                recall = await self._memory.recall(state.conversation_id, user_msg)
            with contextlib.suppress(Exception):
                facts = await self._memory.query_facts(state.conversation_id)
            if len(facts) > self._max_facts:
                facts = dict(list(facts.items())[: self._max_facts])
            db_ops = self._recall_db_ops(recall, facts)

        if not agents:
            return {
                "plan_error": "No agents registered; cannot build a plan.",
                "plan": {"subtasks": []},
                "blackboard": {},
                "db_ops": db_ops,
            }

        prompt = self._build_system_prompt(state, agents, recall, facts)
        try:
            response = await self._llm.complete(
                [Message(role="system", content=prompt), Message(role="user", content=user_msg)],
                max_tokens=800,
                temperature=0.0,
            )
            raw = response.content
        except Exception as exc:  # noqa: BLE001
            logger.error("planner_llm_failed", error=str(exc))
            return {
                "error": "Planner could not reach the model.",
                "plan": {"subtasks": []},
                "db_ops": db_ops,
            }

        parsed = extract_json(raw)
        if parsed is None:
            logger.error("planner_parse_failed", raw=(raw or "")[:300])
            return {
                "error": "Planner could not parse a plan from the model.",
                "plan": {"subtasks": []},
                "db_ops": db_ops,
            }

        plan = self._build_plan(parsed, agents)
        if not plan.subtasks:
            logger.info("planner_empty_plan")
            return {
                "plan": plan.model_dump(),
                "agent_versions": {},
                "blackboard": {},
                "db_ops": db_ops,
            }

        agent_versions = {t.id: t.agent_version for t in plan.subtasks}
        logger.info(
            "planner_plan_built",
            count=len(plan.subtasks),
            agent_ids=[t.agent_id for t in plan.subtasks],
        )
        return {
            "plan": plan.model_dump(),
            "agent_versions": agent_versions,
            "blackboard": {},
            "blackboard_snapshot": None,
            "replan_reason": None,
            "db_ops": db_ops,
        }
