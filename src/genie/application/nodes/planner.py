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
from genie.application.state import GraphState, Message
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)

_PLAN_SCHEMA_HINT = (
    "Respond ONLY with valid JSON in this exact shape:\n"
    '{"subtasks":['
    '{"id":"t1","agent_id":"<one of the agents>","args":{...},"depends_on":[],"sla_ms":10000}'
    "]}\n"
    "No extra text, no markdown fences, no explanation — just the JSON."
)


def _last_user_message(state: GraphState) -> str:
    for m in reversed(state.messages):
        if m.role == "user":
            return m.content
    return ""


class PlannerNode:
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
        return [a for a in self._registry.list_all() if a.enabled]

    def _build_system_prompt(
        self, state: GraphState, agents: list, recall: list[dict], facts: dict[str, str]
    ) -> str:
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
        return (
            "You are a planning agent. Look at the user's request and split it into "
            "one or more SUBTASKS, where each subtask is assigned to exactly one "
            "registered agent below. Match user intent to agent capability + tags.\n\n"
            "REGISTERED AGENTS:\n"
            f"{menu}\n\n"
            "How to match:\n"
            "- Read each agent's capability description AND tags. Phrasing like 'show', "
            "'list', 'tell me about', 'top N', 'forecast', 'report' are common synonyms; "
            "match the agent that performs the underlying capability.\n"
            "- Required inputs are marked with an asterisk (*). Optional inputs may be "
            "omitted — when an agent works fine with empty args, pass {}.\n"
            "- depends_on=[] means a subtask can run independently. Populate depends_on "
            "ONLY when one task literally needs another task's output as input.\n"
            "- CHAINING: to feed an earlier subtask's result into a later one, put a "
            "reference in the later subtask's args AND add <id> to its depends_on. Use "
            "${<id>.text} for the task's text output, or ${<id>.view.<path>} for a field "
            "of its structured view. References are replaced at run time.\n"
            "- Only return an empty subtasks list when truly NO registered agent can "
            "address the request.\n\n"
            "Examples:\n"
            'User: "What\'s the weather in Paris?"\n'
            '→ {"subtasks":[{"id":"t1","agent_id":"weather","args":{"location":"paris"},"depends_on":[]}]}\n\n'
            'User: "Show me the top 5 outages."\n'
            '→ {"subtasks":[{"id":"t1","agent_id":"outage","args":{},"depends_on":[]}]}\n\n'
            'User: "Weather in Tokyo and the top outages."\n'
            '→ {"subtasks":['
            '{"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},'
            '{"id":"t2","agent_id":"outage","args":{},"depends_on":[]}'
            "]}\n\n"
            'User: "Top 5 outages, then full details of the first one." (chained)\n'
            '→ {"subtasks":['
            '{"id":"t1","agent_id":"outage","args":{},"depends_on":[]},'
            '{"id":"t2","agent_id":"outage","args":{"outage_id":"${t1.view.items.0.id}"},"depends_on":["t1"]}'
            "]}\n\n"
            "Output rules:\n"
            "- Use only agent_ids from the list above.\n"
            "- Give each subtask a stable id like 't1','t2'.\n"
            "- City names go in args as lowercase strings.\n"
            f"{recall_block}"
            f"{facts_block}"
            f"{replan_block}\n\n"
            f"{_PLAN_SCHEMA_HINT}"
        )

    def _build_plan(self, parsed: dict, agents: list) -> Plan:
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

    async def _plan(self, state: GraphState) -> dict[str, Any]:
        user_msg = _last_user_message(state)
        agents = self._agents()

        recall: list[dict] = []
        facts: dict[str, str] = {}
        if self._memory is not None and user_msg:
            with contextlib.suppress(Exception):
                recall = await self._memory.recall(state.conversation_id, user_msg)
            with contextlib.suppress(Exception):
                facts = await self._memory.query_facts(state.conversation_id)
            if len(facts) > self._max_facts:
                facts = dict(list(facts.items())[: self._max_facts])

        if not agents:
            return {
                "plan_error": "No agents registered; cannot build a plan.",
                "plan": {"subtasks": []},
                "blackboard": {},
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
            return {"error": "Planner could not reach the model.", "plan": {"subtasks": []}}

        parsed = extract_json(raw)
        if parsed is None:
            logger.error("planner_parse_failed", raw=(raw or "")[:300])
            return {
                "error": "Planner could not parse a plan from the model.",
                "plan": {"subtasks": []},
            }

        plan = self._build_plan(parsed, agents)
        if not plan.subtasks:
            logger.info("planner_empty_plan")
            return {"plan": plan.model_dump(), "agent_versions": {}, "blackboard": {}}

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
        }
