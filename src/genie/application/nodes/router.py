"""RouterNode — cheap, registry-aware intent triage in front of the Planner.

Ported from BaseAgentFramework ``router/router_agent.py``. Decides one of three
routes before the expensive Planner runs:
  - ``fast``     — request maps to exactly one agent with fillable args → build a
                   one-task plan + waves and jump straight to the Executor.
  - ``chitchat`` — greeting / thanks / meta, no agent needed → straight to the
                   Synthesizer (its empty-plan path returns a clarification).
  - ``plan``     — anything ambiguous or multi-intent → the full Planner.

It **fails open to ``plan``** on any doubt, registry outage, or LLM/parse failure,
so it can only ever speed things up, never reduce what the system can answer.
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

from genie.agents.registry import AgentRegistry
from genie.application.dag import Plan, Subtask
from genie.application.parsing import extract_json, normalize_agent_id, render_capability_menu
from genie.application.prompts import (
    DEFAULT_SYSTEM_CONTEXT,
    DEFAULT_SYSTEM_PROMPT,
    ROUTER_PROMPT,
    ROUTER_SCHEMA_HINT,
)
from genie.application.state import GraphState, Message
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)

# Cheap, pre-LLM signal that a prompt is clearly multi-intent (always falls through
# to the planner anyway). Conservative additive connectors only.
_DEFAULT_MULTI_INTENT_PATTERN = r"(?i)\b(also|as well as|and also|additionally|moreover)\b|;"


def _last_user_message(state: GraphState) -> str:
    """Return the most recent user message text — the request being triaged."""
    for m in reversed(state.messages):
        if m.role == "user":
            return m.content
    return ""


class RouterNode:
    """Cheap intent triage that picks fast/chitchat/plan, failing open to plan."""

    def __init__(self, llm_provider: Any, agent_registry: AgentRegistry, settings: Any) -> None:
        self._llm = llm_provider
        self._registry = agent_registry
        self._settings = settings
        # A "fast" match below this confidence is downgraded to the safe "plan" route.
        self._min_confidence = float(getattr(settings, "router_min_confidence", 0.7))
        self._multi_intent_re = re.compile(_DEFAULT_MULTI_INTENT_PATTERN)

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        """Choose a route and record it on the trace span."""
        with node_span("router") as span:
            with contextlib.suppress(Exception):
                if span is not None:
                    span.set_inputs({"message": _last_user_message(state)[:200]})
            result = await self._route(state)
            with contextlib.suppress(Exception):
                if span is not None:
                    span.set_outputs({"route": result.get("route", "")})
            return result

    def _agents(self) -> list:
        """The currently enabled agents the router can match a fast route against."""
        return [a for a in self._registry.list_all() if a.enabled]

    def _build_system_prompt(self, agents: list) -> str:
        """Build the routing prompt: the three routes plus the live capability menu."""
        return ROUTER_PROMPT.safe_substitute(
            system_prompt=self._settings.app_system_prompt or DEFAULT_SYSTEM_PROMPT,
            system_context=self._settings.app_system_context or DEFAULT_SYSTEM_CONTEXT,
            capability_menu=render_capability_menu(agents),
            schema_hint=ROUTER_SCHEMA_HINT,
        )

    async def _route(self, state: GraphState) -> dict[str, Any]:
        """Decide the route; any registry/LLM/parse failure falls back to ``plan``."""
        user_msg = _last_user_message(state)
        agents = self._agents()

        if not agents:
            return self._route_plan(reason="no_agents")

        # Clearly multi-intent prompts fall through to the planner regardless.
        if user_msg and self._multi_intent_re.search(user_msg):
            return self._route_plan(reason="multi_intent_regex")

        prompt = self._build_system_prompt(agents)
        try:
            response = await self._llm.complete(
                [Message(role="system", content=prompt), Message(role="user", content=user_msg)],
                max_tokens=200,
                temperature=0.0,
            )
            raw = response.content
        except Exception as exc:  # noqa: BLE001
            logger.warning("router_llm_failed", error=str(exc))
            return self._route_plan(reason="llm_failed")

        parsed = extract_json(raw) or {}
        route = str(parsed.get("route") or "plan").strip().lower()

        if route == "chitchat":
            return self._route_chitchat()
        if route == "fast":
            fast = self._try_fast(parsed, agents)
            if fast is not None:
                return fast
        return self._route_plan(reason="default")

    def _try_fast(self, parsed: dict, agents: list) -> dict[str, Any] | None:
        """Build a one-task plan if the LLM's fast pick resolves, clears confidence,
        and validates its args; otherwise None so the caller falls back to ``plan``."""
        by_id = {a.agent_id: a for a in agents}
        agent_id = normalize_agent_id(parsed.get("agent_id"), set(by_id))
        info = by_id.get(agent_id) if agent_id else None
        if info is None:
            return None
        try:
            confidence = float(parsed.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < self._min_confidence:
            return None
        args = parsed.get("args") or {}
        ok, _err = info.validate_args(args)
        if not ok:
            return None
        subtask = Subtask(
            id="t1",
            agent_id=info.agent_id,
            agent_version=info.version,
            args=args,
            depends_on=[],
            sla_ms=info.sla_ms,
        )
        plan = Plan(subtasks=[subtask])
        logger.info("router_decision", route="fast", agent_id=info.agent_id, confidence=confidence)
        return {
            "route": "fast",
            "plan": plan.model_dump(),
            "agent_versions": {"t1": info.version},
            "waves": [["t1"]],
            "blackboard": {},
            "blackboard_snapshot": None,
        }

    def _route_chitchat(self) -> dict[str, Any]:
        """Route to the synthesizer with an empty plan (yields a clarification)."""
        logger.info("router_decision", route="chitchat")
        return {
            "route": "chitchat",
            "plan": {"subtasks": []},
            "agent_versions": {},
            "blackboard": {},
        }

    def _route_plan(self, *, reason: str = "default") -> dict[str, Any]:
        """The safe default: hand off to the full Planner (``reason`` aids tracing)."""
        logger.info("router_decision", route="plan", reason=reason)
        return {"route": "plan"}
