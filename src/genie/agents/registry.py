"""AgentRegistry — registration, discovery, enable/disable, and live health tracking."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from genie.agents.base import AgentInfo, BaseAgent
from genie.platform.errors import ErrorCode, GenieError
from genie.observability.logging import get_logger

_HEALTH_CHECK_INTERVAL_SECONDS = 30


class AgentRegistry:
    """Central index of all registered agents.

    Improvements over the previous version:
    - Rejects duplicate agent IDs at registration time (Gap 5)
    - Warns when two agents share a capability string (Gap 2)
    - Maintains a live health cache updated by a background coroutine (Gap 1)
    - ``find_by_capability`` skips unhealthy agents (Gap 1)
    - ``list_all`` injects live health from the cache, not the agent's own hardcoded value (Gap 1)
    - ``enable`` / ``disable`` call the agent's own methods instead of mutating private attrs (Gap 8)
    - ``unregister`` allows runtime removal without a server restart (Gap 6)
    """

    def __init__(self) -> None:
        """Start empty; the health cache is populated as agents register."""
        self._agents: dict[str, BaseAgent] = {}
        # Separate health cache — owned by the registry, not by agents.
        # Agents must NOT set health themselves; the registry polls and writes here.
        self._health: dict[str, str] = {}
        self._logger = get_logger(__name__)
        self._health_task: Optional[asyncio.Task] = None

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, agent: BaseAgent) -> None:
        """Index an agent by id; reject duplicates and warn on capability collisions."""
        # Gap 5: reject duplicate agent IDs — silent overwrite hides bugs
        if agent.agent_id in self._agents:
            raise ValueError(
                f"Agent ID '{agent.agent_id}' is already registered by "
                f"{type(self._agents[agent.agent_id]).__name__}. "
                "Each agent must have a unique agent_id."
            )
        # Gap 2: warn when two agents share a capability string
        for cap in agent.capabilities:
            colliders = [a.agent_id for a in self._agents.values() if cap in a.capabilities]
            if colliders:
                self._logger.warning(
                    "agent_capability_collision",
                    capability=cap,
                    new_agent=agent.agent_id,
                    existing_agents=colliders,
                )
        self._agents[agent.agent_id] = agent
        self._health[agent.agent_id] = "healthy"
        self._logger.info(
            "agent_registered",
            agent_id=agent.agent_id,
            capabilities=list(agent.capabilities),
        )

    def unregister(self, agent_id: str) -> None:
        """Gap 6: remove a running agent without a server restart."""
        if agent_id not in self._agents:
            raise GenieError(ErrorCode.NOT_FOUND, f"Agent '{agent_id}' not found")
        del self._agents[agent_id]
        self._health.pop(agent_id, None)
        self._logger.info("agent_unregistered", agent_id=agent_id)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get(self, agent_id: str) -> BaseAgent | None:
        """Return the agent for ``agent_id``, or None if not registered."""
        return self._agents.get(agent_id)

    def require(self, agent_id: str) -> BaseAgent:
        """Like ``get`` but raise NOT_FOUND when the agent is missing."""
        agent = self.get(agent_id)
        if agent is None:
            raise GenieError(ErrorCode.NOT_FOUND, f"Agent '{agent_id}' not found")
        return agent

    def find_by_capability(self, capability: str) -> list[BaseAgent]:
        """Return enabled, non-unhealthy agents that declare the given capability.

        Gap 1: an agent whose health_check recently returned 'unhealthy' is excluded
        so the Planner never dispatches to a known-dead backend.
        'degraded' agents are still included — they may still serve requests.
        """
        return [
            a
            for a in self._agents.values()
            if capability in a.capabilities
            and a.enabled
            and self._health.get(a.agent_id, "healthy") != "unhealthy"
        ]

    def list_all(self) -> list[AgentInfo]:
        """Return AgentInfo for every registered agent with live health injected.

        Gap 1: the returned health value comes from the registry's health cache,
        not from the agent's own ``get_info()`` (which used to always return 'healthy').
        """
        infos = []
        for a in self._agents.values():
            info = a.get_info()
            live_health = self._health.get(a.agent_id, "healthy")
            infos.append(info.model_copy(update={"health": live_health}))
        return infos

    # ── Lifecycle management ──────────────────────────────────────────────────

    def enable(self, agent_id: str) -> None:
        """Gap 8: call the agent's own enable() instead of mutating a private attr."""
        agent = self.require(agent_id)
        if hasattr(agent, "enable") and callable(getattr(agent, "enable")):
            agent.enable()
        self._logger.info("agent_enabled", agent_id=agent_id)

    def disable(self, agent_id: str) -> None:
        """Gap 8: call the agent's own disable() instead of mutating a private attr."""
        agent = self.require(agent_id)
        if hasattr(agent, "disable") and callable(getattr(agent, "disable")):
            agent.disable()
        self._logger.info("agent_disabled", agent_id=agent_id)

    # ── Health check loop ─────────────────────────────────────────────────────

    async def _check_one(self, agent: BaseAgent) -> str:
        """Call agent.health_check(); default to 'healthy' if not implemented."""
        try:
            if hasattr(agent, "health_check") and callable(getattr(agent, "health_check")):
                return await agent.health_check()
            return "healthy"
        except Exception as exc:
            self._logger.warning(
                "agent_health_check_error",
                agent_id=agent.agent_id,
                error=str(exc),
            )
            return "degraded"

    async def run_health_checks(self) -> None:
        """Gap 1: background loop — poll every agent's health_check() every 30 s."""
        while True:
            try:
                await asyncio.sleep(_HEALTH_CHECK_INTERVAL_SECONDS)
                for agent in list(self._agents.values()):
                    result = await self._check_one(agent)
                    prev = self._health.get(agent.agent_id)
                    if prev != result:
                        self._logger.info(
                            "agent_health_changed",
                            agent_id=agent.agent_id,
                            previous=prev,
                            current=result,
                        )
                    self._health[agent.agent_id] = result
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self._logger.warning("health_check_loop_error", error=str(exc))

    async def start(self) -> None:
        """Start the background health-check loop. Called from the app lifespan."""
        self._health_task = asyncio.create_task(self.run_health_checks(), name="agent-health-loop")
        self._logger.info("agent_registry_health_loop_started")

    async def stop(self) -> None:
        """Cancel the health-check loop. Called from the app lifespan on shutdown."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        self._logger.info("agent_registry_health_loop_stopped")

    # ── Backward compat ───────────────────────────────────────────────────────

    def list_agents(self) -> list[str]:
        """Return the registered agent ids (legacy helper)."""
        return list(self._agents.keys())
