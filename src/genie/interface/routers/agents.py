"""Agent management endpoints — list, inspect, enable/disable, unregister, and register."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from genie.agents.base import AgentInfo, AgentResult, AgentTask
from genie.platform.errors import ErrorCode, GenieError

router = APIRouter(prefix="/api/v1/agents", tags=["agents"])


def _get_registry(request: Request) -> Any:
    registry = getattr(request.app.state, "agent_registry", None)
    if registry is None:
        raise GenieError(ErrorCode.INTERNAL_ERROR, "Agent registry not initialised")
    return registry


# ── Read endpoints ────────────────────────────────────────────────────────────

@router.get("", response_model=list[AgentInfo], summary="List all registered agents")
async def list_agents(request: Request) -> list[AgentInfo]:
    return _get_registry(request).list_all()


@router.get("/{agent_id}", response_model=AgentInfo, summary="Get agent details and health")
async def get_agent(agent_id: str, request: Request) -> AgentInfo:
    registry = _get_registry(request)
    agent = registry.get(agent_id)
    if agent is None:
        raise GenieError(ErrorCode.NOT_FOUND, f"Agent '{agent_id}' not found")
    # Return info with live health from registry cache
    info = agent.get_info()
    live_health = registry._health.get(agent_id, "healthy")
    return info.model_copy(update={"health": live_health})


# ── Lifecycle management ──────────────────────────────────────────────────────

@router.post("/{agent_id}/enable", summary="Enable a disabled agent")
async def enable_agent(agent_id: str, request: Request) -> JSONResponse:
    """Gap 8: calls the agent's own enable() via the registry."""
    _get_registry(request).enable(agent_id)
    return JSONResponse({"agent_id": agent_id, "enabled": True})


@router.post("/{agent_id}/disable", summary="Disable an agent without removing it")
async def disable_agent(agent_id: str, request: Request) -> JSONResponse:
    """Gap 8: calls the agent's own disable() via the registry."""
    _get_registry(request).disable(agent_id)
    return JSONResponse({"agent_id": agent_id, "enabled": False})


@router.delete("/{agent_id}", summary="Unregister an agent at runtime")
async def unregister_agent(agent_id: str, request: Request) -> JSONResponse:
    """Gap 6: remove an agent without a server restart.

    Use this when an agent's backend is permanently decommissioned.
    A server restart re-registers all providers from ``app.py``.
    """
    _get_registry(request).unregister(agent_id)
    return JSONResponse({"agent_id": agent_id, "status": "unregistered"})


# ── Dynamic registration ──────────────────────────────────────────────────────

class _ExternalAgentProxy:
    """Lightweight BaseAgent implementation backed by an AgentInfo card.

    Gap 9: allows external or test agents to be registered at runtime by
    posting their card to ``POST /api/v1/agents/register``.  The ``execute``
    method raises NotImplementedError — wiring a remote A2A execute() is
    Phase 2.  The agent shows up in discovery immediately after registration.
    """

    def __init__(self, info: AgentInfo) -> None:
        self._info = info
        self._enabled = info.enabled

    @property
    def agent_id(self) -> str:
        return self._info.agent_id

    @property
    def name(self) -> str:
        return self._info.name

    @property
    def description(self) -> str:
        return self._info.description

    @property
    def capabilities(self) -> list[str]:
        return self._info.capabilities

    @property
    def version(self) -> str:
        return self._info.version

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    async def health_check(self) -> str:
        return "healthy"

    def get_info(self) -> AgentInfo:
        return self._info.model_copy(update={"enabled": self._enabled})

    async def execute(self, task: AgentTask, context: dict) -> AgentResult:
        raise NotImplementedError(
            f"Agent '{self.agent_id}' was dynamically registered from an AgentInfo card "
            "and has no local execute() implementation. "
            "Wire a RemoteAgent (A2A) executor to enable execution."
        )


@router.post("/register", response_model=AgentInfo, summary="Dynamically register an agent card")
async def register_agent(body: AgentInfo, request: Request) -> AgentInfo:
    """Gap 9: register an agent at runtime from its AgentInfo card.

    The registered proxy appears in ``GET /api/v1/agents`` and will be selected
    by the Planner for matching capability strings.  Calling ``execute()`` on it
    raises until a real execution backend is connected (Phase 2 — A2A).

    Useful for: integration tests, staging agent previews, and future A2A agents.
    """
    registry = _get_registry(request)
    proxy = _ExternalAgentProxy(body)
    registry.register(proxy)
    return proxy.get_info()
