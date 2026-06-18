"""Request/response models for the Registry/Discovery Service.

These define the wire contract between agent processes (which register and
heartbeat) and consumers (which discover agents). The agent payload itself is
the shared :class:`AgentMeta` pydantic model so there is no schema drift between
what an agent advertises and what consumers deserialize.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from registry_service.agent_meta import AgentMeta


class RegisterRequest(BaseModel):
    """POST /register body: the full :class:`AgentMeta` an agent advertises."""

    meta: AgentMeta


class RegisterResponse(BaseModel):
    """Registration reply: assigned instance id plus liveness/heartbeat timing."""

    instance_id: str
    ttl_seconds: int
    heartbeat_interval_seconds: int


class HeartbeatRequest(BaseModel):
    """Heartbeat body: the instance to refresh, with an optional status update."""

    instance_id: str
    status: Literal["active", "deprecated"] | None = None


class HeartbeatResponse(BaseModel):
    """Heartbeat reply. ``known`` is the actionable flag for the agent harness."""

    ok: bool
    # False when the registry has no record for this instance_id — the agent
    # harness treats this as a signal to re-register (e.g. after a TTL sweep).
    known: bool


class DeregisterRequest(BaseModel):
    """Deregister body: the instance id to remove from the registry."""

    instance_id: str


class ListResponse(BaseModel):
    """Discovery reply: the live agent records matching the query."""

    agents: list[AgentMeta] = Field(default_factory=list)
