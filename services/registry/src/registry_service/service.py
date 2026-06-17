"""Standalone Registry/Discovery Service.

Independent FastAPI app that agents self-register with (and heartbeat to), and
that consumers query for agent discovery.

Run: python -m registry_service.service   (binds 0.0.0.0:2005 by default)

Endpoints:
  POST /register                  — register/refresh an agent instance
  POST /heartbeat/{instance_id}   — refresh liveness for an instance
  POST /deregister                — remove an instance
  GET  /agents                    — list live agents (optional ?agent_id= / ?tag=)
  GET  /agents/{agent_id}         — list live instances of one agent
  GET  /health                    — liveness probe (unauthenticated)

All routes except /health are guarded by an optional bearer token
(REGISTRY_AUTH_TOKEN); when unset, the guard is a no-op (local dev).
"""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException

from registry_service.contracts import (
    DeregisterRequest,
    HeartbeatRequest,
    HeartbeatResponse,
    ListResponse,
    RegisterRequest,
    RegisterResponse,
)
from registry_service.store import get_registry_store

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
_log = logging.getLogger("registry_service")


def require_auth(authorization: str | None = Header(None)) -> None:
    """Bearer-token gate. No-op when REGISTRY_AUTH_TOKEN is unset (local dev)."""
    token = os.getenv("REGISTRY_AUTH_TOKEN")
    if not token:
        return
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="invalid registry token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = get_registry_store()
    await store.ensure_indexes()
    _log.info("registry.indexes_ensured ttl_seconds=%s", store.ttl_seconds)
    yield
    store.close()


app = FastAPI(title="Agent Registry Service", lifespan=lifespan)


def _heartbeat_interval() -> int:
    return int(
        os.getenv(
            "REGISTRY_HEARTBEAT_SECONDS",
            str(max(1, get_registry_store().ttl_seconds // 3)),
        )
    )


@app.post("/register", response_model=RegisterResponse, dependencies=[Depends(require_auth)])
async def register(req: RegisterRequest) -> RegisterResponse:
    store = get_registry_store()
    meta = req.meta
    if not meta.endpoint:
        raise HTTPException(status_code=422, detail="meta.endpoint is required for remote agents")
    if not meta.instance_id:
        meta.instance_id = uuid.uuid4().hex
    await store.register(meta)
    _log.info(
        "registry.register agent_id=%s instance_id=%s endpoint=%s",
        meta.agent_id,
        meta.instance_id,
        meta.endpoint,
    )
    return RegisterResponse(
        instance_id=meta.instance_id,
        ttl_seconds=store.ttl_seconds,
        heartbeat_interval_seconds=_heartbeat_interval(),
    )


@app.post(
    "/heartbeat/{instance_id}",
    response_model=HeartbeatResponse,
    dependencies=[Depends(require_auth)],
)
async def heartbeat(instance_id: str, req: HeartbeatRequest | None = None) -> HeartbeatResponse:
    """Refresh liveness for an instance. The instance id travels in the path;
    an optional body may carry a status update."""
    store = get_registry_store()
    status = req.status if req else None
    known = await store.heartbeat(instance_id, status)
    if not known:
        _log.warning("registry.heartbeat_unknown instance_id=%s", instance_id)
    return HeartbeatResponse(ok=known, known=known)


@app.post("/deregister", dependencies=[Depends(require_auth)])
async def deregister(req: DeregisterRequest) -> dict:
    store = get_registry_store()
    removed = await store.deregister(req.instance_id)
    _log.info("registry.deregister instance_id=%s removed=%s", req.instance_id, removed)
    return {"ok": True, "removed": removed}


@app.get("/agents", response_model=ListResponse, dependencies=[Depends(require_auth)])
async def list_agents(agent_id: str | None = None, tag: str | None = None) -> ListResponse:
    store = get_registry_store()
    agents = await (store.get_agent(agent_id) if agent_id else store.list_active())
    if tag:
        agents = [m for m in agents if tag in (m.capability_tags or [])]
    return ListResponse(agents=agents)


@app.get("/agents/{agent_id}", response_model=ListResponse, dependencies=[Depends(require_auth)])
async def get_agent(agent_id: str) -> ListResponse:
    store = get_registry_store()
    return ListResponse(agents=await store.get_agent(agent_id))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    port = int(os.getenv("REGISTRY_PORT", "2005"))
    uvicorn.run("registry_service.service:app", host="0.0.0.0", port=port)
