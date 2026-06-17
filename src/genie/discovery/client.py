"""Async HTTP client the platform uses to discover live agents.

Ported from BaseAgentFramework ``registry/registry_client.py`` and made async
(the bootstrap discovery bridge + refresh loop run in the event loop). A short TTL
cache absorbs repeated lookups; on a refresh failure the last good list is served
when ``serve_stale`` is set.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from genie.discovery.agent_meta import AgentMeta
from genie.observability.logging import get_logger

logger = get_logger(__name__)


class RegistryUnavailable(RuntimeError):
    """Registry service unreachable or returned a malformed response."""


class DiscoveryClient:
    def __init__(self, settings: Any) -> None:
        self._base_url = str(getattr(settings, "registry_url", "http://127.0.0.1:2005")).rstrip("/")
        self._cache_ttl = float(getattr(settings, "registry_cache_ttl_seconds", 5.0))
        self._timeout = float(getattr(settings, "registry_timeout_seconds", 3.0))
        self._serve_stale = bool(getattr(settings, "registry_serve_stale", True))
        token = getattr(settings, "registry_auth_token", None)
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._cache: list[AgentMeta] | None = None
        self._cache_at = 0.0

    async def list_active(self, *, force_refresh: bool = False) -> list[AgentMeta]:
        fresh = self._cache is not None and (time.monotonic() - self._cache_at) < self._cache_ttl
        if fresh and not force_refresh:
            return self._cache
        try:
            agents = await self._fetch_agents()
        except RegistryUnavailable:
            if self._serve_stale and self._cache is not None:
                logger.warning("discovery_serving_stale")
                return self._cache
            raise
        self._cache, self._cache_at = agents, time.monotonic()
        return agents

    async def get(self, agent_id: str) -> AgentMeta | None:
        return next((m for m in await self.list_active() if m.agent_id == agent_id), None)

    async def _fetch_agents(self) -> list[AgentMeta]:
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(f"{self._base_url}/agents", headers=self._headers)
                resp.raise_for_status()
                raw = resp.json().get("agents", [])
        except (httpx.HTTPError, ValueError) as e:
            logger.warning("discovery_fetch_failed", error=str(e))
            raise RegistryUnavailable(str(e)) from e
        out: list[AgentMeta] = []
        for rec in raw:
            try:
                out.append(AgentMeta.model_validate(rec))
            except Exception as e:  # noqa: BLE001 — tolerate one bad record
                logger.warning("discovery_bad_record", error=str(e))
        return [m for m in out if m.status == "active"]
