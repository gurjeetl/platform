"""RedisStore — async hot blackboard mirror (redis.asyncio).

Ports BaseAgentFramework's redis_store. ``redis`` is an optional extra and is
imported lazily; ``enabled`` is False when ``redis_url`` is None or the package
is missing, in which case every method no-ops — keeping the platform usable for
dev without standing up Redis.

Loop-aware: redis.asyncio connections are bound to the event loop that created
them. We keep one client per running event loop so a connection created on a
transient executor loop is never shared with the main loop.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from genie.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_TTL_SECONDS = 3600  # 1h blackboard TTL (working memory for a run)


class RedisStore:
    """Thin async wrapper for blackboard hot-storage. No-ops when disabled."""

    def __init__(self, redis_url: str | None) -> None:
        """Probe for the ``redis`` driver and enable the store; degrade to disabled
        when ``redis_url`` is unset or the package is missing."""
        self._url = redis_url
        self._enabled = False
        self._clients: dict = {}  # event loop -> redis client
        if not self._url:
            logger.warning("redis_disabled", reason="redis_url unset")
            return
        try:
            from redis import asyncio as redis_asyncio  # noqa: F401  (lazy probe)
        except ImportError:
            logger.warning("redis_disabled", reason="redis package not installed")
            return
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """True when the ``redis`` driver is available and a URL is configured."""
        return self._enabled

    def _client(self):
        """A redis client bound to the CURRENT running loop (created on demand)."""
        if not self._enabled:
            return None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        # Drop clients whose loop has closed so the dict doesn't grow unbounded.
        for dead in [lp for lp in self._clients if lp.is_closed()]:
            self._clients.pop(dead, None)
        client = self._clients.get(loop)
        if client is None:
            from redis import asyncio as redis_asyncio

            # protocol=2 (RESP2): redis-py 8 defaults to RESP3 and sends HELLO 3
            # on connect, which Redis < 6 rejects. RESP2 works on all versions.
            client = redis_asyncio.from_url(
                self._url, encoding="utf-8", decode_responses=True, protocol=2
            )
            self._clients[loop] = client
        return client

    async def set_json(self, key: str, value: Any, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        """JSON-encode and store ``value`` under ``key`` with a TTL. No-op when
        disabled; logs and swallows any error."""
        client = self._client()
        if not client:
            return
        try:
            await client.set(key, json.dumps(value, default=str), ex=ttl_seconds)
        except Exception as exc:
            logger.warning("redis_set_failed", key=key, error=str(exc))

    async def get_json(self, key: str) -> Any | None:
        """Fetch and JSON-decode ``key``; None when missing, disabled, or on error."""
        client = self._client()
        if not client:
            return None
        try:
            raw = await client.get(key)
            return json.loads(raw) if raw else None
        except Exception as exc:
            logger.warning("redis_get_failed", key=key, error=str(exc))
            return None

    async def aclose(self) -> None:
        """Close every per-loop client and clear the registry (shutdown hook)."""
        for client in list(self._clients.values()):
            try:
                await client.aclose()
            except Exception:
                try:
                    await client.close()
                except Exception:  # noqa: S110 - best-effort close on shutdown
                    pass
        self._clients.clear()
