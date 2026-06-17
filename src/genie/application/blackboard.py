"""Shared blackboard for the Executor.

In-memory map ``{task_id: {agent_id, text, view?} | {error: ...}}`` with an optional
hot mirror to Redis (wired in Phase 3 via ``redis_store``; no-ops when None). Ported
in spirit from BaseAgentFramework ``orchestrator/blackboard.py``.
"""

from __future__ import annotations

import contextlib
from typing import Any


class Blackboard:
    def __init__(
        self, thread_id: str = "", run_id: str = "", redis_store: Any | None = None
    ) -> None:
        self.thread_id = thread_id
        self.run_id = run_id
        self._mem: dict[str, dict] = {}
        self._redis = redis_store

    def get(self, task_id: str) -> dict | None:
        return self._mem.get(task_id)

    def snapshot(self) -> dict[str, dict]:
        return dict(self._mem)

    async def write(self, task_id: str, payload: dict) -> None:
        self._mem[task_id] = payload
        await self._mirror(task_id, payload)

    async def write_error(self, task_id: str, message: str) -> None:
        payload = {"error": message}
        self._mem[task_id] = payload
        await self._mirror(task_id, payload)

    async def _mirror(self, task_id: str, payload: dict) -> None:
        if self._redis is None:
            return
        key = f"bb:{self.thread_id}:{self.run_id}:{task_id}"
        with contextlib.suppress(Exception):
            await self._redis.set_json(key, payload, ttl_seconds=3600)
