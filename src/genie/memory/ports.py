"""Memory store protocols — session and long-term memory abstractions."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel


class SessionEntry(BaseModel):
    """One key/value slot scoped to a single conversation."""

    conversation_id: str
    key: str
    value: Any


@runtime_checkable
class SessionMemoryStore(Protocol):
    """Per-conversation scratch storage — ephemeral working memory for a turn."""

    async def get(self, conversation_id: str, key: str) -> Any | None: ...

    async def set(self, conversation_id: str, key: str, value: Any) -> None: ...

    async def get_all(self, conversation_id: str) -> dict[str, Any]: ...

    async def clear(self, conversation_id: str) -> None: ...


@runtime_checkable
class LongTermMemoryStore(Protocol):
    """Durable per-user memory with keyword/semantic ``search`` for recall."""

    async def save(
        self,
        user_id: str,
        key: str,
        value: Any,
        metadata: dict | None = None,
    ) -> None: ...

    async def get(self, user_id: str, key: str) -> Any | None: ...

    async def search(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]: ...

    async def delete(self, user_id: str, key: str) -> bool: ...
