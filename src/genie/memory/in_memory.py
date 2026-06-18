"""In-process implementations of SessionMemoryStore and LongTermMemoryStore."""

from __future__ import annotations

from typing import Any


class InMemorySessionStore:
    """Thread-safe in-memory session store backed by a nested dict."""

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    async def get(self, conversation_id: str, key: str) -> Any | None:
        """Value for ``key`` in this conversation, or None if absent."""
        return self._store.get(conversation_id, {}).get(key)

    async def set(self, conversation_id: str, key: str, value: Any) -> None:
        """Store ``value`` under ``key`` for this conversation."""
        if conversation_id not in self._store:
            self._store[conversation_id] = {}
        self._store[conversation_id][key] = value

    async def get_all(self, conversation_id: str) -> dict[str, Any]:
        """Snapshot copy of every key/value for this conversation."""
        return dict(self._store.get(conversation_id, {}))

    async def clear(self, conversation_id: str) -> None:
        """Drop all entries for this conversation."""
        self._store.pop(conversation_id, None)


class InMemoryLongTermStore:
    """In-memory long-term store with keyword search support."""

    def __init__(self) -> None:
        # { user_id: { key: {"value": ..., "metadata": {...}} } }
        self._store: dict[str, dict[str, dict[str, Any]]] = {}

    async def save(
        self,
        user_id: str,
        key: str,
        value: Any,
        metadata: dict | None = None,
    ) -> None:
        """Upsert ``value`` (with optional ``metadata``) under ``key`` for a user."""
        if user_id not in self._store:
            self._store[user_id] = {}
        self._store[user_id][key] = {"value": value, "metadata": metadata or {}}

    async def get(self, user_id: str, key: str) -> Any | None:
        """Stored value for ``key``, or None if the user has no such entry."""
        entry = self._store.get(user_id, {}).get(key)
        return entry["value"] if entry is not None else None

    async def search(self, user_id: str, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Up to ``limit`` entries whose key or value substring-matches ``query``."""
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        for key, entry in self._store.get(user_id, {}).items():
            if query_lower in key.lower() or query_lower in str(entry.get("value", "")).lower():
                results.append({"key": key, **entry})
        return results[:limit]

    async def delete(self, user_id: str, key: str) -> bool:
        """Remove ``key`` for the user; True if it existed, else False."""
        if user_id in self._store and key in self._store[user_id]:
            del self._store[user_id][key]
            return True
        return False
