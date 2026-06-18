"""MongoStore — async multi-collection durable memory (motor).

Ports BaseAgentFramework's mongo_store / facts_store / commit_store into one
async adapter. The driver (``motor``) is an optional extra and is imported
lazily inside ``__init__``; when it is missing the store degrades to
``enabled = False`` and every method no-ops, mirroring the source stores that
no-op when their env is unset.

Collection names + semantics mirror the source:
  - ``short_term_memory`` — hot recent turns, 24h TTL on ``updated_at``.
  - ``conversations``     — durable conversation history (no TTL).
  - ``agent_facts``       — structured facts, global + session scope.
  - ``agent_commits``     — durable per-task agent outputs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from genie.observability.logging import get_logger

logger = get_logger(__name__)

# Hot recent-context cache TTL (24h). The durable ``conversations`` collection
# (no TTL) is the source of truth for listing/resuming.
_SHORT_TERM_TTL_SECONDS = 86400  # 24 hours

# Session facts get a sliding 30-day TTL: every read/write pushes expiry out so
# an actively-resumed conversation never loses its facts while abandoned threads
# self-clean. Globals never set ``expireAt`` and so never expire.
_SESSION_TTL = timedelta(days=30)


class MongoStore:
    """Async MongoDB-backed store for facts, commits, and conversations.

    Lazily imports ``motor``; ``enabled`` is False when the driver is missing.
    All methods are async and best-effort — a driver failure is logged and
    swallowed so a turn never crashes on a persistence error.
    """

    def __init__(self, uri: str, db: str) -> None:
        """Connect to Mongo and bind the collections; degrade to disabled if the
        ``motor`` driver is missing or the connection cannot be set up."""
        self._enabled = False
        self._client = None
        try:
            import motor.motor_asyncio  # lazy: optional extra
        except ImportError:
            logger.warning("mongo_disabled", reason="motor not installed")
            return
        try:
            self._client = motor.motor_asyncio.AsyncIOMotorClient(uri)
            self._db = self._client[db]
            self._short_term = self._db["short_term_memory"]
            self._conversations = self._db["conversations"]
            self._facts = self._db["agent_facts"]
            self._commits = self._db["agent_commits"]
            self._enabled = True
        except Exception as exc:  # connection/config error → degrade
            logger.warning("mongo_connect_failed", uri=uri, error=str(exc))
            self._client = None

    @property
    def enabled(self) -> bool:
        """True when the driver loaded and the client connected."""
        return self._enabled

    # ── indexes ────────────────────────────────────────────────────────────
    async def ensure_indexes(self) -> None:
        """Create the TTL and lookup indexes (idempotent; reconciles TTL drift)."""
        if not self._enabled:
            return
        try:
            from pymongo import ASCENDING, DESCENDING
            from pymongo.errors import OperationFailure
        except ImportError:  # motor ships pymongo, but be defensive
            return
        try:
            try:
                await self._short_term.create_index(
                    "updated_at", expireAfterSeconds=_SHORT_TERM_TTL_SECONDS
                )
            except OperationFailure as exc:
                # Index already exists with a different TTL — update via collMod.
                if getattr(exc, "code", None) == 85:  # IndexOptionsConflict
                    await self._db.command(
                        {
                            "collMod": "short_term_memory",
                            "index": {
                                "keyPattern": {"updated_at": 1},
                                "expireAfterSeconds": _SHORT_TERM_TTL_SECONDS,
                            },
                        }
                    )
                else:
                    raise
            # Durable conversations — no TTL, sorted by recency for the list.
            await self._conversations.create_index([("updated_at", DESCENDING)])
            # Facts: lookup + session TTL (scoped to session docs).
            await self._facts.create_index([("thread_id", ASCENDING)])
            await self._facts.create_index([("scope", ASCENDING)])
            await self._facts.create_index(
                [("expireAt", ASCENDING)],
                expireAfterSeconds=0,
                partialFilterExpression={"scope": "session"},
            )
            # Commits: lookup by run / thread.
            await self._commits.create_index([("run_id", ASCENDING)])
            await self._commits.create_index([("thread_id", ASCENDING)])
        except Exception as exc:
            logger.warning("mongo_ensure_indexes_failed", error=str(exc))

    # ── facts ──────────────────────────────────────────────────────────────
    async def upsert_fact(
        self,
        scope: str,
        key: str,
        value: str,
        thread_id: str | None = None,
        run_id: str = "",
    ) -> None:
        """Upsert one structured fact (global = stable; session = this thread)."""
        if not self._enabled:
            return
        now = datetime.now(UTC)
        try:
            if scope == "global":
                await self._facts.update_one(
                    {"_id": f"g::{key}"},
                    {
                        "$set": {
                            "scope": "global",
                            "key": key,
                            "value": value,
                            "entity": key,
                            "thread_id": None,
                            "run_id": run_id,
                            "updated_at": now,
                        },
                        # Clear any stale TTL if a key was previously session-scoped.
                        "$unset": {"expireAt": ""},
                    },
                    upsert=True,
                )
            else:
                await self._facts.update_one(
                    {"_id": f"s::{thread_id}::{key}"},
                    {
                        "$set": {
                            "scope": "session",
                            "key": key,
                            "value": value,
                            "entity": None,
                            "thread_id": thread_id,
                            "run_id": run_id,
                            "updated_at": now,
                            "expireAt": now + _SESSION_TTL,
                        }
                    },
                    upsert=True,
                )
        except Exception as exc:
            logger.warning("mongo_upsert_fact_failed", scope=scope, key=key, error=str(exc))

    async def query_facts(self, thread_id: str) -> dict[str, str]:
        """Merged facts visible to this thread: all globals plus this thread's
        session facts (session overrides global on a key collision). Slides the
        session TTL so an actively-resumed conversation keeps its facts.
        """
        out: dict[str, str] = {}
        if not self._enabled:
            return out
        try:
            async for doc in self._facts.find({"scope": "global"}):
                out[doc["key"]] = doc["value"]
            async for doc in self._facts.find({"scope": "session", "thread_id": thread_id}):
                out[doc["key"]] = doc["value"]
            if thread_id:
                await self._facts.update_many(
                    {"scope": "session", "thread_id": thread_id},
                    {"$set": {"expireAt": datetime.now(UTC) + _SESSION_TTL}},
                )
        except Exception as exc:
            logger.warning("mongo_query_facts_failed", error=str(exc))
        return out

    # ── commits ──────────────────────────────────────────────────────────────
    async def commit(
        self,
        run_id: str,
        thread_id: str,
        agent_id: str,
        agent_version: str,
        task_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Persist one agent's output as a durable ``agent_commits`` document."""
        if not self._enabled:
            return
        try:
            await self._commits.insert_one(
                {
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "agent_id": agent_id,
                    "agent_version": agent_version,
                    "task_id": task_id,
                    "payload": payload,
                    "committed_at": datetime.now(UTC),
                }
            )
        except Exception as exc:
            logger.warning("mongo_commit_failed", run_id=run_id, agent_id=agent_id, error=str(exc))

    # ── conversations ──────────────────────────────────────────────────────
    async def save_turn(self, thread_id: str, role: str, content: str) -> None:
        """Append one {role, content} turn to the durable conversation.

        Sets the title from the first user turn on insert; bumps ``updated_at``
        and mirrors to the hot short-term cache.
        """
        if not self._enabled:
            return
        now = datetime.now(UTC)
        turn = {"role": role, "content": content}
        try:
            title = (
                content.strip().replace("\n", " ")[:60] if role == "user" else "New conversation"
            )
            await self._conversations.update_one(
                {"_id": thread_id},
                {
                    "$push": {"messages": turn},
                    "$inc": {"message_count": 1},
                    "$set": {"updated_at": now},
                    "$setOnInsert": {"created_at": now, "title": title or "New conversation"},
                },
                upsert=True,
            )
            await self._short_term.update_one(
                {"_id": thread_id},
                {"$push": {"messages": turn}, "$set": {"updated_at": now}},
                upsert=True,
            )
        except Exception as exc:
            logger.warning("mongo_save_turn_failed", thread_id=thread_id, error=str(exc))

    async def list_conversations(self, limit: int = 50) -> list[dict]:
        """Recent conversations for a sidebar: id, title, recency, size."""
        out: list[dict] = []
        if not self._enabled:
            return out
        try:
            from pymongo import DESCENDING

            cursor = (
                self._conversations.find({}, {"title": 1, "updated_at": 1, "message_count": 1})
                .sort("updated_at", DESCENDING)
                .limit(limit)
            )
            async for d in cursor:
                updated = d.get("updated_at")
                out.append(
                    {
                        "thread_id": d["_id"],
                        "title": d.get("title") or "Conversation",
                        "updated_at": updated.isoformat() if updated else None,
                        "message_count": d.get("message_count", 0),
                    }
                )
        except Exception as exc:
            logger.warning("mongo_list_conversations_failed", error=str(exc))
        return out

    async def get_conversation(self, thread_id: str) -> list[dict]:
        """Full conversation as simple {role, content} turns for the UI."""
        if not self._enabled:
            return []
        try:
            conv = await self._conversations.find_one({"_id": thread_id})
        except Exception as exc:
            logger.warning("mongo_get_conversation_failed", thread_id=thread_id, error=str(exc))
            return []
        if not conv or not conv.get("messages"):
            return []
        turns: list[dict] = []
        for m in conv["messages"]:
            content = str(m.get("content", "") or "").strip()
            role = m.get("role")
            if role == "user":
                turns.append({"role": "user", "content": content})
            elif role == "assistant" and content:
                turns.append({"role": "assistant", "content": content})
        return turns

    async def delete_conversation(self, thread_id: str) -> None:
        """Delete a conversation from both durable and short-term collections."""
        if not self._enabled:
            return
        try:
            await self._conversations.delete_one({"_id": thread_id})
            await self._short_term.delete_one({"_id": thread_id})
        except Exception as exc:
            logger.warning("mongo_delete_conversation_failed", thread_id=thread_id, error=str(exc))

    async def aclose(self) -> None:
        """Close the Mongo client and disable the store (shutdown hook)."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: S110 - best-effort close on shutdown
                pass
            self._client = None
            self._enabled = False
