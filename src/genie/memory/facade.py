"""MemoryFacade — bundles the multi-store backends behind the small async API
the planner / synthesizer nodes call.

Every method is best-effort: any backend failure is logged and the method
returns a benign value so a turn never crashes on a memory error. The facade
holds already-built stores (none of which it constructs) so the factory owns all
driver/connection concerns and degradation.
"""

from __future__ import annotations

import json
from typing import Any

from genie.observability.logging import get_logger

logger = get_logger(__name__)

# Mirrors BaseAgentFramework synthesizer._FACTS_PROMPT.
_FACTS_PROMPT = (
    "You extract durable facts from an assistant's answer. Return ONLY JSON of the form:\n"
    '{"facts":[{"key":"<short_snake_case>","value":"<concise string>","scope":"global|session"}]}\n'
    "- GLOBAL: stable facts about the user or the world that stay true across ALL future "
    "conversations (the user's name, their home city, a place's coordinates, a standing "
    "preference).\n"
    "- SESSION: facts that are only meaningful inside THIS conversation (a specific outage "
    "id the user is looking at, a one-off selection).\n"
    '- When unsure, choose "session".\n'
    "- Keys are lowercase snake_case, short and reusable. Omit anything that is not a "
    "confident, reusable fact. An empty list is fine.\n"
    "- Output JSON only — no markdown, no prose."
)


class MemoryFacade:
    """Best-effort async memory API over already-built, optional backends.

    Each backend may be None or disabled; methods degrade to benign values so a
    turn never crashes on a memory error.
    """

    def __init__(
        self,
        mongo: Any = None,
        vector: Any = None,
        redis: Any = None,
        llm: Any = None,
        embed: Any = None,
    ) -> None:
        # Public: bootstrap calls mongo.ensure_indexes() and the UI router uses it
        # directly for save_turn / conversation list / get / delete. The planner
        # reads mongo/vector .enabled to report its recall db_ops in the trace.
        self.mongo = mongo
        self.vector = vector
        # Exposed so bootstrap can pass it to the executor as the blackboard mirror.
        self.redis = redis
        self._llm = llm
        self._embed = embed

    # ── reads ──────────────────────────────────────────────────────────────
    async def recall(self, conversation_id: str, query: str) -> list[dict]:
        """Semantically-similar past memories for this conversation.

        Returns a list of {"content": ...}. Empty when vector memory is absent
        or disabled.
        """
        if self.vector is None or not getattr(self.vector, "enabled", False):
            return []
        try:
            return await self.vector.search(conversation_id, query)
        except Exception as exc:
            logger.warning("memory_recall_failed", error=str(exc))
            return []

    async def query_facts(self, conversation_id: str) -> dict[str, str]:
        """Merged global + session facts visible to this conversation."""
        if self.mongo is None or not getattr(self.mongo, "enabled", False):
            return {}
        try:
            return await self.mongo.query_facts(conversation_id)
        except Exception as exc:
            logger.warning("memory_query_facts_failed", error=str(exc))
            return {}

    # ── write-back ───────────────────────────────────────────────────────────
    async def writeback(self, state: Any, blackboard: dict, text: str) -> list[dict[str, Any]]:
        """Persist this turn: durable commits, semantic embedding, durable facts.

        Best-effort throughout — each step catches its own errors so a failure
        just logs and writeback returns normally. Returns one trace ``db_op`` per
        real datastore write so the Synthesizer step can surface them in the UI.
        """
        conversation_id = getattr(state, "conversation_id", "") or ""
        run_id = getattr(state, "run_id", "") or ""
        ops: list[dict[str, Any]] = []

        # (a) Commit persistable fields to Mongo. We don't have the agent registry
        # output_schema persist flags wired here, so commit the whole turn text
        # per task (spec's fallback).
        for task_id in await self._commit(state, blackboard, text, conversation_id, run_id):
            ops.append(
                {
                    "store": "mongodb",
                    "op": "write",
                    "node": "synthesizer",
                    "detail": f"commit {task_id}",
                    "code": (
                        f"db.agent_commits.insertOne({{run_id:'{run_id}', "
                        f"task_id:'{task_id}'}})"
                    ),
                    "enabled": True,
                }
            )

        # (b) Embed the final answer for future semantic recall.
        summary = (text or "").strip()
        if summary and self.vector is not None and getattr(self.vector, "enabled", False):
            try:
                await self.vector.add(conversation_id, summary[:1000])
                ops.append(
                    {
                        "store": "milvus",
                        "op": "write",
                        "node": "synthesizer",
                        "detail": "answer embedded for semantic recall",
                        "code": f"milvus.insert(long_term_memory, thread='{conversation_id}')",
                        "enabled": True,
                    }
                )
            except Exception as exc:
                logger.warning("memory_vector_add_failed", error=str(exc))

        # (c) LLM-extract durable facts and upsert each into Mongo.
        fact_keys = await self._extract_and_store_facts(
            state, blackboard, text, conversation_id, run_id
        )
        if fact_keys:
            n = len(fact_keys)
            ops.append(
                {
                    "store": "mongodb",
                    "op": "write",
                    "node": "synthesizer",
                    "detail": f"{n} durable fact{'s' if n != 1 else ''} upserted",
                    "code": "db.agent_facts.updateOne({scope, key}, ..., {upsert:true})",
                    "enabled": True,
                    "hits": fact_keys,
                }
            )
        return ops

    async def _commit(
        self, state: Any, blackboard: dict, text: str, conversation_id: str, run_id: str
    ) -> list[str]:
        """Persist each non-error blackboard entry as a durable per-task commit.

        Returns the task ids actually committed (empty when Mongo is absent).
        """
        if self.mongo is None or not getattr(self.mongo, "enabled", False):
            return []
        plan = getattr(state, "plan", None) or {}
        subtasks = plan.get("subtasks") if isinstance(plan, dict) else None
        by_id: dict[str, dict] = {}
        if isinstance(subtasks, list):
            for st in subtasks:
                if isinstance(st, dict) and st.get("id"):
                    by_id[st["id"]] = st
        committed: list[str] = []
        try:
            for task_id, entry in (blackboard or {}).items():
                if not isinstance(entry, dict) or "error" in entry:
                    continue
                subtask = by_id.get(task_id, {})
                await self.mongo.commit(
                    run_id=run_id,
                    thread_id=conversation_id,
                    agent_id=subtask.get("agent_id", "") if isinstance(subtask, dict) else "",
                    agent_version=subtask.get("agent_version", "")
                    if isinstance(subtask, dict)
                    else "",
                    task_id=task_id,
                    payload={"text": text},
                )
                committed.append(task_id)
        except Exception as exc:
            logger.warning("memory_commit_failed", error=str(exc))
        return committed

    async def _extract_and_store_facts(
        self, state: Any, blackboard: dict, text: str, conversation_id: str, run_id: str
    ) -> list[str]:
        """LLM-extract durable facts from a complete answer and upsert each.

        Returns the keys of the facts actually upserted (empty when gated off,
        the LLM/Mongo are absent, or nothing extractable was found).
        """
        stored: list[str] = []
        # Gate: only extract from a real, complete answer.
        if getattr(state, "partial", False) or not (text or "").strip():
            return stored
        if self._llm is None or self.mongo is None or not getattr(self.mongo, "enabled", False):
            return stored
        try:
            from genie.application.state import Message

            user_input = ""
            messages = getattr(state, "messages", None) or []
            for m in reversed(messages):
                if getattr(m, "role", None) == "user":
                    user_input = getattr(m, "content", "") or ""
                    break
            prompt = (
                f"USER REQUEST:\n{user_input}\n\n"
                f"FINAL ANSWER:\n{text}\n\n"
                f"BLACKBOARD (JSON):\n{self._render_blackboard(blackboard)}\n\n"
                "Extract the facts now."
            )
            resp = await self._llm.complete(
                [
                    Message(role="system", content=_FACTS_PROMPT),
                    Message(role="user", content=prompt),
                ],
                max_tokens=512,
                temperature=0.0,
            )
            parsed = _extract_json(resp.content) or {}
            facts = self._validate_facts(parsed.get("facts"))
            for f in facts:
                await self.mongo.upsert_fact(
                    scope=f["scope"],
                    key=f["key"],
                    value=f["value"],
                    thread_id=conversation_id,
                    run_id=run_id,
                )
                stored.append(f["key"])
            logger.info("memory_facts_extracted", count=len(stored))
        except Exception as exc:
            logger.warning("memory_facts_extract_failed", error=str(exc))
        return stored

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _render_blackboard(
        blackboard: dict, per_entry_cap: int = 2500, total_cap: int = 8000
    ) -> str:
        """Render the blackboard as compact JSON for the facts prompt, capping
        each entry and the total length so the prompt stays bounded."""
        parts: list[str] = []
        for tid, entry in (blackboard or {}).items():
            if not isinstance(entry, dict):
                continue
            try:
                s = json.dumps(entry, default=str)
            except Exception:
                s = str(entry)
            if len(s) > per_entry_cap:
                s = s[:per_entry_cap] + "...(truncated)"
            parts.append(f'"{tid}": {s}')
        return ("{" + ", ".join(parts) + "}")[:total_cap]

    @staticmethod
    def _validate_facts(raw: Any) -> list[dict]:
        """Coerce the LLM's facts array into clean {key, value, scope} dicts.
        Drops malformed entries; defaults unknown scope to 'session'; caps to 10
        facts and 500-char values."""
        if not isinstance(raw, list):
            return []
        out: list[dict] = []
        for item in raw[:10]:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            scope = item.get("scope")
            scope = scope if scope in ("global", "session") else "session"
            out.append({"key": key.strip(), "value": value.strip()[:500], "scope": scope})
        return out


def _extract_json(raw: str) -> dict | None:
    """Find the first balanced JSON object in ``raw`` and parse it (tolerant of
    LLM tics like code fences / trailing junk). Mirrors planner.parsing.extract_json.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    return None
    try:
        return json.loads(raw[start:])
    except json.JSONDecodeError:
        return None
