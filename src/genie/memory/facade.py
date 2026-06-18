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
        self._mongo = mongo
        self._vector = vector
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
        if self._vector is None or not getattr(self._vector, "enabled", False):
            return []
        try:
            return await self._vector.search(conversation_id, query)
        except Exception as exc:
            logger.warning("memory_recall_failed", error=str(exc))
            return []

    async def query_facts(self, conversation_id: str) -> dict[str, str]:
        """Merged global + session facts visible to this conversation."""
        if self._mongo is None or not getattr(self._mongo, "enabled", False):
            return {}
        try:
            return await self._mongo.query_facts(conversation_id)
        except Exception as exc:
            logger.warning("memory_query_facts_failed", error=str(exc))
            return {}

    # ── write-back ───────────────────────────────────────────────────────────
    async def writeback(self, state: Any, blackboard: dict, text: str) -> None:
        """Persist this turn: durable commits, semantic embedding, durable facts.

        Best-effort throughout — each step catches its own errors so a failure
        just logs and writeback returns normally.
        """
        conversation_id = getattr(state, "conversation_id", "") or ""
        run_id = getattr(state, "run_id", "") or ""

        # (a) Commit persistable fields to Mongo. We don't have the agent registry
        # output_schema persist flags wired here, so commit the whole turn text
        # per task (spec's fallback).
        await self._commit(state, blackboard, text, conversation_id, run_id)

        # (b) Embed the final answer for future semantic recall.
        summary = (text or "").strip()
        if summary and self._vector is not None and getattr(self._vector, "enabled", False):
            try:
                await self._vector.add(conversation_id, summary[:1000])
            except Exception as exc:
                logger.warning("memory_vector_add_failed", error=str(exc))

        # (c) LLM-extract durable facts and upsert each into Mongo.
        await self._extract_and_store_facts(state, blackboard, text, conversation_id, run_id)

    async def _commit(
        self, state: Any, blackboard: dict, text: str, conversation_id: str, run_id: str
    ) -> None:
        """Persist each non-error blackboard entry as a durable per-task commit."""
        if self._mongo is None or not getattr(self._mongo, "enabled", False):
            return
        plan = getattr(state, "plan", None) or {}
        subtasks = plan.get("subtasks") if isinstance(plan, dict) else None
        by_id: dict[str, dict] = {}
        if isinstance(subtasks, list):
            for st in subtasks:
                if isinstance(st, dict) and st.get("id"):
                    by_id[st["id"]] = st
        try:
            for task_id, entry in (blackboard or {}).items():
                if not isinstance(entry, dict) or "error" in entry:
                    continue
                subtask = by_id.get(task_id, {})
                await self._mongo.commit(
                    run_id=run_id,
                    thread_id=conversation_id,
                    agent_id=subtask.get("agent_id", "") if isinstance(subtask, dict) else "",
                    agent_version=subtask.get("agent_version", "")
                    if isinstance(subtask, dict)
                    else "",
                    task_id=task_id,
                    payload={"text": text},
                )
        except Exception as exc:
            logger.warning("memory_commit_failed", error=str(exc))

    async def _extract_and_store_facts(
        self, state: Any, blackboard: dict, text: str, conversation_id: str, run_id: str
    ) -> None:
        """LLM-extract durable facts from a complete answer and upsert each."""
        # Gate: only extract from a real, complete answer.
        if getattr(state, "partial", False) or not (text or "").strip():
            return
        if self._llm is None or self._mongo is None or not getattr(self._mongo, "enabled", False):
            return
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
                await self._mongo.upsert_fact(
                    scope=f["scope"],
                    key=f["key"],
                    value=f["value"],
                    thread_id=conversation_id,
                    run_id=run_id,
                )
            logger.info("memory_facts_extracted", count=len(facts))
        except Exception as exc:
            logger.warning("memory_facts_extract_failed", error=str(exc))

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
