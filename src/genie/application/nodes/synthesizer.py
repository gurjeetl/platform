"""SynthesizerNode — reads the blackboard and composes one user-facing answer.

Ported from BaseAgentFramework ``synthesizer/synthesizer.py``. Two fast paths
shortcut the LLM:
  - empty plan → friendly clarification.
  - exactly one task with a structured ``view`` → pass it through unchanged
    (preserves the {response, view} chat contract).
Otherwise the LLM merges the blackboard entries into prose. Durable write-back
(facts/commits/embeddings) is delegated to the memory subsystem (Phase 3; no-ops
when ``memory`` is None).
"""

from __future__ import annotations

import contextlib
import json
from typing import Any

from genie.application.dag import Plan
from genie.application.state import GraphState, Message
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are a synthesis agent. You will receive a JSON blackboard whose keys are "
    'task ids and whose values are agent outputs (or {"error": ...} entries). '
    "Compose one concise, helpful answer to the user's original request by merging "
    "the successful outputs. For any blackboard entry that contains an error, mark "
    "that section [PARTIAL] in the final answer. Do not invent facts. Do not include "
    "raw JSON in the output."
)

_CLARIFICATION = "I can help with weather or grid outages. Could you tell me what you need?"


def _last_user_message(state: GraphState) -> str:
    """Return the most recent user message text — the request being answered."""
    for m in reversed(state.messages):
        if m.role == "user":
            return m.content
    return ""


class SynthesizerNode:
    """Composes the final user-facing answer from the blackboard (with fast paths)."""

    def __init__(self, llm_provider: Any, settings: Any, memory: Any | None = None) -> None:
        self._llm = llm_provider
        self._settings = settings
        self._memory = memory

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        """Synthesize the answer and record its length on the trace span."""
        with node_span("synthesizer") as span:
            result = await self._synthesize(state)
            with contextlib.suppress(Exception):
                if span is not None:
                    span.set_outputs({"response_length": len(result.get("final_response") or "")})
            return result

    @staticmethod
    def _emit(state: GraphState, text: str, view: dict | None = None) -> dict[str, Any]:
        """Finalize: append the assistant turn, set final_response + is_complete."""
        messages = list(state.messages) + [Message(role="assistant", content=text)]
        out: dict[str, Any] = {"final_response": text, "messages": messages, "is_complete": True}
        if view:
            out["view"] = view
        return out

    @staticmethod
    def _render_blackboard(
        blackboard: dict[str, dict], per_entry_cap: int = 2500, total_cap: int = 8000
    ) -> str:
        """Serialize the blackboard to a size-capped JSON string for the LLM prompt."""
        parts: list[str] = []
        for tid, entry in blackboard.items():
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

    async def _synthesize(self, state: GraphState) -> dict[str, Any]:
        """Apply the fast paths, else LLM-merge the blackboard into one prose answer."""
        blackboard: dict[str, dict] = state.blackboard or {}
        plan = Plan(**(state.plan or {}))

        # Empty plan → clarification.
        if not plan.subtasks:
            return self._emit(state, _CLARIFICATION)

        successful = [
            (tid, entry)
            for tid, entry in blackboard.items()
            if isinstance(entry, dict) and "error" not in entry
        ]

        # Single task with a structured view → pass through.
        if len(plan.subtasks) == 1 and len(successful) == 1:
            _tid, entry = successful[0]
            view = entry.get("view")
            text = entry.get("text") or ""
            if state.partial:
                text = f"[PARTIAL] {text}".strip()
            await self._writeback(state, blackboard, text)
            return self._emit(state, text, view)

        # Multi-task or no view — LLM-synthesize prose.
        user_input = _last_user_message(state)
        bb_for_prompt = self._render_blackboard(blackboard)
        try:
            response = await self._llm.complete(
                [
                    Message(role="system", content=_SYSTEM_PROMPT),
                    Message(
                        role="user",
                        content=(
                            f"USER REQUEST:\n{user_input}\n\n"
                            f"BLACKBOARD (JSON):\n{bb_for_prompt}\n\n"
                            "Compose the final answer now."
                        ),
                    ),
                ],
                max_tokens=getattr(self._settings, "llm_max_tokens", 1024),
                temperature=getattr(self._settings, "llm_temperature", 0.3),
            )
            text = response.content or ""
        except Exception as exc:  # noqa: BLE001
            logger.error("synthesizer_llm_failed", error=str(exc))
            return self._emit(state, "Could not compose the final answer.")

        if state.partial and "[PARTIAL]" not in text:
            text = f"[PARTIAL] {text}"
        await self._writeback(state, blackboard, text)
        return self._emit(state, text)

    async def _writeback(self, state: GraphState, blackboard: dict[str, dict], text: str) -> None:
        """Durable persistence + semantic embedding + fact extraction (Phase 3)."""
        if self._memory is None or state.partial or not (text or "").strip():
            return
        with contextlib.suppress(Exception):
            await self._memory.writeback(state, blackboard, text)
