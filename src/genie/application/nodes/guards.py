"""Input/output content-guard nodes that bracket the pipeline.

Ported from BaseAgentFramework ``security/guards.py``. ``InputGuardNode`` runs
before the Router (scans + sanitizes the user prompt); ``OutputGuardNode`` runs
after the Synthesizer (scans the final answer). Blocking reuses the chitchat
fast-path mechanism: set ``final_response`` + ``is_complete`` and let the graph
route straight to END.
"""

from __future__ import annotations

from typing import Any

from genie.application.state import GraphState, Message
from genie.observability.logging import get_logger
from genie.tracking import node_span

logger = get_logger(__name__)

_REFUSAL = (
    "I can't help with that request — it was flagged by our safety filter. "
    "Please rephrase and try again."
)


def _last_user_index(state: GraphState) -> int:
    for i in range(len(state.messages) - 1, -1, -1):
        if state.messages[i].role == "user":
            return i
    return -1


class InputGuardNode:
    """Scan the incoming user prompt; block high-risk content, redact PII/secrets."""

    def __init__(self, guard: Any) -> None:
        self._guard = guard

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        with node_span("input_guard"):
            idx = _last_user_index(state)
            text = state.messages[idx].content if idx >= 0 else ""
            res = await self._guard.ascan_input(text)

            if not res["valid"]:
                logger.warning("input_guard_blocked", findings=res["findings"])
                return {
                    "guard_block": {
                        "stage": "input",
                        "findings": res["findings"],
                        "scores": res["scores"],
                    },
                    "guard_input": {
                        "scanned": True,
                        "blocked": True,
                        "findings": res["findings"],
                        "scores": res["scores"],
                        "redacted": False,
                    },
                    "final_response": _REFUSAL,
                    "is_complete": True,
                    "messages": list(state.messages)
                    + [Message(role="assistant", content=_REFUSAL)],
                }

            out: dict[str, Any] = {
                "guard_block": None,
                "guard_input": {
                    "scanned": True,
                    "blocked": False,
                    "findings": [],
                    "scores": res["scores"],
                    "redacted": res["sanitized"] != text,
                },
            }
            # Replace the user message with the sanitized text so agents never see raw PII.
            if res["sanitized"] != text and idx >= 0:
                msgs = list(state.messages)
                msgs[idx] = Message(role="user", content=res["sanitized"])
                out["messages"] = msgs
            return out


class OutputGuardNode:
    """Scan the synthesized answer before it reaches the user."""

    def __init__(self, guard: Any) -> None:
        self._guard = guard

    async def __call__(self, state: GraphState) -> dict[str, Any]:
        with node_span("output_guard"):
            answer = state.final_response or ""
            if not answer:
                return {}
            idx = _last_user_index(state)
            prompt = state.messages[idx].content if idx >= 0 else ""
            res = await self._guard.ascan_output(prompt, answer)

            if not res["valid"]:
                logger.warning("output_guard_blocked", findings=res["findings"])
                return {
                    "guard_block": {
                        "stage": "output",
                        "findings": res["findings"],
                        "scores": res["scores"],
                    },
                    "guard_output": {
                        "scanned": True,
                        "blocked": True,
                        "findings": res["findings"],
                        "scores": res["scores"],
                        "redacted": False,
                    },
                    "final_response": _REFUSAL,
                    "view": None,
                }
            return {
                "final_response": res["sanitized"],
                "guard_output": {
                    "scanned": True,
                    "blocked": False,
                    "findings": [],
                    "scores": res["scores"],
                    "redacted": res["sanitized"] != answer,
                },
            }
