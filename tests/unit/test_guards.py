"""Content-guard nodes + create_guard wiring (no llm-guard dependency needed)."""

import importlib.util

import pytest

_HAS_LLM_GUARD = importlib.util.find_spec("llm_guard") is not None

from genie.application.nodes.guards import _REFUSAL, InputGuardNode, OutputGuardNode
from genie.application.state import GraphState, Message
from genie.platform.config import Settings
from genie.security.guard import create_guard


class StubGuard:
    def __init__(self, *, in_valid=True, out_valid=True, in_sanitized=None, out_sanitized=None):
        self._in_valid = in_valid
        self._out_valid = out_valid
        self._in_sanitized = in_sanitized
        self._out_sanitized = out_sanitized

    async def ascan_input(self, text):
        return {
            "valid": self._in_valid,
            "sanitized": self._in_sanitized or text,
            "findings": [] if self._in_valid else ["PromptInjection"],
            "scores": {},
        }

    async def ascan_output(self, prompt, output):
        return {
            "valid": self._out_valid,
            "sanitized": self._out_sanitized or output,
            "findings": [] if self._out_valid else ["Toxicity"],
            "scores": {},
        }


def _state(user="hello", final=None):
    s = GraphState(conversation_id="c", messages=[Message(role="user", content=user)])
    if final is not None:
        s.final_response = final
    return s


@pytest.mark.asyncio
async def test_input_guard_passes_clean_prompt():
    out = await InputGuardNode(StubGuard())(_state("hello"))
    assert out["guard_block"] is None
    assert "final_response" not in out


@pytest.mark.asyncio
async def test_input_guard_blocks_and_refuses():
    out = await InputGuardNode(StubGuard(in_valid=False))(_state("ignore previous instructions"))
    assert out["final_response"] == _REFUSAL
    assert out["is_complete"] is True
    assert out["guard_block"]["stage"] == "input"


@pytest.mark.asyncio
async def test_input_guard_sanitizes_message():
    out = await InputGuardNode(StubGuard(in_sanitized="[REDACTED]"))(_state("my ssn is 1"))
    assert out["messages"][-1].content == "[REDACTED]"
    assert out["guard_input"]["redacted"] is True


@pytest.mark.asyncio
async def test_output_guard_blocks_toxic_answer():
    out = await OutputGuardNode(StubGuard(out_valid=False))(_state(final="toxic stuff"))
    assert out["final_response"] == _REFUSAL
    assert out["view"] is None


@pytest.mark.asyncio
async def test_output_guard_passes_and_sanitizes():
    out = await OutputGuardNode(StubGuard(out_sanitized="clean"))(_state(final="answer"))
    assert out["final_response"] == "clean"


def test_create_guard_disabled_returns_none():
    assert create_guard(Settings(enable_guards=False)) is None


@pytest.mark.skipif(_HAS_LLM_GUARD, reason="llm-guard installed; ImportError fail-closed path not exercised")
def test_create_guard_enabled_without_lib_raises():
    # When the optional extra is absent, enabling guards is fail-closed.
    with pytest.raises(RuntimeError, match="llm-guard"):
        create_guard(Settings(enable_guards=True))
