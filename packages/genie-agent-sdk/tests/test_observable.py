"""Observable auto-tracing + observer-surface tests.

mlflow is now a hard dependency, so spans are created for real; we assert on
behavior (return values, exception propagation, wrapping mechanics, the log /
log_event surface) rather than on a tracking backend being configured.
"""
import asyncio

import pytest
from mlflow.entities import SpanType

from genie_agent_sdk.observable import _TRACED_FLAG, Observable


def test_run_is_wrapped_on_subclass():
    """A direct subclass has its ``run`` auto-wrapped exactly once."""
    class Agent(Observable):
        def run(self, state):
            return {**state, "ran": True}

    assert getattr(Agent.__dict__["run"], _TRACED_FLAG, False) is True
    assert Agent().run({"x": 1}) == {"x": 1, "ran": True}


def test_default_span_type_is_agent():
    """Observable defaults to an AGENT span type."""
    assert Observable._span_type == SpanType.AGENT


def test_wrapping_does_not_change_behavior_or_swallow_errors():
    """The span wrapper is transparent: return values pass through, errors raise."""
    class Boom(Observable):
        def run(self, state):
            raise ValueError("kaboom")

    with pytest.raises(ValueError, match="kaboom"):
        Boom().run({})


def test_inherited_run_not_double_wrapped():
    """A subclass that does not override ``run`` inherits the single wrapped one."""
    class Base(Observable):
        def run(self, state):
            return "base"

    class Child(Base):
        pass

    assert "run" not in Child.__dict__
    assert Child().run({}) == "base"


def test_overridden_run_is_independently_wrapped():
    """An overriding subclass wraps its own ``run`` (not the parent's)."""
    class Base(Observable):
        def run(self, state):
            return "base"

    class Child(Base):
        def run(self, state):
            return "child"

    assert getattr(Child.__dict__["run"], _TRACED_FLAG, False) is True
    assert Child().run({}) == "child"


def test_extra_traced_methods():
    """``_traced_methods`` lets a subclass trace additional entry points."""
    class Agent(Observable):
        _traced_methods = ("run", "plan")

        def run(self, state):
            return "run"

        def plan(self, state):
            return "plan"

    assert getattr(Agent.__dict__["plan"], _TRACED_FLAG, False) is True
    assert Agent().plan({}) == "plan"


def test_async_run_is_wrapped():
    """Coroutine methods are wrapped with an async span wrapper and awaited."""
    class AsyncAgent(Observable):
        async def run(self, state):
            await asyncio.sleep(0)
            return "async-done"

    assert asyncio.run(AsyncAgent().run({})) == "async-done"


def test_observer_surface_log_and_log_event():
    """``log`` / ``log_event`` are callable and never raise (satisfy Observer)."""
    obs = Observable()
    # No active span: annotation is a best-effort no-op, must not raise.
    obs.log("info", "some.event", a=1, b="two")
    obs.log("error", "some.error", error="boom", exc_info=True)
    obs.log_event("named.event", k="v")
