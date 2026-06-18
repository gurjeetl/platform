"""Observability base class: auto-trace a component's ``run()`` in an MLflow span.

This mirrors the framework's ``Observable`` contract: any class that inherits
:class:`Observable` has its ``run()`` method (and any other names listed in
``_traced_methods``) automatically wrapped in an MLflow span, and gains a small
``log`` / ``log_event`` observer surface used by the LLM/MCP clients.

mlflow is a hard dependency of the SDK (imported at module top, like the
reference). Span *operations* are still wrapped defensively so a tracing-backend
hiccup never breaks an actual agent run — matching the platform's ``node_span``.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import time
from typing import Any, Callable, Iterator

import mlflow
from mlflow.entities import SpanType

_log = logging.getLogger("genie_agent_sdk.observable")

_TRACED_FLAG = "_genie_traced"

# Levels accepted by ``log`` mapped to stdlib logging levels.
_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "warn": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


@contextlib.contextmanager
def _component_span(name: str, span_type: str) -> Iterator[Any]:
    """Open an MLflow span for ``name``; degrade to a stdlib timing span on failure.

    Never suppresses exceptions raised by the wrapped body: if the body raises,
    the MLflow span records the error (via its own ``__exit__``) and the
    exception propagates unchanged. Span *setup* failures (no backend, version
    skew) fall back to a stdlib-logging timing span so the run continues.
    """
    start = time.perf_counter()
    with contextlib.ExitStack() as stack:
        span = None
        try:
            span = stack.enter_context(mlflow.start_span(name=name, span_type=span_type))
        except Exception:  # pragma: no cover - backend down / no active trace
            span = None
        try:
            yield span
        finally:
            if span is None:
                _log.debug(
                    "component.span name=%s elapsed_ms=%.1f",
                    name,
                    (time.perf_counter() - start) * 1000,
                )


def _span_name(instance: object, method_name: str) -> str:
    """Span label like ``WeatherAgent.run`` — resolved from the concrete class."""
    return f"{type(instance).__name__}.{method_name}"


def _wrap_traced(method_name: str, fn: Callable) -> Callable:
    """Wrap an instance method so each call runs inside a component span."""
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            span_type = getattr(self, "_span_type", SpanType.AGENT)
            with _component_span(_span_name(self, method_name), span_type):
                return await fn(self, *args, **kwargs)

    else:

        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            span_type = getattr(self, "_span_type", SpanType.AGENT)
            with _component_span(_span_name(self, method_name), span_type):
                return fn(self, *args, **kwargs)

    setattr(wrapper, _TRACED_FLAG, True)
    return wrapper


class Observable:
    """Mixin that auto-wraps a component's ``run()`` in an MLflow span.

    Subclasses inherit tracing for free and also gain ``log`` / ``log_event``,
    so they can be passed as the ``observer`` to the LLM/MCP clients. Override
    ``_span_type`` to label the span (``SpanType.AGENT``, ``SpanType.CHAIN``, …)
    or ``_traced_methods`` to trace more entry points than ``run``.
    """

    #: Span type recorded for this component's traced methods.
    _span_type: str = SpanType.AGENT

    #: Human label for the kind of component (used in event attributes).
    _component_kind: str = "agent"

    #: Names of instance methods to auto-wrap in a span on each subclass.
    _traced_methods: tuple[str, ...] = ("run",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Wrap each ``_traced_methods`` entry defined on this subclass exactly once."""
        super().__init_subclass__(**kwargs)
        for name in cls._traced_methods:
            fn = cls.__dict__.get(name)
            # Only wrap plain methods declared on *this* class; inherited methods
            # are already wrapped on the class that defined them.
            if fn is None or not inspect.isfunction(fn):
                continue
            if getattr(fn, _TRACED_FLAG, False):
                continue
            setattr(cls, name, _wrap_traced(name, fn))

    # ------------------------------------------------------------------
    # Observer surface (satisfies genie_agent_sdk.llm_client.Observer)
    # ------------------------------------------------------------------
    def log(self, level: str, event: str, **attrs: Any) -> None:
        """Emit an event at ``level`` via stdlib logging + annotate the active span."""
        exc_info = bool(attrs.pop("exc_info", False))
        _log.log(_LEVELS.get(level.lower(), logging.INFO), "%s %s", event, attrs, exc_info=exc_info)
        self._annotate_span(event, attrs)

    def log_event(self, name: str, **attrs: Any) -> None:
        """Emit a named debug-level event + annotate the active span."""
        _log.debug("%s %s", name, attrs)
        self._annotate_span(name, attrs)

    @staticmethod
    def _annotate_span(event: str, attrs: dict[str, Any]) -> None:
        """Best-effort: record ``event`` on the active MLflow span. Never raises."""
        with contextlib.suppress(Exception):
            span = mlflow.get_current_active_span()
            if span is None:
                return
            span.set_attribute(f"event.{event}", {k: str(v) for k, v in attrs.items()})
