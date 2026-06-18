"""In-process async event bus with a broker-shaped interface."""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Awaitable, Callable

import structlog
from pydantic import BaseModel, Field

# ── Topic constants ───────────────────────────────────────────────────────────
TOPIC_AUDIT = "audit.action"
TOPIC_METRICS = "metrics.recorded"
TOPIC_RAG_INGESTION_COMPLETED = "rag.ingestion.completed"
TOPIC_AGENT_EXECUTED = "agent.executed"
TOPIC_WORKFLOW_STARTED = "workflow.started"
TOPIC_WORKFLOW_COMPLETED = "workflow.completed"
TOPIC_WORKFLOW_FAILED = "workflow.failed"

# ── Types ─────────────────────────────────────────────────────────────────────
EventHandler = Callable[["Event"], Awaitable[None]]


class Event(BaseModel):
    """An event published to the bus."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    topic: str
    payload: dict[str, Any] = {}
    correlation_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Bus ───────────────────────────────────────────────────────────────────────


class EventBus:
    """Simple in-process async publish/subscribe event bus.

    All handler exceptions are caught and logged individually so that a
    failing subscriber never prevents other subscribers from receiving the
    event.
    """

    def __init__(self) -> None:
        """Initialise empty subscriber tables and a dedicated structlog logger."""
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)
        self._started: bool = False
        # Use structlog directly to avoid import-time circular dependency with
        # genie.observability.logging (which imports from correlation, etc.)
        self._logger = structlog.get_logger(__name__)

    # ── Subscription ──────────────────────────────────────────────────────────

    def subscribe(self, topic: str, handler: EventHandler) -> None:
        """Register *handler* to receive events published on *topic*."""
        self._handlers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: EventHandler) -> None:
        """Remove *handler* from *topic* (no-op if not registered)."""
        try:
            self._handlers[topic].remove(handler)
        except ValueError:
            pass

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        correlation_id: str = "",
    ) -> None:
        """Create an Event and fan-out to all subscribers concurrently."""
        event = Event(topic=topic, payload=payload, correlation_id=correlation_id)
        handlers = list(self._handlers.get(topic, []))

        if not handlers:
            return

        async def _safe_call(handler: EventHandler) -> None:
            try:
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "event_handler_error",
                    topic=topic,
                    event_id=event.event_id,
                    handler=getattr(handler, "__name__", repr(handler)),
                    error=str(exc),
                    exc_info=exc,
                )

        await asyncio.gather(*(_safe_call(h) for h in handlers))

    # ── Lifecycle (no-op for in-process bus) ──────────────────────────────────

    async def start(self) -> None:
        """No-op for the in-process bus; required for interface compatibility."""
        self._started = True

    async def stop(self) -> None:
        """No-op for the in-process bus; required for interface compatibility."""
        self._started = False
