"""Unit tests for EventBus."""

import pytest
from genie.platform.event_bus import Event, EventBus


async def test_publish_calls_subscriber() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("test.topic", handler)
    await bus.publish("test.topic", payload={"key": "value"})
    assert len(received) == 1
    assert received[0].payload == {"key": "value"}


async def test_publish_no_subscribers_is_silent() -> None:
    bus = EventBus()
    await bus.publish("no.subscribers", payload={})


async def test_unsubscribe_removes_handler() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe("t", handler)
    bus.unsubscribe("t", handler)
    await bus.publish("t", payload={})
    assert received == []


async def test_failing_handler_does_not_block_others() -> None:
    bus = EventBus()
    results: list[str] = []

    async def bad_handler(event: Event) -> None:
        raise RuntimeError("boom")

    async def good_handler(event: Event) -> None:
        results.append("ok")

    bus.subscribe("t", bad_handler)
    bus.subscribe("t", good_handler)
    await bus.publish("t", payload={})
    assert results == ["ok"]


async def test_start_stop_no_op() -> None:
    bus = EventBus()
    await bus.start()
    await bus.stop()
