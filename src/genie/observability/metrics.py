"""Simple in-memory metrics abstraction.

Designed so that a Prometheus or OpenTelemetry backend can replace it by
swapping out the MetricsRecorder implementation and re-assigning the module-
level ``_recorder`` singleton.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any


def _label_key(name: str, labels: dict | None) -> str:
    """Build a stable string key from a metric name and optional label dict."""
    if not labels:
        return name
    label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    return f"{name}{{{label_str}}}"


class MetricsRecorder:
    """Records counters and histograms in memory."""

    def __init__(self) -> None:
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)

    # ── Counters ──────────────────────────────────────────────────────────────

    def increment(
        self,
        name: str,
        value: float = 1.0,
        labels: dict | None = None,
    ) -> None:
        """Increment a named counter by *value* (default 1)."""
        key = _label_key(name, labels)
        self._counters[key] += value

    def get_counter(self, name: str) -> float:
        """Return the current value of a counter (0 if never recorded)."""
        return self._counters.get(name, 0.0)

    # ── Histograms ────────────────────────────────────────────────────────────

    def record(
        self,
        name: str,
        value: float,
        labels: dict | None = None,
    ) -> None:
        """Append *value* to a named histogram."""
        key = _label_key(name, labels)
        self._histograms[key].append(value)

    def get_histogram(self, name: str) -> list[float]:
        """Return all recorded values for a histogram (empty list if none)."""
        return list(self._histograms.get(name, []))

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy of all metrics."""
        return {
            "counters": dict(self._counters),
            "histograms": {k: list(v) for k, v in self._histograms.items()},
        }

    def reset(self) -> None:
        """Clear all metrics — useful in tests."""
        self._counters.clear()
        self._histograms.clear()


# ── Module-level singleton ────────────────────────────────────────────────────
_recorder = MetricsRecorder()


def get_metrics() -> MetricsRecorder:
    """Return the global MetricsRecorder singleton."""
    return _recorder
