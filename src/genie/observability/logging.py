"""Structured logging via structlog."""
from __future__ import annotations

import logging
import logging.config
import sys
from typing import Any

import structlog


class CorrelationProcessor:
    """structlog processor that injects the current correlation ID."""

    def __call__(
        self,
        logger: Any,
        method: str,
        event_dict: dict[str, Any],
    ) -> dict[str, Any]:
        # Lazy import to avoid import-time circular dependency
        from genie.observability.correlation import get_correlation_id

        cid = get_correlation_id()
        if cid:
            event_dict["correlation_id"] = cid
        return event_dict


def _add_logger_name(
    logger: Any, method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Safe logger-name extractor that works with both stdlib and PrintLogger."""
    name = getattr(logger, "name", None)
    if name is None:
        name = event_dict.get("logger", "")
    if name:
        event_dict["logger"] = name
    return event_dict


def configure_logging(debug: bool = False, service_name: str = "genie") -> None:
    """Configure structlog with JSON (production) or Console (debug) rendering.

    Call this once at application startup.
    """

    def _add_service(
        logger: Any, method: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        event_dict.setdefault("service", service_name)
        return event_dict

    shared_processors: list[Any] = [
        _add_service,
        structlog.stdlib.add_log_level,
        _add_logger_name,
        CorrelationProcessor(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if debug:
        renderer: Any = structlog.dev.ConsoleRenderer(colors=False)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.DEBUG if debug else logging.INFO
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Return a structlog BoundLogger bound to *name*."""
    return structlog.get_logger(name)
