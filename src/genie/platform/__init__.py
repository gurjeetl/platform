"""Genie platform core: config, errors, event bus, and feature flags."""

from .config import Settings, get_settings, override_settings
from .errors import ErrorCode, ErrorResponse, GenieError, error_response
from .event_bus import (
    Event,
    EventBus,
    TOPIC_AGENT_EXECUTED,
    TOPIC_AUDIT,
    TOPIC_METRICS,
    TOPIC_RAG_INGESTION_COMPLETED,
    TOPIC_WORKFLOW_COMPLETED,
    TOPIC_WORKFLOW_FAILED,
    TOPIC_WORKFLOW_STARTED,
)
from .feature_flags import FeatureFlags

__all__ = [
    # config
    "Settings",
    "get_settings",
    "override_settings",
    # errors
    "ErrorCode",
    "GenieError",
    "ErrorResponse",
    "error_response",
    # event_bus
    "EventBus",
    "Event",
    "TOPIC_AUDIT",
    "TOPIC_METRICS",
    "TOPIC_RAG_INGESTION_COMPLETED",
    "TOPIC_AGENT_EXECUTED",
    "TOPIC_WORKFLOW_STARTED",
    "TOPIC_WORKFLOW_COMPLETED",
    "TOPIC_WORKFLOW_FAILED",
    # feature_flags
    "FeatureFlags",
]
