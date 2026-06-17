"""Factory that returns the correct RAG adapter based on settings.rag_mode."""
from __future__ import annotations

from typing import Any


def create_rag_adapter(settings: Any) -> Any:
    """Return a LocalRAGAdapter or RemoteRAGAdapter based on settings.rag_mode.

    The returned object satisfies both RetrievalService and IngestionService
    protocols defined in genie.rag.
    """
    mode = getattr(settings, "rag_mode", "local")
    if mode == "remote":
        from genie.rag.adapters.remote import RemoteRAGAdapter

        return RemoteRAGAdapter(
            base_url=settings.rag_service_url,
            timeout=settings.rag_timeout_seconds,
            max_retries=settings.rag_max_retries,
            api_key=getattr(settings, "api_key", None),
        )

    from genie.rag.adapters.local import LocalRAGAdapter

    return LocalRAGAdapter()
