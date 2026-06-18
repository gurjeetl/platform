"""RAG service protocols."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse


@runtime_checkable
class RetrievalService(Protocol):
    """Read side of RAG — answers a RetrievalRequest with scored chunks."""

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


@runtime_checkable
class IngestionService(Protocol):
    """Write side of RAG — indexes document content for later retrieval."""

    async def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> None: ...


__all__ = ["RetrievalService", "IngestionService"]
