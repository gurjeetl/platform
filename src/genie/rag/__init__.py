"""RAG service protocols."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse


@runtime_checkable
class RetrievalService(Protocol):
    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse: ...


@runtime_checkable
class IngestionService(Protocol):
    async def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> None: ...


__all__ = ["RetrievalService", "IngestionService"]
