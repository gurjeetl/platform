"""Indexer — handles ingestion requests and delegates to the retriever."""
from __future__ import annotations

import uuid
from datetime import datetime

from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest

from rag_service.retriever import KeywordRetriever


class Indexer:
    """Manages document ingestion into the retriever index."""

    def __init__(self, retriever: KeywordRetriever) -> None:
        self._retriever = retriever

    async def ingest_file(self, request: IngestRequest) -> IngestJobStatus:
        job_id = str(uuid.uuid4())
        try:
            with open(request.document_path) as fh:
                content = fh.read()
            metadata = dict(request.metadata)
            metadata["document_id"] = request.document_id
            chunk_ids = self._retriever.add(
                document_id=request.document_id,
                content=content,
                metadata=metadata,
            )
            return IngestJobStatus(
                job_id=job_id,
                document_id=request.document_id,
                status="completed",
                correlation_id=request.correlation_id,
                completed_at=datetime.utcnow(),
                chunk_count=len(chunk_ids),
            )
        except Exception as exc:
            return IngestJobStatus(
                job_id=job_id,
                document_id=request.document_id,
                status="failed",
                correlation_id=request.correlation_id,
                error=str(exc),
            )

    async def ingest_content(
        self,
        content: str,
        document_id: str | None = None,
        metadata: dict | None = None,
    ) -> IngestJobStatus:
        doc_id = document_id or str(uuid.uuid4())
        job_id = str(uuid.uuid4())
        try:
            chunk_ids = self._retriever.add(
                document_id=doc_id,
                content=content,
                metadata=metadata or {},
            )
            return IngestJobStatus(
                job_id=job_id,
                document_id=doc_id,
                status="completed",
                completed_at=datetime.utcnow(),
                chunk_count=len(chunk_ids),
            )
        except Exception as exc:
            return IngestJobStatus(
                job_id=job_id,
                document_id=doc_id,
                status="failed",
                error=str(exc),
            )
