"""RAG ingest and index-stats endpoints.

Other microservices push documents here; the platform indexes them so the
chat pipeline can retrieve relevant context for user queries.
"""
from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from genie.observability.logging import get_logger

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])
logger = get_logger(__name__)


# ── Request / response models ─────────────────────────────────────────────────

class IngestDocumentRequest(BaseModel):
    title: str
    content: str
    source: str = "manual"
    metadata: dict[str, Any] = {}


class IngestDocumentResponse(BaseModel):
    document_id: str
    title: str
    status: str
    chunks: int


class RagStatsResponse(BaseModel):
    adapter: str
    indexed_chunks: int
    enabled: bool


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_adapter(request: Request) -> Any:
    adapter = getattr(request.app.state, "rag_adapter", None)
    if adapter is None:
        raise HTTPException(
            status_code=503,
            detail="RAG adapter not enabled. Set enable_rag=true in config.",
        )
    return adapter


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestDocumentResponse, summary="Ingest a document")
async def ingest_document(body: IngestDocumentRequest, request: Request) -> IngestDocumentResponse:
    """Add a document to the RAG index.

    Called by other microservices to push knowledge into the platform.
    The content is chunked and indexed immediately; retrieval is available
    on the next query with no restart required.
    """
    adapter = _get_adapter(request)

    doc_id = body.metadata.get("document_id") or str(uuid.uuid4())
    metadata: dict[str, Any] = {
        "document_id": doc_id,
        "title": body.title,
        "source": body.source,
        **body.metadata,
    }

    await adapter.ingest(body.content, metadata)

    # Count chunks that landed in the index (LocalRAGAdapter exposes _index).
    # For RemoteRAGAdapter we fall back to an estimate.
    chunks = sum(
        1
        for entry in getattr(adapter, "_index", [])
        if entry.get("metadata", {}).get("document_id") == doc_id
    ) or max(1, len(body.content) // 512 + 1)

    logger.info("rag_document_ingested", doc_id=doc_id, title=body.title, chunks=chunks)

    return IngestDocumentResponse(
        document_id=doc_id,
        title=body.title,
        status="ingested",
        chunks=chunks,
    )


@router.get("/stats", response_model=RagStatsResponse, summary="RAG index statistics")
async def rag_stats(request: Request) -> RagStatsResponse:
    """Return the number of indexed chunks and the active adapter type."""
    adapter = getattr(request.app.state, "rag_adapter", None)
    if adapter is None:
        return RagStatsResponse(adapter="none", indexed_chunks=0, enabled=False)

    return RagStatsResponse(
        adapter=type(adapter).__name__,
        indexed_chunks=len(getattr(adapter, "_index", [])),
        enabled=True,
    )
