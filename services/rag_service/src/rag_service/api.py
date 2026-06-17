"""FastAPI router definitions for the RAG service."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from genie_rag_contracts.api import INGEST_BASE_PATH, RETRIEVE_PATH
from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest
from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse

retrieve_router = APIRouter(tags=["retrieval"])
ingest_router = APIRouter(tags=["ingestion"])


# ── Retrieval ─────────────────────────────────────────────────────────────────

@retrieve_router.post(RETRIEVE_PATH, response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest, req: Request) -> RetrievalResponse:
    retriever = req.app.state.retriever
    return retriever.retrieve(request)


# ── Ingestion ─────────────────────────────────────────────────────────────────

class ContentIngestRequest(BaseModel):
    document_id: str | None = None
    content: str
    metadata: dict[str, Any] = {}
    correlation_id: str = ""


@ingest_router.post(f"{INGEST_BASE_PATH}/file", response_model=IngestJobStatus)
async def ingest_file(request: IngestRequest, req: Request) -> IngestJobStatus:
    indexer = req.app.state.indexer
    return await indexer.ingest_file(request)


@ingest_router.post(f"{INGEST_BASE_PATH}/content", response_model=IngestJobStatus)
async def ingest_content(body: ContentIngestRequest, req: Request) -> IngestJobStatus:
    indexer = req.app.state.indexer
    return await indexer.ingest_content(
        content=body.content,
        document_id=body.document_id,
        metadata=body.metadata,
    )


# ── Health ────────────────────────────────────────────────────────────────────

health_router = APIRouter(tags=["health"])


@health_router.get("/health")
async def health(req: Request) -> dict[str, Any]:
    retriever = req.app.state.retriever
    return {
        "status": "ok",
        "documents": retriever.document_count,
        "chunks": retriever.chunk_count,
    }
