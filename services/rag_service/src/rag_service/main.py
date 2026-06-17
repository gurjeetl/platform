"""RAG service entrypoint — builds the FastAPI app and wires dependencies."""
from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from rag_service.api import health_router, ingest_router, retrieve_router
from rag_service.indexer import Indexer
from rag_service.retriever import KeywordRetriever

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    retriever = KeywordRetriever()
    indexer = Indexer(retriever=retriever)
    app.state.retriever = retriever
    app.state.indexer = indexer
    logger.info("rag_service_started")
    yield
    logger.info("rag_service_stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Genie RAG Service",
        description="Standalone retrieval-augmented generation service",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health_router)
    app.include_router(retrieve_router)
    app.include_router(ingest_router)
    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_service.main:app", host="0.0.0.0", port=8001, reload=False)
