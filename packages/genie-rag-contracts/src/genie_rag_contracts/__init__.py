"""Shared contracts for the genie RAG service boundary."""
from .api import API_VERSION, CORRELATION_ID_HEADER
from .ingestion import IngestRequest, IngestJobStatus
from .retrieval import RetrievalRequest, RetrievalResult, RetrievalResponse

__all__ = [
    "API_VERSION",
    "CORRELATION_ID_HEADER",
    "IngestRequest",
    "IngestJobStatus",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalResponse",
]
