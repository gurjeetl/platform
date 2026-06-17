from typing import Any
from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: dict[str, Any] = {}
    correlation_id: str = ""


class RetrievalResult(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


class RetrievalResponse(BaseModel):
    results: list[RetrievalResult]
    query: str
    correlation_id: str = ""
    retrieval_available: bool = True
