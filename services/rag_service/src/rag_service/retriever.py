"""In-memory keyword retriever used by the RAG service in local mode."""
from __future__ import annotations

from typing import Any

from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult


class KeywordRetriever:
    """Simple keyword-overlap retriever backed by an in-memory index."""

    def __init__(self) -> None:
        self._index: list[dict[str, Any]] = []

    def add(
        self,
        document_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Split *content* into chunks, add to index, return chunk IDs."""
        chunks = _split(content)
        chunk_ids: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}:{i}"
            self._index.append(
                {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "content": chunk,
                    "metadata": metadata or {},
                }
            )
            chunk_ids.append(chunk_id)
        return chunk_ids

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        query_terms = set(request.query.lower().split())
        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._index:
            overlap = len(query_terms & set(entry["content"].lower().split()))
            if overlap:
                scored.append((overlap / max(len(query_terms), 1), entry))
        scored.sort(key=lambda t: t[0], reverse=True)
        results = [
            RetrievalResult(
                document_id=e["document_id"],
                chunk_id=e["chunk_id"],
                content=e["content"],
                score=round(s, 4),
                metadata=e["metadata"],
            )
            for s, e in scored[: request.top_k]
        ]
        return RetrievalResponse(
            results=results,
            query=request.query,
            correlation_id=request.correlation_id,
            retrieval_available=True,
        )

    @property
    def document_count(self) -> int:
        ids = {e["document_id"] for e in self._index}
        return len(ids)

    @property
    def chunk_count(self) -> int:
        return len(self._index)


def _split(text: str, max_chars: int = 512) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        if length + len(word) + 1 > max_chars and current:
            chunks.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks or [""]
