"""LocalRAGAdapter — in-process retrieval for zero-dependency mode."""

from __future__ import annotations

from typing import Any

from genie_rag_contracts.ingestion import IngestJobStatus, IngestRequest
from genie_rag_contracts.retrieval import RetrievalRequest, RetrievalResponse, RetrievalResult

from genie.observability.logging import get_logger

logger = get_logger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "with",
    "by",
    "from",
    "and",
    "or",
    "but",
    "not",
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "that",
    "this",
    "it",
    "its",
    "i",
    "me",
    "my",
    "you",
    "your",
    "we",
    "our",
    "they",
    "their",
}


class LocalRAGAdapter:
    """In-process RAG adapter backed by a simple keyword index.

    Suitable for local development and CI without running the RAG service.
    The underlying index is a list of (document_id, chunk_id, content, metadata)
    tuples. Retrieval scores by keyword overlap; ingestion appends to the list.
    """

    def __init__(self) -> None:
        self._index: list[dict[str, Any]] = []

    # ── IngestionService ──────────────────────────────────────────────────────

    async def ingest(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a document to the in-memory index."""
        import uuid

        doc_id = (metadata or {}).get("document_id", str(uuid.uuid4()))
        chunks = _split_chunks(content)
        for i, chunk in enumerate(chunks):
            self._index.append(
                {
                    "document_id": doc_id,
                    "chunk_id": f"{doc_id}:{i}",
                    "content": chunk,
                    "metadata": metadata or {},
                }
            )
        logger.debug("local_rag_ingested", doc_id=doc_id, chunks=len(chunks))

    async def ingest_request(self, request: IngestRequest) -> IngestJobStatus:
        """Ingest a document by path (reads file content from disk)."""
        import uuid

        job_id = str(uuid.uuid4())
        try:
            with open(request.document_path) as fh:
                content = fh.read()
            metadata = dict(request.metadata)
            metadata["document_id"] = request.document_id
            await self.ingest(content, metadata)
            return IngestJobStatus(
                job_id=job_id,
                document_id=request.document_id,
                status="completed",
                correlation_id=request.correlation_id,
                chunk_count=len(_split_chunks(content)),
            )
        except Exception as exc:
            return IngestJobStatus(
                job_id=job_id,
                document_id=request.document_id,
                status="failed",
                correlation_id=request.correlation_id,
                error=str(exc),
            )

    # ── RetrievalService ──────────────────────────────────────────────────────

    async def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        """Return the top-k chunks scored by keyword overlap.

        Stopwords are excluded from scoring so common words like "is", "the",
        "of" don't cause every document to match every query.
        """
        query_terms = set(request.query.lower().split()) - _STOPWORDS

        scored: list[tuple[float, dict[str, Any]]] = []
        for entry in self._index:
            content_terms = set(entry["content"].lower().split())
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                score = overlap / max(len(query_terms), 1)
                scored.append((score, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        top_k = scored[: request.top_k]

        results = [
            RetrievalResult(
                document_id=entry["document_id"],
                chunk_id=entry["chunk_id"],
                content=entry["content"],
                score=round(score, 4),
                metadata=entry["metadata"],
            )
            for score, entry in top_k
        ]

        logger.debug(
            "local_rag_retrieved",
            query=request.query[:80],
            result_count=len(results),
        )

        return RetrievalResponse(
            results=results,
            query=request.query,
            correlation_id=request.correlation_id,
            retrieval_available=True,
        )


def _split_chunks(text: str, max_chars: int = 512) -> list[str]:
    """Split text into chunks of at most max_chars on word boundaries."""
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
