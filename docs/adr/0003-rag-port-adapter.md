# ADR 0003 — Port+Adapter Pattern for RAG Integration

**Status:** Accepted  
**Date:** 2026-06-06

## Context

The retrieval back-end may change: keyword index today, vector database tomorrow, hybrid search next quarter. The control plane must never be coupled to the retrieval implementation.

## Decision

Define two **ports** in `genie.rag` as `Protocol` classes:

- `RetrievalService` — `retrieve(RetrievalRequest) → RetrievalResponse`
- `IngestionService` — `ingest(content, metadata) → None`

Provide two **adapters**:

- `LocalRAGAdapter` — in-process keyword index; zero dependencies; used in local/test mode.
- `RemoteRAGAdapter` — async `httpx` client talking to `services/rag_service/`; used in production.

A `create_rag_adapter(settings)` **factory** reads `settings.rag_mode` (`"local"` or `"remote"`) and returns the appropriate adapter. Callers (the `ExecutorNode`) hold a reference to the protocol type, not a concrete class.

The data contract (`RetrievalRequest`, `RetrievalResponse`, `RetrievalResult`) lives in the `genie-rag-contracts` package that both the control plane and the RAG service depend on.

## Consequences

**Positive**
- Swapping adapters requires only a config change — no code changes in the pipeline.
- `LocalRAGAdapter` makes the full platform runnable without any infrastructure.
- `RemoteRAGAdapter` degrades gracefully to `retrieval_available=False` on network error.

**Negative**
- Keyword retrieval quality is poor compared to vector similarity; applications that need relevance will need the remote adapter sooner.

## Alternatives considered

- **Direct `httpx` calls in the executor node:** rejected — couples the pipeline to HTTP transport and breaks testability.
- **Single adapter with a feature flag inside it:** rejected — feature flags inside adapters violate the Open/Closed Principle.
