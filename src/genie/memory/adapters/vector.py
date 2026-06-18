"""VectorStore — Milvus semantic long-term memory with OpenAI embeddings.

Ports BaseAgentFramework's vector_store. pymilvus is an optional extra and is
imported lazily inside ``__init__``/methods; ``enabled`` is False when neither a
remote ``milvus_uri`` nor a usable ``milvus_db_path`` is configured, or when
pymilvus is missing — every method then no-ops (search → [], add → no-op).

pymilvus' ``MilvusClient`` and the OpenAI embedding call are synchronous, so the
async ``add``/``search`` methods offload the blocking work to ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from genie.observability.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_DIM = 1536

# Skip re-embedding an answer that's near-identical (cosine >= this) to one
# already stored for the thread — repeated runs of the same prompt produce
# reworded answers that would otherwise pile up as near-duplicate vectors.
_DEDUP_COSINE = 0.93


class VectorStore:
    """Milvus-backed semantic memory. Disabled (no-op) when unconfigured or
    pymilvus is unavailable."""

    def __init__(
        self,
        *,
        milvus_uri: str | None = None,
        milvus_db_path: str | None = None,
        collection: str = "long_term_memory",
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = _DEFAULT_DIM,
    ) -> None:
        """Connect a MilvusClient; degrade to disabled when unconfigured or when
        pymilvus is missing/unreachable."""
        # milvus_uri may be a remote http(s) endpoint; milvus_db_path is a local
        # Milvus Lite file (no server needed). Prefer the explicit remote uri.
        self._uri = milvus_uri or milvus_db_path
        self._collection = collection
        self._embed_model = embed_model
        self._dim = embed_dim
        self._client = None
        self._embeddings = None

        if not self._uri:
            logger.warning("milvus_disabled", reason="no milvus_uri/milvus_db_path")
            return
        # pymilvus' ORM reads os.environ['MILVUS_URI'] at import and only accepts
        # http(s) URIs — a local file path there would crash the import. Move any
        # non-http value out of the env before importing.
        env_uri = os.environ.get("MILVUS_URI")
        if env_uri and not env_uri.lower().startswith("http"):
            os.environ.pop("MILVUS_URI", None)
        try:
            from pymilvus import MilvusClient  # lazy: optional extra
        except ImportError:
            logger.warning("milvus_disabled", reason="pymilvus not installed")
            return
        try:
            self._client = MilvusClient(uri=self._uri, token=os.getenv("MILVUS_TOKEN") or "")
        except Exception as exc:
            logger.warning("milvus_connect_failed", uri=self._uri, error=str(exc))
            self._client = None

    @property
    def enabled(self) -> bool:
        """True when a Milvus client connected (otherwise every method no-ops)."""
        return self._client is not None

    # ── embeddings (OpenAI, installed) ────────────────────────────────────────
    def _embed(self, text: str) -> list[float] | None:
        """Embed text via the OpenAI-compatible endpoint. None on any failure."""
        try:
            if self._embeddings is None:
                from openai import OpenAI  # lazy

                self._embeddings = OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    base_url=os.getenv("OPENAI_BASE_URL") or None,
                )
            resp = self._embeddings.embeddings.create(model=self._embed_model, input=text)
            return list(resp.data[0].embedding)
        except Exception as exc:
            logger.warning("milvus_embed_failed", error=str(exc))
            return None

    def _ensure_collection(self) -> None:
        """Create + load the collection (with a COSINE vector index) on first use;
        a no-op once it already exists."""
        if not self._client:
            return
        try:
            from pymilvus import DataType

            if self._client.has_collection(self._collection):
                self._client.load_collection(self._collection)
                return
            schema = self._client.create_schema(auto_id=True, enable_dynamic_field=False)
            schema.add_field("id", DataType.INT64, is_primary=True)
            schema.add_field("thread_id", DataType.VARCHAR, max_length=256)
            schema.add_field("content", DataType.VARCHAR, max_length=8192)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)

            index_params = self._client.prepare_index_params()
            index_params.add_index(
                field_name="embedding", index_type="AUTOINDEX", metric_type="COSINE"
            )
            self._client.create_collection(
                collection_name=self._collection, schema=schema, index_params=index_params
            )
            self._client.load_collection(self._collection)
            logger.info("milvus_collection_ready", collection=self._collection)
        except Exception as exc:
            logger.warning("milvus_ensure_failed", error=str(exc))

    # ── public async API ──────────────────────────────────────────────────────
    async def search(self, thread_id: str, query: str, limit: int = 5) -> list[dict]:
        """Up to ``limit`` semantically-similar past memories for this thread.

        Each hit is {"content": str}. Empty list when Milvus or embeddings are
        unavailable.
        """
        if not self._client or not query:
            return []
        return await asyncio.to_thread(self._search_sync, thread_id, query, limit)

    def _search_sync(self, thread_id: str, query: str, limit: int) -> list[dict]:
        """Blocking embed + Milvus search (offloaded by ``search``)."""
        vector = self._embed(query)
        if vector is None:
            return []
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[vector],
                limit=limit,
                filter=f'thread_id == "{thread_id}"',
                output_fields=["content"],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as exc:
            logger.warning("milvus_search_failed", error=str(exc))
            return []
        hits = results[0] if results else []
        out: list[dict] = []
        for h in hits:
            entity = h.get("entity", {}) if isinstance(h, dict) else {}
            out.append({"content": entity.get("content", "")})
        return out

    async def add(self, thread_id: str, text: str) -> dict[str, Any]:
        """Embed and insert one memory, skipping near-duplicate re-inserts."""
        if not self._client:
            return {"enabled": False}
        return await asyncio.to_thread(self._add_sync, thread_id, text)

    def _add_sync(self, thread_id: str, text: str) -> dict[str, Any]:
        """Blocking embed + dedup-check + insert (offloaded by ``add``)."""
        self._ensure_collection()
        vector = self._embed(text)
        if vector is None:
            return {"enabled": True, "inserted": False, "reason": "embed_failed"}
        dup = self._nearest_score(thread_id, vector)
        if dup is not None and dup >= _DEDUP_COSINE:
            return {"enabled": True, "inserted": False, "reason": f"duplicate {dup:.3f}"}
        try:
            self._client.insert(
                collection_name=self._collection,
                data=[
                    {
                        "thread_id": thread_id,
                        "content": text[:8192],
                        "embedding": vector,
                    }
                ],
            )
            return {"enabled": True, "inserted": True}
        except Exception as exc:
            logger.warning("milvus_insert_failed", error=str(exc))
            return {"enabled": True, "inserted": False, "reason": str(exc)}

    def _nearest_score(self, thread_id: str, vector: list[float]) -> float | None:
        """Cosine similarity of the closest existing memory, or None if none."""
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[vector],
                limit=1,
                filter=f'thread_id == "{thread_id}"',
                output_fields=[],
                search_params={"metric_type": "COSINE"},
            )
        except Exception as exc:
            logger.warning("milvus_dedup_search_failed", error=str(exc))
            return None
        hits = results[0] if results else []
        if not hits or not isinstance(hits[0], dict):
            return None
        return float(hits[0].get("distance", 0.0))

    async def aclose(self) -> None:
        """Close the Milvus client (offloaded; best-effort shutdown hook)."""
        if self._client is not None:
            try:
                await asyncio.to_thread(self._client.close)
            except Exception:  # noqa: S110 - best-effort close on shutdown
                pass
            self._client = None
