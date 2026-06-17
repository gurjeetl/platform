"""Factory that builds the multi-store MemoryFacade from settings.

Mirrors the genie.rag.factory style. Every per-store build is wrapped so an
import or connection error degrades that store to absent (logged) and never
raises — the facade then simply no-ops for that backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from genie.observability.logging import get_logger

if TYPE_CHECKING:
    from genie.memory.adapters.redis import RedisStore
    from genie.memory.facade import MemoryFacade

logger = get_logger(__name__)


def create_memory(settings: Any, llm: Any = None) -> MemoryFacade | None:
    """Build the MemoryFacade for the configured backend.

    Returns None when ``memory_backend == "in_memory"``. For "mongo", builds a
    MongoStore plus a VectorStore (when Milvus is configured) and a RedisStore
    (when ``redis_url`` is set), then returns a MemoryFacade. Per-store failures
    degrade (log warning) — this function never raises.
    """
    backend = getattr(settings, "memory_backend", "in_memory")
    if backend == "in_memory":
        return None

    from genie.memory.facade import MemoryFacade

    mongo = None
    try:
        from genie.memory.adapters.mongo import MongoStore

        mongo = MongoStore(
            uri=getattr(settings, "mongodb_uri", "mongodb://localhost:27017"),
            db=getattr(settings, "mongodb_db", "agent_memory"),
        )
    except Exception as exc:
        logger.warning("memory_mongo_build_failed", error=str(exc))
        mongo = None

    vector = None
    milvus_uri = getattr(settings, "milvus_uri", None)
    milvus_db_path = getattr(settings, "milvus_db_path", None)
    if milvus_uri or milvus_db_path:
        try:
            from genie.memory.adapters.vector import VectorStore

            vector = VectorStore(
                milvus_uri=milvus_uri,
                milvus_db_path=milvus_db_path,
                collection=getattr(settings, "milvus_collection", "long_term_memory"),
                embed_model=getattr(settings, "openai_embed_model", "text-embedding-3-small"),
            )
        except Exception as exc:
            logger.warning("memory_vector_build_failed", error=str(exc))
            vector = None

    redis = create_redis(settings)

    return MemoryFacade(mongo=mongo, vector=vector, redis=redis, llm=llm)


def create_redis(settings: Any) -> RedisStore | None:
    """Standalone RedisStore helper (blackboard mirror) — usable even when
    ``memory_backend`` is in_memory. Returns None when ``redis_url`` is unset or
    the build fails. Lazy/degrade-safe."""
    redis_url = getattr(settings, "redis_url", None)
    if not redis_url:
        return None
    try:
        from genie.memory.adapters.redis import RedisStore

        return RedisStore(redis_url=redis_url)
    except Exception as exc:
        logger.warning("memory_redis_build_failed", error=str(exc))
        return None
