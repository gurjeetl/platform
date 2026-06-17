"""Multi-store memory adapters (Mongo, Milvus vector, Redis).

Each adapter imports its driver lazily and degrades gracefully (``enabled``
flag) when the driver or its configuration is absent — so the dev environment,
which has no motor/redis/pymilvus installed, still imports and runs.
"""

from __future__ import annotations

__all__ = ["MongoStore", "VectorStore", "RedisStore"]


def __getattr__(name: str):  # pragma: no cover - thin lazy re-export
    if name == "MongoStore":
        from genie.memory.adapters.mongo import MongoStore

        return MongoStore
    if name == "VectorStore":
        from genie.memory.adapters.vector import VectorStore

        return VectorStore
    if name == "RedisStore":
        from genie.memory.adapters.redis import RedisStore

        return RedisStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
