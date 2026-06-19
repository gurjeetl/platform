"""Unit tests for the memory subsystem.

Covers the existing in-memory stores plus the ported multi-store MemoryFacade
and factory. The real drivers (motor/redis/pymilvus) are NOT installed in this
environment, so the facade/factory tests use hand-written async stub stores and
never construct a real adapter against a live backend.
"""

from genie.application.state import GraphState
from genie.llm.base import LLMResponse
from genie.memory.facade import MemoryFacade
from genie.memory.factory import create_memory, create_redis
from genie.memory.in_memory import InMemoryLongTermStore, InMemorySessionStore

# ── existing in-memory store tests (ported unchanged) ─────────────────────────


async def test_session_store_set_get() -> None:
    store = InMemorySessionStore()
    await store.set("conv1", "foo", "bar")
    assert await store.get("conv1", "foo") == "bar"


async def test_session_store_get_missing_returns_none() -> None:
    store = InMemorySessionStore()
    assert await store.get("conv1", "missing") is None


async def test_session_store_get_all() -> None:
    store = InMemorySessionStore()
    await store.set("conv1", "a", 1)
    await store.set("conv1", "b", 2)
    data = await store.get_all("conv1")
    assert data == {"a": 1, "b": 2}


async def test_session_store_clear() -> None:
    store = InMemorySessionStore()
    await store.set("conv1", "x", 99)
    await store.clear("conv1")
    assert await store.get("conv1", "x") is None


async def test_long_term_store_save_get() -> None:
    store = InMemoryLongTermStore()
    await store.save("user1", "pref", {"theme": "dark"})
    val = await store.get("user1", "pref")
    assert val == {"theme": "dark"}


async def test_long_term_store_search() -> None:
    store = InMemoryLongTermStore()
    await store.save("user1", "energy_policy", "Carbon reduction targets 2030")
    await store.save("user1", "weather_notes", "Cold snaps in Chicago")
    results = await store.search("user1", "energy")
    assert any("energy" in r["key"] for r in results)


async def test_long_term_store_delete() -> None:
    store = InMemoryLongTermStore()
    await store.save("user1", "temp_key", "value")
    deleted = await store.delete("user1", "temp_key")
    assert deleted is True
    assert await store.get("user1", "temp_key") is None


# ── hand-written async stub stores (no external mocking libs needed) ──────────


class StubMongo:
    enabled = True

    def __init__(self, facts: dict[str, str] | None = None) -> None:
        self._facts = facts or {}
        self.commits: list[dict] = []
        self.upserts: list[dict] = []
        self.queried: list[str] = []

    async def query_facts(self, thread_id: str) -> dict[str, str]:
        self.queried.append(thread_id)
        return dict(self._facts)

    async def commit(self, **kwargs) -> None:
        self.commits.append(kwargs)

    async def upsert_fact(self, **kwargs) -> None:
        self.upserts.append(kwargs)


class StubVector:
    def __init__(self, enabled: bool = True, hits: list[dict] | None = None) -> None:
        self.enabled = enabled
        self._hits = hits or [{"content": "remembered"}]
        self.added: list[tuple[str, str]] = []

    async def search(self, thread_id: str, query: str, limit: int = 5) -> list[dict]:
        return list(self._hits)

    async def add(self, thread_id: str, text: str) -> dict:
        self.added.append((thread_id, text))
        return {"enabled": True, "inserted": True}


class StubLLM:
    """Async LLM stub returning a fixed facts JSON payload."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, messages, *, max_tokens=1024, temperature=0.7, **kwargs):
        return LLMResponse(content=self._content, model="stub")


class StubSettings:
    def __init__(self, **kwargs) -> None:
        self.memory_backend = kwargs.get("memory_backend", "in_memory")
        self.mongodb_uri = kwargs.get("mongodb_uri", "mongodb://localhost:27017")
        self.mongodb_db = kwargs.get("mongodb_db", "agent_memory")
        self.redis_url = kwargs.get("redis_url", None)
        self.milvus_uri = kwargs.get("milvus_uri", None)
        self.milvus_db_path = kwargs.get("milvus_db_path", None)
        self.milvus_collection = kwargs.get("milvus_collection", "long_term_memory")
        self.openai_embed_model = kwargs.get("openai_embed_model", "text-embedding-3-small")


# ── factory tests ──────────────────────────────────────────────────────────


def test_create_memory_in_memory_returns_none() -> None:
    assert create_memory(StubSettings(memory_backend="in_memory")) is None


def test_create_redis_none_when_unset() -> None:
    assert create_redis(StubSettings(redis_url=None)) is None


def test_create_memory_mongo_degrades_without_drivers() -> None:
    # motor/pymilvus/redis are not installed → stores degrade, facade still built.
    facade = create_memory(
        StubSettings(memory_backend="mongo", milvus_db_path="./x.db", redis_url="redis://x")
    )
    assert isinstance(facade, MemoryFacade)
    # redis attribute is exposed for bootstrap (degraded but present object or None).
    assert hasattr(facade, "redis")


# ── facade tests with stubs ──────────────────────────────────────────────────


async def test_recall_returns_vector_hits() -> None:
    facade = MemoryFacade(vector=StubVector(hits=[{"content": "abc"}]))
    out = await facade.recall("conv1", "what happened")
    assert out == [{"content": "abc"}]


async def test_recall_empty_when_vector_disabled() -> None:
    facade = MemoryFacade(vector=StubVector(enabled=False))
    assert await facade.recall("conv1", "q") == []


async def test_recall_empty_when_no_vector() -> None:
    facade = MemoryFacade()
    assert await facade.recall("conv1", "q") == []


async def test_query_facts_from_mongo() -> None:
    mongo = StubMongo(facts={"name": "Ada"})
    facade = MemoryFacade(mongo=mongo)
    assert await facade.query_facts("conv1") == {"name": "Ada"}
    assert mongo.queried == ["conv1"]


async def test_query_facts_empty_when_no_mongo() -> None:
    facade = MemoryFacade()
    assert await facade.query_facts("conv1") == {}


async def test_writeback_commits_embeds_and_extracts_facts() -> None:
    mongo = StubMongo()
    vector = StubVector()
    llm = StubLLM('{"facts":[{"key":"home_city","value":"Chicago","scope":"global"}]}')
    facade = MemoryFacade(mongo=mongo, vector=vector, llm=llm)

    state = GraphState(conversation_id="conv1", run_id="run1")
    state.messages = [
        __import__("genie.application.state", fromlist=["Message"]).Message(
            role="user", content="where do I live"
        )
    ]
    blackboard = {"t1": {"agent_id": "weather", "text": "It is sunny in Chicago."}}

    await facade.writeback(state, blackboard, "You live in Chicago.")

    # (a) committed the task
    assert len(mongo.commits) == 1
    assert mongo.commits[0]["task_id"] == "t1"
    # (b) embedded the answer
    assert vector.added == [("conv1", "You live in Chicago.")]
    # (c) extracted + upserted the fact
    assert len(mongo.upserts) == 1
    assert mongo.upserts[0]["key"] == "home_city"
    assert mongo.upserts[0]["scope"] == "global"


async def test_writeback_returns_db_ops_for_each_write() -> None:
    """writeback reports a trace db_op per real datastore write so the
    Synthesizer step can surface them in the trace UI."""
    mongo = StubMongo()
    vector = StubVector()
    llm = StubLLM('{"facts":[{"key":"home_city","value":"Chicago","scope":"global"}]}')
    facade = MemoryFacade(mongo=mongo, vector=vector, llm=llm)

    state = GraphState(conversation_id="conv1", run_id="run1")
    state.messages = [
        __import__("genie.application.state", fromlist=["Message"]).Message(
            role="user", content="where do I live"
        )
    ]
    blackboard = {"t1": {"agent_id": "weather", "text": "It is sunny in Chicago."}}

    ops = await facade.writeback(state, blackboard, "You live in Chicago.")

    commit_ops = [o for o in ops if o["store"] == "mongodb" and o["detail"].startswith("commit ")]
    assert commit_ops and commit_ops[0]["detail"] == "commit t1"
    assert any(o["store"] == "milvus" and o["op"] == "write" for o in ops)
    fact_ops = [o for o in ops if o["store"] == "mongodb" and "fact" in o["detail"]]
    assert fact_ops and fact_ops[0]["hits"] == ["home_city"]


async def test_writeback_returns_empty_without_backends() -> None:
    """No stores configured → nothing written → no db_ops to report."""
    facade = MemoryFacade()
    assert await facade.writeback(GraphState(conversation_id="c"), {"t1": {"text": "x"}}, "a") == []


async def test_writeback_skips_facts_when_partial() -> None:
    mongo = StubMongo()
    llm = StubLLM('{"facts":[{"key":"x","value":"y","scope":"session"}]}')
    facade = MemoryFacade(mongo=mongo, llm=llm)
    state = GraphState(conversation_id="conv1")
    state.partial = True
    await facade.writeback(state, {}, "some text")
    assert mongo.upserts == []


async def test_writeback_best_effort_swallows_errors() -> None:
    class Boom:
        enabled = True

        async def commit(self, **kwargs):
            raise RuntimeError("db down")

        async def upsert_fact(self, **kwargs):
            raise RuntimeError("db down")

    facade = MemoryFacade(mongo=Boom())
    state = GraphState(conversation_id="conv1")
    # Must not raise.
    await facade.writeback(state, {"t1": {"text": "x"}}, "answer")
