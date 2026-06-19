"""PlannerNode DAG construction + validation."""

import pytest

from genie.application.nodes.planner import PlannerNode
from genie.application.prompts import DEFAULT_SYSTEM_PROMPT
from genie.application.state import GraphState, Message
from genie.platform.config import Settings
from tests.conftest import StubLLM


def _state(text: str) -> GraphState:
    return GraphState(conversation_id="c", messages=[Message(role="user", content=text)])


def test_system_prompt_uses_app_persona_or_platform_default(settings, agent_registry):
    """The node injects the app-provided persona when set, else the platform default."""
    agents = agent_registry.list_all()

    # No app persona configured → platform's generic default is used.
    default_node = PlannerNode(agent_registry, settings, StubLLM())
    default_prompt = default_node._build_system_prompt(_state("x"), agents, [], {})
    assert DEFAULT_SYSTEM_PROMPT in default_prompt

    # App supplies its own persona/context → those appear, default is suppressed.
    app_settings = Settings(
        app_system_prompt="You are ACME bot.",
        app_system_context="ACME handles widgets.",
    )
    app_node = PlannerNode(agent_registry, app_settings, StubLLM())
    app_prompt = app_node._build_system_prompt(_state("x"), agents, [], {})
    assert "You are ACME bot." in app_prompt and "ACME handles widgets." in app_prompt
    assert DEFAULT_SYSTEM_PROMPT not in app_prompt


@pytest.mark.asyncio
async def test_builds_multi_task_plan(settings, agent_registry):
    plan_json = (
        '{"subtasks":['
        '{"id":"t1","agent_id":"weather","args":{"location":"tokyo"},"depends_on":[]},'
        '{"id":"t2","agent_id":"outage","args":{},"depends_on":[]}]}'
    )
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
    )
    out = await node(_state("weather in tokyo and top outages"))
    ids = [t["agent_id"] for t in out["plan"]["subtasks"]]
    assert ids == ["weather", "outage"]
    assert out["agent_versions"] == {"t1": "1.0.0", "t2": "1.0.0"}


@pytest.mark.asyncio
async def test_drops_unknown_agent_and_invalid_args(settings, agent_registry):
    plan_json = (
        '{"subtasks":['
        '{"id":"t1","agent_id":"ghost","args":{},"depends_on":[]},'
        '{"id":"t2","agent_id":"weather","args":{},"depends_on":[]}]}'  # missing required location
    )
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
    )
    out = await node(_state("bad plan"))
    assert out["plan"]["subtasks"] == []


@pytest.mark.asyncio
async def test_unparseable_response_yields_error(settings, agent_registry):
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: "not json"),
    )
    out = await node(_state("hi"))
    assert out["plan"]["subtasks"] == []
    assert "error" in out


class _FakeStore:
    def __init__(self, enabled):
        self.enabled = enabled


class _FakeMemory:
    """Memory whose recall/query_facts return canned data; stores report enabled."""

    def __init__(self, *, vector_enabled, mongo_enabled):
        self.vector = _FakeStore(vector_enabled)
        self.mongo = _FakeStore(mongo_enabled)

    async def recall(self, conversation_id, query):
        return [{"content": "Outage 18553223 was a transmission fault."}]

    async def query_facts(self, conversation_id):
        return {"home_city": "Minneapolis", "outage_id": "18553223"}


@pytest.mark.asyncio
async def test_emits_recall_db_ops(settings, agent_registry):
    plan_json = '{"subtasks":[{"id":"t1","agent_id":"outage","args":{},"depends_on":[]}]}'
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
        memory=_FakeMemory(vector_enabled=True, mongo_enabled=True),
    )
    out = await node(_state("top outages"))
    ops = out["db_ops"]
    milvus = next(o for o in ops if o["store"] == "milvus")
    assert milvus["op"] == "search" and milvus["enabled"] is True
    assert milvus["hits"] and "transmission" in milvus["hits"][0]
    mongo = next(o for o in ops if o["store"] == "mongodb")
    assert mongo["op"] == "read" and mongo["enabled"] is True
    assert "2 facts" in mongo["detail"]


@pytest.mark.asyncio
async def test_recall_db_ops_mark_disabled_backends(settings, agent_registry):
    plan_json = '{"subtasks":[{"id":"t1","agent_id":"outage","args":{},"depends_on":[]}]}'
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
        memory=_FakeMemory(vector_enabled=False, mongo_enabled=True),
    )
    out = await node(_state("top outages"))
    milvus = next(o for o in out["db_ops"] if o["store"] == "milvus")
    assert milvus["enabled"] is False
    assert "disabled" in milvus["detail"]


@pytest.mark.asyncio
async def test_no_memory_yields_no_db_ops(settings, agent_registry):
    plan_json = '{"subtasks":[{"id":"t1","agent_id":"outage","args":{},"depends_on":[]}]}'
    node = PlannerNode(
        agent_registry=agent_registry,
        settings=settings,
        llm_provider=StubLLM(lambda s, u: plan_json),
    )
    out = await node(_state("top outages"))
    assert out["db_ops"] == []
