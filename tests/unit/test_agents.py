"""AgentRegistry + AgentInfo behavior with the distributed-agent protocol."""

import pytest

from genie.agents import AgentRegistry
from genie.platform.errors import GenieError
from tests.conftest import FakeAgent


def test_register_list_and_get(agent_registry: AgentRegistry):
    ids = {a.agent_id for a in agent_registry.list_all()}
    assert ids == {"weather", "outage"}
    assert agent_registry.get("weather").agent_id == "weather"


def test_duplicate_registration_rejected(agent_registry: AgentRegistry):
    with pytest.raises(ValueError):
        agent_registry.register(FakeAgent("weather"))


def test_unregister_unknown_raises(agent_registry: AgentRegistry):
    with pytest.raises(GenieError):
        agent_registry.unregister("ghost")


def test_find_by_capability_skips_disabled(agent_registry: AgentRegistry):
    agent_registry.disable("weather")
    assert agent_registry.find_by_capability("weather") == []


def test_agent_info_validate_args():
    info = FakeAgent(
        "weather", input_schema={"location": {"type": "string", "required": True}}
    ).get_info()
    ok, _ = info.validate_args({"location": "paris"})
    assert ok
    bad, err = info.validate_args({})
    assert not bad and "location" in err
