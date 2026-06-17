"""DiscoveryClient parsing + bootstrap discovery bridge reconciliation."""

import pytest

from genie.agents import AgentRegistry
from genie.agents.remote import RemoteAgent
from genie.discovery.agent_meta import AgentMeta
from genie.discovery.client import DiscoveryClient
from genie.interface.bootstrap import _refresh_discovered_agents
from genie.platform.config import Settings


def _settings() -> Settings:
    return Settings(
        agent_mode="distributed",
        registry_url="http://registry:2005",
        registry_cache_ttl_seconds=0.0,
    )


@pytest.mark.asyncio
async def test_discovery_lists_active_agents(httpx_mock):
    httpx_mock.add_response(
        url="http://registry:2005/agents",
        json={
            "agents": [
                {"agent_id": "weather", "endpoint": "http://h:2010", "status": "active"},
                {"agent_id": "old", "endpoint": "http://h:2011", "status": "deprecated"},
            ]
        },
    )
    client = DiscoveryClient(_settings())
    agents = await client.list_active()
    assert [a.agent_id for a in agents] == ["weather"]  # deprecated filtered out


class _StubDiscovery:
    def __init__(self, metas):
        self._metas = metas

    async def list_active(self, *, force_refresh=False):
        return self._metas


@pytest.mark.asyncio
async def test_bridge_registers_and_unregisters_remote_agents():
    reg = AgentRegistry()
    meta = AgentMeta(agent_id="weather", endpoint="http://h:2010", status="active")

    await _refresh_discovered_agents(reg, _StubDiscovery([meta]))
    assert isinstance(reg.get("weather"), RemoteAgent)

    # Agent drops out of discovery → bridge unregisters the RemoteAgent.
    await _refresh_discovered_agents(reg, _StubDiscovery([]))
    assert reg.get("weather") is None
