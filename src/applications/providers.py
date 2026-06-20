"""Agent providers — the injection seam wired into the platform from ``src/app.py``.

Each provider is ``(*, tool_gateway, settings) -> AgentProtocol``. Add a new in-process
agent by creating ``applications/<name>/agent.py`` and appending a provider here.
"""

from __future__ import annotations

from typing import Any

from applications.outage.agent import OutageAgent
from applications.rag.agent import RagAgent
from applications.weather.agent import WeatherAgent


def weather_provider(*, tool_gateway: Any, settings: Any) -> WeatherAgent:
    """Construct the in-process weather agent (injected deps unused by this static agent)."""
    return WeatherAgent()


def outage_provider(*, tool_gateway: Any, settings: Any) -> OutageAgent:
    """Construct the in-process outage agent (injected deps unused by this static agent)."""
    return OutageAgent()


def rag_provider(*, tool_gateway: Any, settings: Any) -> RagAgent:
    """Construct the in-process RAG agent (injected deps unused by this static agent)."""
    return RagAgent()


AGENT_PROVIDERS = [weather_provider, outage_provider, rag_provider]
