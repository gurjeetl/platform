"""Agent providers — the injection seam wired into the platform from ``src/app.py``.

Each provider is ``(*, tool_gateway, settings) -> BaseAgent``. Add a new in-process
agent by creating ``applications/<name>/agent.py`` and appending a provider here.
"""

from __future__ import annotations

from typing import Any

from applications.outage.agent import OutageAgent
from applications.rag.agent import RagAgent
from applications.weather.agent import WeatherAgent


def weather_provider(*, tool_gateway: Any, settings: Any) -> WeatherAgent:
    return WeatherAgent()


def outage_provider(*, tool_gateway: Any, settings: Any) -> OutageAgent:
    return OutageAgent()


def rag_provider(*, tool_gateway: Any, settings: Any) -> RagAgent:
    return RagAgent()


AGENT_PROVIDERS = [weather_provider, outage_provider, rag_provider]
