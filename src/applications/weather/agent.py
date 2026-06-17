"""In-process weather agent (Genie application plugin).

Implements the platform agent SDK surface (``genie.agents.base``). Returns a static
weather report for a named city — a self-contained reference agent. A production
version would call a weather API or an MCP tool via the injected ``tool_gateway``.
"""

from __future__ import annotations

from typing import Any

from genie.agents.base import AgentInfo, AgentResult, AgentTask, CapabilitySpec

_WEATHER = {
    "paris": "Sunny, 22°C, clear skies",
    "london": "Cloudy, 14°C, light rain expected",
    "tokyo": "Humid, 28°C, chance of thunderstorm",
    "new york": "Partly cloudy, 18°C, mild winds",
    "minneapolis": "Warm and humid, 29°C, small chance of showers",
    "dubai": "Hot and sunny, 41°C, no cloud cover",
}


class WeatherAgent:
    agent_id = "weather"
    name = "weather"
    description = "Reports current weather conditions for a named city."
    capabilities = ["weather"]
    version = "1.0.0"
    enabled = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    async def health_check(self) -> str:
        return "healthy"

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            version=self.version,
            enabled=self.enabled,
            capability_specs=[
                CapabilitySpec(
                    id="weather",
                    display_name="Weather report",
                    description=self.description,
                    routing_keywords=["weather", "forecast", "city"],
                )
            ],
            input_schema={
                "location": {"type": "string", "required": True, "description": "City name"}
            },
            output_schema={"text": {"type": "string", "persist": True}},
            tags=["weather", "forecast", "city"],
            sla_ms=4000,
        )

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> AgentResult:
        loc = str((task.context or {}).get("args", {}).get("location", "")).lower().strip()
        report = _WEATHER.get(loc, f"No weather data available for '{loc or 'that location'}'.")
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            output=f"Weather in {loc.title() or 'the requested city'}: {report}",
            data={"view": {"type": "weather", "city": loc, "report": report}},
        )
