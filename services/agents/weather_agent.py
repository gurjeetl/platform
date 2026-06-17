from genie_agent_sdk import AgentMeta, BaseAgent, FieldSpec, serve_agent


class WeatherAgent(BaseAgent):
    system_prompt = "You are a helpful weather reporter for a travel assistant."
    tool_names: list[str] = ["get_weather"]

    def run(self, state: dict) -> dict:
        city = (state.get("location") or "").lower().strip()
        return self.answer_with_tool(
            state,
            tool_name="get_weather",
            args={"city": city},
            format_text=lambda report: f"Here's the current weather for {city.title()}: {report}",
            city=city,
        )


META = AgentMeta(
    agent_id="weather",
    version="1.0.0",
    capability_tags=["weather", "forecast", "city"],
    description="Reports current weather conditions for a named city.",
    input_schema={
        "location": FieldSpec(type="string", required=True, description="City name."),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Plain-language weather report.", persist=True),
    },
    sla_ms=4000,
)


if __name__ == "__main__":
    # Run this agent as an independent service that self-registers with the
    # Registry Service and exposes the A2A endpoint POST /a2a.
    # Default advisory port: AGENT_PORT=2010 (host/port/registry come from env).
    serve_agent(WeatherAgent(), agent_meta=META)
