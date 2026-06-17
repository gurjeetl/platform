# genie-agent-sdk

A self-registering agent harness for the Genie platform. Subclass `BaseAgent`
(LLM + MCP + memory + tool-calling loop), declare an `AgentMeta`, and run it with
`serve_agent` — the harness exposes the A2A surface (`POST /a2a`,
`GET /.well-known/agent.json`), self-registers with the registry service, and
heartbeats on a TTL loop. It depends on neither `mlflow` nor `genie.*`.

## Build an agent

```python
from genie_agent_sdk import BaseAgent, AgentMeta, FieldSpec, serve_agent

class WeatherAgent(BaseAgent):
    system_prompt = "You report the weather concisely."
    tool_names = ["get_weather"]          # None = all MCP tools, [] = no MCP

    # Optional: override run(state) for custom logic, or rely on the default
    # tool-calling loop. State is a dict; task args are spread in as top-level keys.

META = AgentMeta(
    agent_id="weather",
    description="Reports current weather for a location",
    capability_tags=["weather", "forecast"],
    input_schema={"location": FieldSpec(type="string", required=True)},
    output_schema={"summary": FieldSpec(type="string")},
)

if __name__ == "__main__":
    serve_agent(WeatherAgent(), agent_meta=META, port=8010,
                registry_url="http://127.0.0.1:2005")
```

`serve_agent(agent, *, agent_meta, host=None, port=None, registry_url=None)` runs
a blocking uvicorn server. Unset host/port/registry_url fall back to env
(`AGENT_HOST`, `AGENT_PORT`, `REGISTRY_URL`) then to `127.0.0.1:8010` /
`http://127.0.0.1:2005`.

## A2A wire contract

- Request: JSON-RPC `message/send`, `params.message` is an A2A `Message` whose
  `parts` carry a `DataPart {"args": {...}}` and whose `metadata` carries
  `task_id`, `run_id`, `thread_id`, `blackboard`, `sla_ms`.
- Response: `result` is a `Message` with `role="agent"`,
  `parts=[TextPart(answer), optional DataPart{"view": {...}}]`.

## Environment

LLM: `OPENAI_MODEL`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_TEMPERATURE`.
MCP: `MCP_SERVER_URL`, `MCP_SERVER_NAME`, `MCP_TRANSPORT`, `MCP_AUTH_TOKEN`, `MCP_TIMEOUT`.
Harness: `AGENT_HOST`, `AGENT_PORT`, `AGENT_ADVERTISE_HOST`, `AGENT_ADVERTISE_PORT`,
`REGISTRY_URL`, `REGISTRY_AUTH_TOKEN`, `REGISTRY_HEARTBEAT_SECONDS`,
`REGISTRY_TIMEOUT_S`, `AGENT_INVOKE_TOKEN` (optional bearer guard on `/a2a`).
