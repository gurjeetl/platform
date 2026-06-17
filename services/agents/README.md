# Genie Example Agents

Reference distributed A2A agents + an MCP tool server, built on the
`genie-agent-sdk`. These prove the distributed agent-to-agent path end-to-end
and serve as the canonical examples for building new agents.

Three agents:

| Agent   | Port | Capability                                              |
|---------|------|---------------------------------------------------------|
| weather | 2010 | Current weather for a named city (`get_weather`)        |
| outage  | 2011 | List / describe grid outages (outage MCP tools)         |
| rag     | 2012 | Doc Q&A over the platform docs (`search_docs` + RAG)    |

The MCP tool server runs separately on **port 2002** and exposes:
`get_weather`, `list_outage_ids`, `get_outage_metadata`,
`get_outage_analysis_summary`, `get_outage_attribute_analysis`,
`get_linked_outages`, `get_outage_report_summary`, `search_docs`.

## Prerequisites

- The Registry / discovery service running at `REGISTRY_URL`
  (default `http://127.0.0.1:2005`).
- An OpenAI-compatible `OPENAI_API_KEY` in the environment (the RAG agent calls
  an LLM; weather/outage only need the MCP server).

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY`.

## Run

All commands are run from `D:\genie-platform`. Each process reads its config
from the environment (the SDK loads `.env` automatically).

### 1. Start the MCP tool server (port 2002)

```bash
uv run --no-project python services/agents/mcp_weather_server.py
```

Endpoint: `http://127.0.0.1:2002/mcp` (transport: `streamable_http`).

### 2. Start each agent (separate terminals)

Each agent self-registers with the Registry on startup and heartbeats; the
platform then discovers it. Point every agent at the MCP server and Registry,
and give each its own `AGENT_PORT`.

**Weather agent (port 2010):**

```bash
AGENT_PORT=2010 \
MCP_SERVER_URL=http://127.0.0.1:2002/mcp MCP_TRANSPORT=streamable_http \
REGISTRY_URL=http://127.0.0.1:2005 \
uv run --no-project python services/agents/weather_agent.py
```

**Outage agent (port 2011):**

```bash
AGENT_PORT=2011 \
MCP_SERVER_URL=http://127.0.0.1:2002/mcp MCP_TRANSPORT=streamable_http \
REGISTRY_URL=http://127.0.0.1:2005 \
uv run --no-project python services/agents/outage_agent.py
```

**RAG agent (port 2012):**

```bash
AGENT_PORT=2012 \
OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini \
MCP_SERVER_URL=http://127.0.0.1:2002/mcp MCP_TRANSPORT=streamable_http \
REGISTRY_URL=http://127.0.0.1:2005 \
uv run --no-project python services/agents/rag_agent.py
```

On Windows PowerShell, set the env vars first (`$env:AGENT_PORT="2010"`) then
run the `uv run` command, or just put values in `.env`.

Each agent exposes:

- `POST /a2a` — A2A JSON-RPC (`message/send`)
- `GET /.well-known/agent.json` — the Agent Card
- `GET /health`

## Notes

- `search_docs` is backed by a small **bundled in-memory doc corpus** inside
  `mcp_weather_server.py` (the `DOCS` list) with a compact dependency-free BM25
  ranker, so the server is fully self-contained.
- `Data.Json` (the outage dataset) is bundled in this directory and read by the
  outage MCP tools. Override its path with `OUTAGE_DATA_PATH`.
- Transport is `streamable_http` to match the SDK's MCP client. The server can
  also serve `sse` (`mcp.run(transport="sse")` → `http://127.0.0.1:2002/sse`).
