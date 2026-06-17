# Genie Platform

A production-grade, **distributed multi-agent** AI platform: a LangGraph control
plane (router → planner → orchestrator → executor → completion gate → synthesizer,
bracketed by content guards) that plans a DAG of subtasks and dispatches them to
**independent agent services** over A2A JSON-RPC, discovered through a registry.

It combines BaseAgentFramework's functionality (distributed A2A agents, DAG planning
with wave execution + replan, mandatory content guards, multi-store memory, and the
chat + trace-visualizer UIs) with Genie's engineering structure (layered kernel,
config-driven wiring, `import-linter` boundaries, `uv` packaging, and a full test
suite).

## Architecture

```
genie-platform/
├── src/
│   ├── genie/                      # Platform kernel (control plane)
│   │   ├── platform/               # Config (pydantic-settings), errors, event bus
│   │   ├── application/            # LangGraph pipeline: graph, state, dag, blackboard, nodes/
│   │   ├── agents/                 # BaseAgent protocol, AgentRegistry, RemoteAgent (A2A bridge)
│   │   ├── a2a/                    # A2A JSON-RPC message types + client
│   │   ├── discovery/              # AgentMeta + async DiscoveryClient (registry client)
│   │   ├── llm/                    # LLM provider protocol + registry + Mock
│   │   ├── memory/                 # ports + in_memory + adapters (mongo/redis/vector) + facade + factory
│   │   ├── rag/                    # RAG ports + Local/Remote adapters + factory
│   │   ├── security/               # API-key auth + LLMGuard content scanning
│   │   ├── tracking/ observability/ tools/ mcp/
│   │   └── interface/              # FastAPI app factory, routers, bundled static UIs
│   └── applications/               # (empty by default — agents run as services; hybrid mode can inject here)
├── services/
│   ├── registry/                   # Discovery service (registry-service)         :2005
│   ├── agents/                     # Example distributed agents + MCP tool server  :2010-2012, :2002
│   └── rag_service/                # Extracted RAG microservice (optional)         :2004
├── packages/
│   ├── genie-agent-sdk/            # Self-registering agent harness (BaseAgent + serve_agent)
│   └── genie-rag-contracts/        # Shared RAG contract models
├── tests/                          # unit / integration / e2e / contract
├── config/                         # default.yaml + test.yaml + split.yaml
└── docs/adr/                       # Architecture Decision Records (0001–0011)
```

### LangGraph pipeline

```
user message
   │
[input_guard] ──blocked?──▶ END (safe refusal)          (when enable_guards)
   │
[router] ─┬─ chitchat ───────────────────────────────────────▶ [synthesizer]
          ├─ fast ────────────────────────────────────────────▶ [executor]
          └─ plan ─▶ [planner] ─▶ [orchestrator] ─(HITL?)─▶ [executor]
                                                                 │
                                                         [completion_gate]
                                                     replan ▲        ▼ synthesize
                                                     [planner]    [synthesizer]
                                                                     │
                                                              [output_guard] ─▶ END
```

- **planner** turns the prompt into a DAG of subtasks (over the live discovered-agent
  capability menu); **orchestrator** computes dependency waves (Kahn's algorithm);
  **executor** runs each wave concurrently, dispatching every task through the
  `AgentRegistry` — a `RemoteAgent` performs the actual A2A call to the agent service.
- **completion_gate** re-plans on missing/errored tasks (bounded by `max_replans`);
  **synthesizer** merges the blackboard into one answer (+ optional structured view).

### Distributed agents

Agents are **independent services** built on `packages/genie-agent-sdk`. Each
self-registers with the registry service (`:2005`) and serves A2A at `POST /a2a`. The
platform's discovery bridge surfaces each live agent as a `RemoteAgent` in the
in-process registry, so the pipeline treats local and remote agents identically.
`agent_mode` selects `distributed` (default), `hybrid`, or `local`.

## Services overview

| Terminal | Service | Directory | Port | Purpose |
|----------|---------|-----------|------|---------|
| 1 | MLflow | `/` | **2001** | Experiment tracking + traces |
| 2 | Registry | `services/registry/` | **2005** | Agent discovery (self-register + heartbeat) |
| 3 | MCP tool server | `services/agents/` | **2002** | Weather + outage + docs tools (MCP) |
| 4–6 | Example agents | `services/agents/` | **2010-2012** | weather / outage / rag (A2A services) |
| 7 | Genie Platform | `/` | **2003** | Control plane (FastAPI + LangGraph) |
| (opt) | RAG service | `services/rag_service/` | **2004** | Extracted RAG microservice |

## Quick start

Requires [`uv`](https://docs.astral.sh/uv/). See [SETUP.md](SETUP.md) for details.

```bash
# Platform deps. Guards default ON, so include the guards extra (or set GENIE_ENABLE_GUARDS=false).
uv sync --extra dev --extra guards
# Multi-store memory is optional (default is in-memory): add when you want durability
#   uv sync --extra memory-mongo --extra memory-redis --extra memory-vector

# Bring everything up (Windows PowerShell): opens each service in its own window.
powershell -ExecutionPolicy Bypass -File scripts/run-all.ps1
```

Or start the pieces manually (each in its own terminal, in order):

```bash
uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db
cd services/registry && uv run python -m registry_service.service           # :2005
cd services/agents   && uv run python mcp_weather_server.py                  # :2002
cd services/agents   && AGENT_PORT=2010 uv run python weather_agent.py       # self-registers
cd services/agents   && AGENT_PORT=2011 uv run python outage_agent.py
cd services/agents   && AGENT_PORT=2012 uv run python rag_agent.py
uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload  # :2003
```

Then open **http://localhost:2003/** (chat UI) and **/trace.html** (pipeline trace
visualizer), or POST to `/api/v1/chat`.

## Configuration

Settings load via `pydantic-settings` (`GENIE_` env prefix) layered over
`config/default.yaml`. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `GENIE_AGENT_MODE` | `distributed` | `distributed` / `hybrid` / `local` |
| `GENIE_REGISTRY_URL` | `http://127.0.0.1:2005` | Discovery service base URL |
| `GENIE_ENABLE_GUARDS` | `true` | Mandatory content guards (needs `--extra guards`) |
| `GENIE_MEMORY_BACKEND` | `in_memory` | `in_memory` or `mongo` (durable) |
| `GENIE_ENABLE_HITL` | `false` | HITL approval node (retained, off by default) |
| `GENIE_LLM_PROVIDER` | `gpt_oss` | Named model from `llm_services.models` |
| `GENIE_API_KEY` | `null` | Bearer API key; unset = no auth |

## Running tests

```bash
uv run pytest                  # unit + integration + e2e + contract
uv run lint-imports            # architectural boundary enforcement
uv run ruff format . && uv run ruff check src/genie
```

## Adding a new agent

Build it as a service on `genie-agent-sdk` (see `services/agents/weather_agent.py`),
declare its `AgentMeta`, and run it pointed at `GENIE_REGISTRY_URL`. It self-registers
and the platform discovers it — **no platform code change**.

## Architectural Decision Records

- [0001](docs/adr/0001-hybrid-architecture.md)–[0007](docs/adr/0007-technology-stack.md) — original Genie structure decisions
- [0008](docs/adr/0008-mandatory-content-guards.md) — mandatory content guards
- [0009](docs/adr/0009-multi-store-memory.md) — multi-store memory behind the ports
- [0010](docs/adr/0010-distributed-a2a-agents.md) — distributed A2A agents (primary model)
- [0011](docs/adr/0011-remove-demo-applications.md) — removal of the DLR/webTrader demos

## Architectural boundaries (enforced by import-linter)

```
interface → application → agents → (a2a, discovery, llm, rag ports, tools, mcp, memory ports, security)
genie.*  ✗→ rag_service          (extracted microservice — reached over HTTP)
genie.*  ✗→ registry_service     (extracted microservice — reached over HTTP)
genie.*  ✗→ genie_agent_sdk      (separate deployable — agents run as services)
```
