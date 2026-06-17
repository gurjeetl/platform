# ADR 0001 — Hybrid Modular-Monolith + Microservices Architecture

**Status:** Accepted  
**Date:** 2026-06-06  
**Updated:** 2026-06-07

---

## Context

OATI's Genie platform must serve two goals simultaneously:

1. **Rapid feature development** — application teams need to add new AI agents without requiring platform releases or deep knowledge of infrastructure internals.
2. **Production-grade stability** — the platform team must be able to upgrade LLM providers, pipeline nodes, and infrastructure independently.

Two classical architectures both fall short:

- A **pure monolith** couples all agents to the same release cycle. Changing the LLM provider or retrieval backend risks breaking every agent. Application teams must touch platform code to add features.
- **Pure microservices from day one** adds operational overhead (service discovery, inter-service auth, network latency, separate CI pipelines) before the domain model is stable enough to justify it. Teams spend more time on deployment plumbing than on product features.

The platform also needs a clear answer to a future team-scaling question: *"When do we split application and platform code into separate repositories, and how do we do it without a painful refactor?"*

---

## Decision

Adopt a **three-layer hybrid architecture** that begins as a well-structured single deployment and supports gradual extraction of services as the product matures.

---

### Layer 1 — Modular Monolith: The Platform

The `genie` package is a **modular monolith**. All pipeline components share one Python process and communicate via in-process function calls and the in-process `EventBus`. Modules are isolated by package boundary, not by network boundary.

```
src/genie/
├── agents/        ← BaseAgent protocol, AgentRegistry, AgentCapability
├── application/   ← LangGraph 7-node graph definition
├── interface/     ← FastAPI routes, bootstrap factory
├── llm/           ← LLM provider abstraction (OpenAI-compatible + mock)
├── rag/           ← Port+adapter for retrieval (local keyword or remote Milvus)
├── tools/         ← Tool gateway and base contracts
├── tracking/      ← MLflow experiment and trace wrapper
├── security/      ← API key middleware, prompt injection detection
├── memory/        ← Session store and long-term store
├── mcp/           ← MCP client and tool adapter
├── observability/ ← Structured logging, correlation middleware
└── platform/      ← Settings, error codes, event bus
```

**Why a single process for the control plane?**

| Reason | Detail |
|--------|--------|
| Zero network overhead | Agent dispatch, state propagation, and LLM calls are in-process function calls |
| Single shared state | `GraphState` (Pydantic model) carries all pipeline state across nodes — no serialisation between steps |
| HITL without extra infrastructure | `MemorySaver` checkpointer persists state per `conversation_id`; pause/resume works in-process |
| Simple CI | One `pytest` run, no Docker Compose, no service mesh |
| Fast iteration | Hot-reload with `--reload` restarts in under 2 seconds |

---

### Layer 2 — Application Layer: Agents and Domain Logic

Application-specific code lives **outside** the `genie` package, organised by OATI product:

```
src/
├── app.py                              ← assembles all application providers
└── applications/
    ├── webTrader/
    │   ├── providers.py                ← registers webTrader agents; team edits ONLY this file
    │   ├── meter_data/                 ← Meter Data Availability Agent
    │   └── rules_engine/              ← Rules Engine Agent (GoRules)
    ├── dlr/                            ← Dynamic Line Rating application
    │   ├── providers.py                ← registers DLR agents
    │   └── conductor/                  ← Conductor Data Agent + REST tool
    └── shared/
        └── domain/                     ← Pydantic models shared across applications
            ├── deals/
            ├── meter_data/
            └── weather_data/
```

Each application folder has a `providers.py` that exports `AGENT_PROVIDERS` — a list of provider callables. The platform (`genie.interface.bootstrap`) imports **nothing** from `applications.*`.

```python
# applications/dlr/providers.py — the DLR team owns this file

def _conductor_provider(*, tool_gateway, settings):
    tool_gateway.register("conductor_data", make_conductor_data_handler(...))
    return ConductorAgent(tool_gateway=tool_gateway)

AGENT_PROVIDERS = [_conductor_provider]
```

```python
# src/app.py — assembles all applications; one import per application

from applications.webTrader.providers import AGENT_PROVIDERS as _webtrader_providers
from applications.dlr.providers import AGENT_PROVIDERS as _dlr_providers

def create_app(settings=None):
    return _create_platform_app(
        settings=settings,
        agent_providers=[*_webtrader_providers, *_dlr_providers],
        startup_hooks=[_seed_rag],
    )
```

**To add a new agent to an existing application** (e.g. a new DLR agent):
1. Create `src/applications/dlr/<agent>/agent.py` implementing `BaseAgent`
2. Add a provider function to `src/applications/dlr/providers.py`
3. Add it to `AGENT_PROVIDERS` in that file
4. **Zero changes to webTrader, platform, or any other application**

**To add an entirely new application** (e.g. `gridAnalytics`):
1. Create `src/applications/gridAnalytics/providers.py` with `AGENT_PROVIDERS`
2. Add one import line to `src/app.py`

**To change the LLM provider or pipeline node**, the platform team:
1. Edits `src/genie/llm/` or `src/genie/application/nodes/`
2. Makes **zero application agent changes** (agents talk to `BaseAgent` protocol only)

---

### Layer 3 — Extracted Microservices

The RAG retrieval service is the first extracted microservice. It runs as a standalone FastAPI service with its own vector database (Milvus). The platform accesses it through the `RemoteRAGAdapter` (HTTP). In local development, `LocalRAGAdapter` replaces it in-process.

```
services/
└── rag_service/   ← Standalone service; Milvus backend; deployed independently
```

Future extraction candidates follow the same pattern:
1. Define a protocol (interface) in a shared contracts package
2. Provide a `LocalXxxAdapter` for development
3. Provide a `RemoteXxxAdapter` for production
4. Deploy the service independently when load justifies it

---

## Boundary Enforcement

Architectural boundaries are enforced at commit time by **import-linter** (see [ADR 0004](0004-import-linter.md)). The CI pipeline runs `lint-imports` on every commit.

| Contract | What it enforces |
|----------|-----------------|
| `no-direct-agent-impl-in-nodes` | `genie.application.nodes.*` may not import from `agents.*` |
| `routers-must-not-bypass-application` | `genie.interface.routers.*` may not import from `agents.*` |
| `no-agent-domain-outward` | `agents.domain` may not import from interface or application layers |

This means architectural boundaries cannot be violated accidentally — a developer who tries to shortcut through the injection pattern gets a CI failure, not a runtime bug.

---

## Why the Current Solution Structure Is Production-Ready

### Separation of concerns is already complete

The platform/application boundary is **already enforced by code**, not just convention. The `agent_providers` pattern, import-linter contracts, and startup hooks are all in place. A new team member can add an agent by following the pattern in `src/app.py` without reading any platform code.

### All infrastructure concerns are abstracted

| Concern | Abstraction | Implementation detail hidden |
|---------|-------------|------------------------------|
| LLM provider | `BaseLLMProvider` protocol | OpenAI-compatible URL, vLLM, mock |
| Retrieval | `RetrievalService` protocol | Keyword index, Milvus, Elasticsearch |
| Tool execution | `ToolGateway` | REST call, MCP, in-process mock |
| Experiment tracking | `MLflowTracker` | SQLite, PostgreSQL, remote MLflow |
| Agent dispatch | `AgentRegistry` | Injected at startup |

If any vendor is replaced, only the adapter changes — no agent code, no pipeline node code.

### Observable from day one

MLflow tracking records every chat request (params, metrics, LangChain traces). Structured JSON logging with correlation IDs allows traces to be followed from request to response across all pipeline nodes.

---

## Future Repo Separation Path

When team size or release velocity justifies splitting the repository, the migration is a **packaging operation, not a refactor**:

| Step | Action | Effort |
|------|--------|--------|
| 1 | Create `genie-platform` PyPI package from `src/genie/` | 1 day |
| 2 | Create `genie-webtrader` repo; copy `src/applications/webTrader/` and `src/app.py` | 1 day |
| 3 | Create `genie-dlr` repo; copy `src/applications/dlr/` | 1 day |
| 4 | Replace local import with `pip install genie-platform` in each application repo | 1 hour per repo |
| 5 | Each application repo has its own CI; platform releases are versioned; applications pin a version | Ongoing |
| 6 | `genie-rag-contracts` becomes a separate pip package all repos depend on | 1 day |

Because each application is already isolated in its own folder with no cross-application imports (enforced by import-linter), extraction requires no refactoring — only packaging.

**No refactoring is needed** at extraction time because the boundary is already enforced by import-linter and the injection pattern. The only change is replacing a local Python path with a versioned package dependency.

---

## Consequences

**Positive**
- Application teams can ship new agents without platform releases.
- Platform team can upgrade infrastructure without coordinating with application teams.
- Zero external dependencies required in CI or local development.
- Import-linter prevents boundary violations at commit time.
- Clear, mechanical path to multi-repo when team scale requires it.

**Negative**
- `MemorySaver` is in-process; multi-replica deployments require an external checkpointer (see [ADR 0002](0002-langgraph-pipeline.md)).
- `LocalRAGAdapter` uses keyword matching only; production performance requires `RemoteRAGAdapter` with Milvus.
- Application and platform must agree on the `BaseAgent` protocol version; breaking changes require a coordinated release.

---

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Full microservices from day one | Too much operational overhead before domain model is stable; slows feature velocity |
| Single monolith (agents inside `genie`) | Cannot prevent platform coupling; every new agent requires a platform commit; import boundaries cannot be enforced |
| Separate repos from the start | Import-linter cannot enforce cross-repo boundaries automatically; extraction is harder if code is entangled before the pattern is established |
| Plugin system with dynamic loading | Adds complexity (plugin manifests, discovery, versioning) with no benefit over Python injection |
