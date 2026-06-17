# Agent Registry and Discovery — Deep Dive

**Date:** 2026-06-07  
**Audience:** Platform engineers, application developers, architects

This document covers the full lifecycle: how agents declare what they can do, how they are registered at startup, how the pipeline discovers them at request time, what industry standards exist, and where this implementation can be improved.

---

## 1. Industry Standards for Agent Discovery

Before looking at the project, it is worth understanding what the industry uses.

### A2A — Agent-to-Agent Protocol (Google, selected in ADR 0007)

A2A is an open HTTP protocol (not a framework) where every agent publishes a machine-readable **agent card** at a well-known URL:

```
GET /.well-known/agent.json

{
  "name": "Conductor Data Agent",
  "description": "Retrieves electrical conductor specifications",
  "version": "1.0.0",
  "url": "https://dlr-service.oati.net/a2a",
  "capabilities": {
    "streaming": false,
    "pushNotifications": false
  },
  "skills": [
    {
      "id": "conductor_data",
      "name": "Get Conductor Data",
      "description": "Fetches conductor records filtered by type, code, active status",
      "tags": ["conductor", "acsr", "dlr"],
      "inputModes": ["text"],
      "outputModes": ["text", "data"]
    }
  ],
  "authentication": { "schemes": ["Bearer"] }
}
```

Discovery in A2A works by either:
1. **Service registry** — a central directory where agents publish their card URLs
2. **Well-known endpoint** — callers probe known hostnames at `/.well-known/agent.json`
3. **Out-of-band configuration** — card URLs are listed in a config file

### MCP — Model Context Protocol (Anthropic)

MCP exposes tools (functions), resources (data), and prompts. Discovery is via `tools/list`:

```json
POST /mcp
{ "method": "tools/list" }

Response:
{ "tools": [
    { "name": "get_meter_availability",
      "description": "...",
      "inputSchema": { "type": "object", "properties": { "market": {...}, "duration": {...} } }
    }
  ]
}
```

MCP's distinguishing feature is the **JSON Schema for inputs** — callers know exactly what parameters to pass before calling.

### OpenAI Function Calling (de-facto standard)

Tools declared as JSON Schema objects with a name, description, and parameter schema. The LLM selects the right tool and fills its arguments from natural language. Used by LangChain tool-use patterns.

### How this project compares

| Feature | A2A | MCP | This Project |
|---------|-----|-----|--------------|
| Self-describing manifest | `agent-card.json` | `tools/list` | `AgentInfo` (Python object) |
| HTTP-discoverable | Yes (`/.well-known/agent.json`) | Yes (MCP endpoint) | Via REST `GET /api/v1/agents` |
| Capability declaration | `skills[].id` | `tools[].name` | `capabilities: list[str]` |
| Routing keywords | N/A (LLM-driven) | N/A | `routing_keywords: list[str]` |
| Input schema | JSON Schema in skill | JSON Schema in tool | `dict[str, Any]` — no schema |
| Output schema | N/A | N/A | `str` — no schema |
| Cross-language agents | Yes (HTTP) | Yes (HTTP) | No (Python in-process only) |
| Dynamic registration | Yes (HTTP publish) | Yes | No (startup-only) |

The project's current model is closest to a **lightweight A2A agent card** implemented as a Python protocol — self-describing, with capabilities and routing keywords, but in-process only. The A2A HTTP layer is planned as Phase 2 (ADR 0007).

---

## 2. How Registration Works — Current Implementation

### Step 1 — The agent declares its identity

Every agent implements the `BaseAgent` protocol ([src/genie/agents/base.py](../../src/genie/agents/base.py)). There is no base class to inherit — any class with the right properties satisfies the protocol (Python structural typing via `@runtime_checkable`).

```python
# genie/agents/base.py — the contract

class AgentInfo(BaseModel):
    agent_id: str             # unique string — used as dict key in registry
    name: str                 # human display name
    description: str          # what the agent does (shown in /api/v1/agents)
    capabilities: list[str]   # what the planner matches against (e.g. "conductor_data")
    routing_keywords: list[str] = []  # keywords the router uses for heuristic fallback
    version: str
    enabled: bool
    health: str = "healthy"   # currently hardcoded; not checked at runtime

@runtime_checkable
class BaseAgent(Protocol):
    agent_id: str
    name: str
    description: str
    capabilities: list[str]
    version: str
    enabled: bool
    async def execute(task, context) -> AgentResult: ...
    def get_info(self) -> AgentInfo: ...
```

Each application agent defines its identity as class-level constants:

```python
# applications/dlr/conductor/agent.py
class ConductorAgent:
    _agent_id    = "conductor_agent_v1"
    _name        = "Conductor Data Agent"
    _description = "Retrieves technical specifications for electrical conductors..."
    _capabilities = ["conductor_data"]       # ← matched by Planner
    _version     = "1.0.0"

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id=self._agent_id,
            capabilities=self._capabilities,
            routing_keywords=["conductor", "acsr", "aac", "acss"],  # ← matched by Router
            ...
        )
```

### Step 2 — The provider wires the agent into the platform

Each application has a `providers.py` that exports `AGENT_PROVIDERS` — a list of factory callables:

```python
# applications/dlr/providers.py

def _conductor_provider(*, tool_gateway, settings) -> ConductorAgent:
    svc = settings.application_services.get("conductor_service")
    if svc:
        tool_gateway.register("conductor_data", make_conductor_data_handler(svc.url, ...))
    return ConductorAgent(tool_gateway=tool_gateway)

AGENT_PROVIDERS = [_conductor_provider]
```

The provider does two things:
1. Registers the agent's **tools** into the `ToolGateway`
2. Constructs and returns the **agent** itself

### Step 3 — Bootstrap iterates providers and calls `registry.register()`

```python
# genie/interface/bootstrap.py — _build_dependencies()

agent_registry = AgentRegistry()
for provider in agent_providers or []:
    agent = provider(tool_gateway=tool_gateway, settings=settings)
    agent_registry.register(agent)        # ← stored in dict[agent_id, BaseAgent]
    logger.info("agent_registered", agent_id=agent.agent_id, capabilities=list(agent.capabilities))
```

### Step 4 — Registry stores by agent_id

```python
# genie/agents/registry.py

class AgentRegistry:
    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        self._agents[agent.agent_id] = agent   # silent overwrite if ID collides

    def find_by_capability(self, capability: str) -> list[BaseAgent]:
        return [a for a in self._agents.values()
                if capability in a.capabilities and a.enabled]

    def list_all(self) -> list[AgentInfo]:
        return [a.get_info() for a in self._agents.values()]
```

### Full registration sequence diagram

```
app.py create_app()
    │
    ├─ _webtrader_providers = [_meter_data_provider, _rules_engine_provider]
    └─ _dlr_providers       = [_conductor_provider]
                │
                ▼
bootstrap.py _build_dependencies(agent_providers=[...])
                │
                ├─ AgentRegistry created (empty dict)
                │
                ├─ _meter_data_provider(tool_gateway, settings)
                │    ├─ tool_gateway.register("meter_data_availability", handler)
                │    └─ return MeterDataAgent(tool_gateway)
                │         └─ registry.register(agent)  ← _agents["meter_data_agent_v1"]
                │
                ├─ _rules_engine_provider(tool_gateway, settings)
                │    └─ return RulesEngineAgent()
                │         └─ registry.register(agent)  ← _agents["rules_engine_agent_v1"]
                │
                └─ _conductor_provider(tool_gateway, settings)
                     ├─ tool_gateway.register("conductor_data", handler)
                     └─ return ConductorAgent(tool_gateway)
                          └─ registry.register(agent)  ← _agents["conductor_agent_v1"]

Result: registry._agents = {
    "meter_data_agent_v1":   MeterDataAgent,
    "rules_engine_agent_v1": RulesEngineAgent,
    "conductor_agent_v1":    ConductorAgent,
}
```

---

## 3. How Discovery Works — Current Implementation

There are three discovery consumers, each using the registry differently.

### Consumer 1 — RouterNode (keyword-based heuristic building)

The Router calls `registry.list_all()` on every request to dynamically build its heuristic table:

```python
# genie/application/nodes/router.py — _build_heuristics()

def _build_heuristics(self):
    heuristics = list(_PLATFORM_HEURISTICS)   # rag_query, domain_query rules
    for info in self._registry.list_all():    # ← reads AgentInfo from all agents
        if info.routing_keywords and info.capabilities:
            heuristics.append(
                (info.routing_keywords, "agent_task", info.capabilities[0])
                #  ┌──────────────┘                              └────────────────────┐
                #  keywords that match                           capability hint to pass
                #  (e.g. ["conductor", "acsr"])                  to Planner
            )
    return heuristics
```

Result with current agents:
```
heuristics = [
  (["document", "find in", "retrieve", ...], "rag_query",    ""),
  (["weather"],                               "domain_query", "weather"),
  (["meter", "availability", "market", ...],  "agent_task",   "meter_data_availability"),
  (["deal", "validate", "rules", ...],        "agent_task",   "rules_engine"),
  (["conductor", "acsr", "aac", ...],         "agent_task",   "conductor_data"),
]
```

If the LLM classification fails, the router scans these heuristics in order. **First match wins.**

### Consumer 2 — PlannerNode (capability-based lookup)

The Planner receives `routing_hint` from the Router and calls `find_by_capability()`:

```python
# genie/application/nodes/planner.py

routing_hint = state.metadata.get("routing_hint", "")
# e.g. routing_hint = "conductor_data"

agents = self._registry.find_by_capability(routing_hint)
# → find_by_capability("conductor_data")
# → [ConductorAgent]   (only agent with "conductor_data" in capabilities)

selected = [a.agent_id for a in agents]
# → ["conductor_agent_v1"]
```

`find_by_capability()` scans all agents and returns all whose `capabilities` list contains the given string AND whose `enabled=True`.

### Consumer 3 — ExecutorNode (direct ID lookup)

The Executor uses the exact agent IDs selected by the Planner:

```python
# genie/application/nodes/executor.py

for agent_id in state.selected_agents:  # e.g. ["conductor_agent_v1"]
    agent = self._registry.get(agent_id)   # O(1) dict lookup
    if agent is None:
        logger.warning("executor_agent_not_found", ...)
        continue
    result = await agent.execute(task, context)
```

### Consumer 4 — REST API (human and external discovery)

```
GET /api/v1/agents
→ [
    { "agent_id": "meter_data_agent_v1",   "capabilities": ["meter_data_availability"],
      "routing_keywords": ["meter", "availability", ...], "enabled": true, "health": "healthy" },
    { "agent_id": "rules_engine_agent_v1", "capabilities": ["rules_engine"], ... },
    { "agent_id": "conductor_agent_v1",    "capabilities": ["conductor_data"], ... }
  ]

GET /api/v1/agents/conductor_agent_v1
→ { "agent_id": "conductor_agent_v1", "name": "Conductor Data Agent", ... }
```

### Full discovery flow for a single request

```
User: "Give me 3 active ACSR conductors"
                        │
                        ▼
              ┌─────────────────────┐
              │     RouterNode       │
              │                     │
              │ registry.list_all() │ ← reads AgentInfo from all 3 agents
              │   builds heuristics │
              │                     │
              │ LLM classifies:     │
              │  "agent_task"       │
              │                     │
              │ keyword scan:       │
              │  "acsr" matches     │
              │  ConductorAgent     │
              │  hint="conductor_data"│
              └──────────┬──────────┘
                         │ request_type="agent_task"
                         │ metadata.routing_hint="conductor_data"
                         ▼
              ┌─────────────────────┐
              │     PlannerNode      │
              │                     │
              │ find_by_capability( │
              │  "conductor_data")  │ ← scans all agents, finds ConductorAgent
              │                     │
              │ selected_agents=    │
              │  ["conductor_agent_v1"] │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     ExecutorNode     │
              │                     │
              │ registry.get(       │ ← O(1) dict lookup by agent_id
              │  "conductor_agent_v1")
              │                     │
              │ agent.execute(task) │
              └─────────────────────┘
```

---

## 4. What Each Agent Publishes (Current Agent Cards)

### MeterDataAgent

| Field | Value |
|-------|-------|
| `agent_id` | `meter_data_agent_v1` |
| `capabilities` | `["meter_data_availability"]` |
| `routing_keywords` | `["meter", "availability", "market", "duration", "ercot"]` |
| `description` | Retrieves meter data availability by duration and market |
| `version` | `1.0.0` |
| Input schema | None — parses `task.instruction` free-text |
| Tool used | `meter_data_availability` (ToolGateway) → MCP or JSON fixture |

### RulesEngineAgent

| Field | Value |
|-------|-------|
| `agent_id` | `rules_engine_agent_v1` |
| `capabilities` | `["rules_engine"]` |
| `routing_keywords` | `["deal", "validate", "rules", "san jose"]` |
| `description` | Validates deals against business rules with weather-based inputs |
| `version` | `1.0.0` |
| Input schema | None — reads `task.context["start_date"]`, `task.context["end_date"]` |
| Tool used | None — reads JSON fixture files directly |

### ConductorAgent

| Field | Value |
|-------|-------|
| `agent_id` | `conductor_agent_v1` |
| `capabilities` | `["conductor_data"]` |
| `routing_keywords` | `["conductor", "acsr", "aac", "acss", "aaac"]` |
| `description` | Retrieves technical specifications for electrical conductors |
| `version` | `1.0.0` |
| Input schema | None — parses `task.instruction` free-text |
| Tool used | `conductor_data` (ToolGateway) → REST `GET /api/v1/conductors/` |

---

## 5. Gaps and Improvement Opportunities

### Gap 1 — `health` field is hardcoded, never checked

**Current:** Every agent returns `health: "healthy"` in `AgentInfo` regardless of its actual state.

```python
def get_info(self) -> AgentInfo:
    return AgentInfo(..., health="healthy")   # always "healthy"
```

**Problem:** If the ConductorAgent's REST backend is down, the registry reports it as healthy, the planner selects it, and the executor gets an error at runtime — after a full RAG retrieval and orchestration pass.

**Fix:** Add a health-check method to the protocol and have the registry call it on a background timer:

```python
# Addition to BaseAgent protocol
async def health_check(self) -> str:
    """Return 'healthy', 'degraded', or 'unhealthy'."""
    ...

# In AgentRegistry — background health loop
async def _health_loop(self):
    while True:
        for agent in self._agents.values():
            if hasattr(agent, "health_check"):
                status = await agent.health_check()
                self._health[agent.agent_id] = status
        await asyncio.sleep(30)

# In find_by_capability — filter out unhealthy agents
def find_by_capability(self, capability: str) -> list[BaseAgent]:
    return [
        a for a in self._agents.values()
        if capability in a.capabilities
        and a.enabled
        and self._health.get(a.agent_id, "healthy") != "unhealthy"
    ]
```

---

### Gap 2 — Capability string collision is silent

**Current:** If two agents declare the same capability string, `find_by_capability()` returns both silently. The planner selects all of them, the executor runs both.

```python
# If both MeterDataAgent and a new agent declare "meter_data_availability":
registry.find_by_capability("meter_data_availability")
# → [MeterDataAgent, NewAgent]   — both run, results merged
```

**Problem:** Unintentional capability collision causes unexpected agent pairing with no warning.

**Fix:** Detect and warn (or reject) at registration time:

```python
def register(self, agent: BaseAgent) -> None:
    for cap in agent.capabilities:
        existing = self.find_by_capability(cap)
        if existing:
            ids = [a.agent_id for a in existing]
            self._logger.warning(
                "agent_capability_collision",
                capability=cap,
                new_agent=agent.agent_id,
                existing_agents=ids,
            )
    self._agents[agent.agent_id] = agent
```

---

### Gap 3 — No input/output schema — agents are black boxes to the planner

**Current:** `AgentTask.context` is `dict[str, Any]`. The planner has no way to know that `RulesEngineAgent` reads `context["start_date"]` and `context["end_date"]` from the task context, or that `MeterDataAgent` expects `"market"` and `"duration"` to be parseable from the free-text instruction.

**Problem:** When a request is ambiguous, the planner cannot validate whether a selected agent has everything it needs before dispatching it.

**Fix:** Add an `input_schema` field to `AgentInfo`:

```python
class AgentInfo(BaseModel):
    ...
    input_schema: dict[str, Any] = {}   # JSON Schema for task.context
    output_description: str = ""        # what the output field contains

# In RulesEngineAgent.get_info():
input_schema = {
    "type": "object",
    "properties": {
        "start_date": {"type": "string", "format": "date", "description": "Period start (ISO 8601)"},
        "end_date":   {"type": "string", "format": "date", "description": "Period end (ISO 8601)"},
    },
    "required": ["start_date", "end_date"]
}
```

This also makes the REST endpoint (`GET /api/v1/agents`) a proper machine-readable catalog — identical to the MCP `tools/list` response — and is the first step toward A2A compatibility.

---

### Gap 4 — `routing_keywords` and `capabilities` are two separate lists with no enforced relationship

**Current:** An agent declares:
- `_capabilities = ["conductor_data"]` — used by the Planner
- `routing_keywords=["conductor", "acsr", ...]` — used by the Router, declared separately in `get_info()`

**Problem:** Nothing enforces that the routing keywords actually map to the capability. A developer could declare `_capabilities = ["conductor_data"]` but accidentally write `routing_keywords=["meter"]`, causing the Router to route to ConductorAgent when "meter" is mentioned.

**Fix:** Unify into a single capability descriptor:

```python
class CapabilitySpec(BaseModel):
    id: str                        # "conductor_data" — used by Planner
    display_name: str              # "Conductor Data"
    description: str               # LLM-readable
    routing_keywords: list[str]    # tied to this specific capability
    input_schema: dict = {}

class AgentInfo(BaseModel):
    agent_id: str
    name: str
    version: str
    enabled: bool
    health: str = "healthy"
    capability_specs: list[CapabilitySpec]   # replaces capabilities + routing_keywords
```

The router then builds its heuristics from `spec.routing_keywords` and the planner matches by `spec.id` — both derived from the same `CapabilitySpec` object, impossible to desync.

---

### Gap 5 — Agent ID collision silently overwrites the previous registration

**Current:** `registry.register()` does `self._agents[agent.agent_id] = agent`. If two agents share the same `agent_id`, the second registration silently replaces the first.

**Fix:**
```python
def register(self, agent: BaseAgent) -> None:
    if agent.agent_id in self._agents:
        raise ValueError(
            f"Agent ID '{agent.agent_id}' is already registered by "
            f"{type(self._agents[agent.agent_id]).__name__}. "
            "Use a unique agent_id."
        )
    self._agents[agent.agent_id] = agent
```

---

### Gap 6 — No `unregister()` method

**Current:** Once registered, an agent cannot be removed without a server restart.

**Fix:**
```python
def unregister(self, agent_id: str) -> None:
    if agent_id not in self._agents:
        raise GenieError(ErrorCode.NOT_FOUND, f"Agent '{agent_id}' not found")
    del self._agents[agent_id]
    self._logger.info("agent_unregistered", agent_id=agent_id)
```

This enables graceful shutdown of individual agents and future dynamic registration.

---

### Gap 7 — Planner has no LLM fallback for ambiguous requests

**Current:** The planner does pure capability-string matching. If `routing_hint` doesn't exactly match any capability, it falls back to `AgentCapability.GENERAL` — which only matches agents that explicitly declare `"general"` as a capability (none currently do).

```python
# planner.py — current fallback
agents = self._registry.find_by_capability(routing_hint)
if not agents and capability != AgentCapability.GENERAL:
    agents = self._registry.find_by_capability(AgentCapability.GENERAL)
# If still empty → selected_agents=[]  → executor runs no agents
```

**Problem:** A query like "show me conductors that pass deal validation rules" involves both ConductorAgent and RulesEngineAgent, but the router only sets one routing hint.

**Fix — LLM-assisted multi-agent selection in PlannerNode:**

```python
# When routing_hint matches nothing, ask the LLM to select from the catalog
if not agents:
    catalog = [info.model_dump() for info in self._registry.list_all()]
    prompt = f"""
    Given this user request: "{instruction}"
    And these available agents: {json.dumps(catalog)}
    Which agent IDs should handle this request? Return a JSON list of agent_ids.
    """
    response = await self._llm.complete([Message(role="user", content=prompt)], max_tokens=100)
    selected_ids = json.loads(response.content)
    agents = [a for aid in selected_ids if (a := self._registry.get(aid))]
```

This is the same mechanism LangChain tool-use uses — the LLM reads the `description` and `input_schema` to pick the right tool.

---

### Gap 8 — `enable()`/`disable()` bypasses agent encapsulation

**Current:**
```python
def enable(self, agent_id: str) -> None:
    agent = self.require(agent_id)
    if hasattr(agent, "_enabled"):
        agent._enabled = True     # direct private attribute mutation
```

**Problem:** This only works if the agent stores `_enabled` as a plain attribute. It bypasses any property setter logic an agent might have (e.g., to pause background tasks when disabled).

**Fix:** Add `enable()` and `disable()` to the `BaseAgent` protocol:

```python
@runtime_checkable
class BaseAgent(Protocol):
    ...
    def enable(self) -> None: ...
    def disable(self) -> None: ...
```

Then `AgentRegistry.enable()` calls `agent.enable()` instead of mutating the private field.

---

### Gap 9 — No dynamic registration (future readiness)

**Current:** All agents register at startup. No mechanism to add agents while the server is running.

**Phase 2 requirement:** A2A agents deployed as separate services need to be discoverable and registerable at runtime — not just at startup.

**Minimal design for dynamic registration:**

```python
# New endpoint
POST /api/v1/agents/register
{
  "agent_card_url": "https://dlr-service.oati.net/.well-known/agent.json"
}

# Bootstrap fetches the agent card, creates a RemoteAgent proxy, registers it
class RemoteAgent:
    """Proxy that calls a remote A2A agent endpoint."""
    def __init__(self, card: A2AAgentCard):
        self._card = card
        self._agent_id = card.name
        self._capabilities = [skill.id for skill in card.skills]
        self._routing_keywords = [tag for skill in card.skills for tag in skill.tags]

    async def execute(self, task: AgentTask, context: dict) -> AgentResult:
        # HTTP POST to card.url using A2A task schema
        ...
```

This is the bridge between the current in-process model and the Phase 2 A2A architecture.

---

## 6. Summary: Current State vs Recommended State

| Aspect | Current | Recommended |
|--------|---------|-------------|
| Agent contract | `BaseAgent` Python Protocol | Same + `health_check()`, `enable()`, `disable()`, `input_schema` |
| Capability declaration | Two separate fields (`capabilities` + `routing_keywords`) | Unified `CapabilitySpec` per capability |
| Registration validation | None — silent overwrite on ID collision | Reject duplicate IDs; warn on capability collision |
| Health tracking | Hardcoded `"healthy"` | Background health-check loop, cache result |
| Discovery by capability | Exact string match | Same, plus filter by health status |
| Planner multi-agent selection | Single routing hint → one capability match | LLM fallback for ambiguous / multi-agent requests |
| Agent de-registration | Not supported | `unregister()` method |
| Input schema | None | JSON Schema on `CapabilitySpec` |
| Dynamic registration | No (startup only) | `POST /api/v1/agents/register` → `RemoteAgent` proxy |
| A2A readiness | None | `RemoteAgent` adapter + agent card fetch at registration |

### Recommended implementation order

1. **Now (low effort, high safety):** ID collision detection in `register()` — one-line fix
2. **Now:** Warn on capability collision — three lines in `register()`
3. **This sprint:** Unify `CapabilitySpec` — eliminates routing/capability desync risk
4. **This sprint:** `input_schema` on `AgentInfo` — unlocks LLM-assisted planning and makes REST catalog machine-readable
5. **Next sprint:** Background health-check loop — prevents dispatching to dead agents
6. **Phase 2:** `RemoteAgent` proxy + `POST /api/v1/agents/register` — A2A dynamic registration
