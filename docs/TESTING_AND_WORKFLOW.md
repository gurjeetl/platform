# Genie Platform — Test Suite, Contracts, and Chat Workflow Guide

---

## 1. Running the Tests

```bash
# All 54 tests
python3.11 -m pytest tests/ -v

# By layer
python3.11 -m pytest tests/unit/        # 37 unit tests
python3.11 -m pytest tests/integration/ # 7 integration tests
python3.11 -m pytest tests/e2e/         # 8 end-to-end tests
python3.11 -m pytest tests/contract/    # 2 contract tests

# Import-linter boundary check
lint-imports
```

---

## 2. All 54 Tests

### 2.1 Contract Tests (2 tests)
> File: `tests/contract/test_import_contracts.py`
> 
> Enforce architectural boundaries by running `lint-imports` and static AST inspection.

| # | Test | What it verifies |
|---|------|-----------------|
| 1 | `test_import_linter_contracts_pass` | Runs `lint-imports` as a subprocess; fails the test if any of the 4 configured contracts are broken |
| 2 | `test_domain_does_not_import_interface` | AST-walks every `.py` file under `src/genie/domain/` and asserts zero imports of `genie.interface.*` |

---

### 2.2 End-to-End Tests (8 tests)
> File: `tests/e2e/test_api.py`
> 
> Start the full FastAPI application via `TestClient` (runs real lifespan: registers agents, builds graph, wires dependencies) and test HTTP behaviour.

| # | Test | What it verifies |
|---|------|-----------------|
| 3 | `test_health_endpoint` | `GET /health` returns `{"status": "ok", "version": "0.1.0"}` with HTTP 200 |
| 4 | `test_ready_endpoint` | `GET /ready` returns HTTP 200 |
| 5 | `test_list_agents_endpoint` | `GET /api/v1/agents` returns a list containing both `meter_data_agent_v1` and `rules_engine_agent_v1` |
| 6 | `test_get_agent_endpoint` | `GET /api/v1/agents/meter_data_agent_v1` returns agent details with the correct `agent_id` |
| 7 | `test_chat_general_message` | `POST /api/v1/chat` with "Hello there" returns a non-empty `response` and a valid `conversation_id` |
| 8 | `test_chat_meter_query` | `POST /api/v1/chat` with an ERCOT meter query returns a non-empty `response` |
| 9 | `test_chat_returns_correlation_id` | Every chat response includes a non-empty `correlation_id` for request tracing |
| 10 | `test_prompt_injection_rejected` | Message containing "ignore all previous instructions" is rejected with HTTP 400 |

---

### 2.3 Integration Tests — Pipeline (3 tests)
> File: `tests/integration/test_pipeline.py`
> 
> Build the real LangGraph graph and invoke it end-to-end inside the test process.

| # | Test | What it verifies |
|---|------|-----------------|
| 11 | `test_pipeline_general_chat_end_to_end` | Graph invoked with "Hello, how are you?" produces a non-empty `final_response` |
| 12 | `test_pipeline_meter_query` | Graph invoked with an ERCOT meter query produces a response **and** sets `request_type = "agent_task"` |
| 13 | `test_pipeline_with_rag` | Pre-ingests a document into `LocalRAGAdapter`, runs a retrieval query through the full graph, verifies a non-empty response |

---

### 2.4 Integration Tests — RAG Adapter (4 tests)
> File: `tests/integration/test_rag_adapter.py`
> 
> Test `LocalRAGAdapter` ingestion and retrieval end-to-end.

| # | Test | What it verifies |
|---|------|-----------------|
| 14 | `test_local_rag_ingest_and_retrieve` | Ingest an ERCOT document; query "ERCOT power grid"; top result contains "ERCOT" with `score > 0` |
| 15 | `test_local_rag_retrieve_empty_returns_empty` | Query against an empty index returns `results = []` with `retrieval_available = True` |
| 16 | `test_local_rag_retrieves_top_k` | Ingest 10 documents; query with `top_k=3`; result set never exceeds 3 items |
| 17 | `test_local_rag_ingest_splits_chunks` | Ingest a 500-word document; verify the internal index holds more than 1 chunk (chunking actually ran) |

---

### 2.5 Unit Tests — Agent Implementations (4 tests)
> File: `tests/unit/test_agents.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 18 | `test_meter_data_agent_returns_result` | `MeterDataAgent.execute()` with an ERCOT query returns `success=True`, response contains "ERCOT", `execution_time_ms ≥ 0` |
| 19 | `test_meter_data_agent_unknown_market_uses_ercot` | Instruction with no recognised market keyword defaults to ERCOT and still succeeds |
| 20 | `test_rules_engine_agent_validates_deals` | `RulesEngineAgent.execute()` for San Jose returns `success=True` with "San Jose" in the output |
| 21 | `test_rules_engine_agent_info` | `get_info()` returns `AgentInfo` with `agent_id="rules_engine_agent_v1"` and `enabled=True` |

---

### 2.6 Unit Tests — CompletionGateNode (4 tests)
> File: `tests/unit/test_completion_gate.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 22 | `test_gate_passes_with_results` | State with a successful `ToolResultRecord` → `gate_passed = True` |
| 23 | `test_gate_retries_on_error` | State with `error` set → `gate_passed = False`, `retry_count` incremented to 1 |
| 24 | `test_gate_stops_retrying_at_max` | `retry_count = MAX_RETRIES (2)` with error → gate passes anyway to prevent infinite retry |
| 25 | `test_gate_sets_soft_error_with_no_results` | Empty state (no results, no error) → gate passes but injects a soft error message |

---

### 2.7 Unit Tests — EventBus (5 tests)
> File: `tests/unit/test_event_bus.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 26 | `test_publish_calls_subscriber` | Registered handler is called with the correct `Event.payload` after `publish()` |
| 27 | `test_publish_no_subscribers_is_silent` | Publishing to a topic with no subscribers completes without error |
| 28 | `test_unsubscribe_removes_handler` | Handler removed via `unsubscribe()` is never called on subsequent `publish()` |
| 29 | `test_failing_handler_does_not_block_others` | A handler that raises `RuntimeError` does not prevent other handlers on the same topic from executing |
| 30 | `test_start_stop_no_op` | `await bus.start()` / `await bus.stop()` complete without error (lifecycle compatibility) |

---

### 2.8 Unit Tests — LLM Provider and Registry (5 tests)
> File: `tests/unit/test_llm.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 31 | `test_mock_llm_returns_response` | `MockLLMProvider.complete()` returns `LLMResponse` with non-empty content, correct model name, and `prompt_tokens > 0` |
| 32 | `test_mock_llm_stream` | `MockLLMProvider.stream()` yields tokens that concatenate to a non-empty string |
| 33 | `test_mock_llm_classifies_meter_query` | Classification prompt with ERCOT meter keywords returns exactly `"agent_task"` (used by RouterNode) |
| 34 | `test_llm_registry_get_registered` | `LLMRegistry.get("mock")` returns the previously registered `MockLLMProvider` instance |
| 35 | `test_llm_registry_raises_for_missing` | `LLMRegistry.get("unknown")` raises `GenieError` (NOT_FOUND) |

---

### 2.9 Unit Tests — Memory Stores (7 tests)
> File: `tests/unit/test_memory.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 36 | `test_session_store_set_get` | Set a key in a conversation scope; get it back with the same value |
| 37 | `test_session_store_get_missing_returns_none` | Get a key that was never set returns `None` (no KeyError) |
| 38 | `test_session_store_get_all` | Two keys set in same conversation → `get_all()` returns both |
| 39 | `test_session_store_clear` | After `clear(conv_id)`, previously set key returns `None` |
| 40 | `test_long_term_store_save_get` | Save a dict; retrieve it by `(user_id, key)` → exact same value returned |
| 41 | `test_long_term_store_search` | Keyword search across stored entries returns results where the key or value matches |
| 42 | `test_long_term_store_delete` | Delete an existing key returns `True`; subsequent `get()` returns `None` |

---

### 2.10 Unit Tests — PlannerNode (4 tests)
> File: `tests/unit/test_planner_node.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 43 | `test_planner_general_chat_selects_no_agents` | `request_type = "general_chat"` → `selected_agents = []` (no agent dispatch needed) |
| 44 | `test_planner_meter_hint_selects_meter_agent` | `routing_hint = "meter_data"` → `selected_agents` contains `"meter_data_agent_v1"` |
| 45 | `test_planner_rules_hint_selects_rules_agent` | `routing_hint = "rules_engine"` → `selected_agents` contains `"rules_engine_agent_v1"` |
| 46 | `test_planner_metadata_updated` | Metadata dict is updated with `planned_capability` and `agent_count` after planning |

---

### 2.11 Unit Tests — RouterNode (4 tests)
> File: `tests/unit/test_router_node.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 47 | `test_router_empty_messages_defaults_general_chat` | No messages in state → `request_type = "general_chat"` |
| 48 | `test_router_classifies_meter_query` | "Check ERCOT meter availability for last week" → `request_type = "agent_task"` |
| 49 | `test_router_classifies_rag_query` | "Find documents about energy trading" → `request_type` is `rag_query` or `general_chat` |
| 50 | `test_router_classifies_deal_validation` | "Validate deals for San Jose" → `request_type = "agent_task"` |

---

### 2.12 Unit Tests — GraphState (4 tests)
> File: `tests/unit/test_state.py`

| # | Test | What it verifies |
|---|------|-----------------|
| 51 | `test_graph_state_defaults` | `GraphState()` has `user_id="anonymous"`, empty lists, `requires_approval=False`, `rag_unavailable=False` |
| 52 | `test_graph_state_with_messages` | State constructed with 2 messages has correct `role` values and list length |
| 53 | `test_graph_state_serialization` | `model_dump()` round-trips losslessly through `GraphState(**data)` |
| 54 | `test_message_roles` | All 4 valid roles (`system`, `user`, `assistant`, `tool`) construct without validation error |

---

## 3. Import-Linter Contracts

Configured in `.importlinter`. Run with `lint-imports`.

### Contract 1 — Domain layer must not import platform modules

```
source_modules  : genie.domain
forbidden_modules: genie.interface, genie.application, genie.llm,
                   genie.rag, genie.tools, genie.security, genie.tracking
```

**Why:** The domain layer contains pure business logic and Pydantic models. It must have zero infrastructure dependencies so it can be tested and reasoned about in isolation. Any violation means business rules have leaked a dependency on how the platform is deployed.

**Example violation caught:** `genie.domain.deals.service` importing `from genie.llm import MockLLMProvider` would be flagged.

---

### Contract 2 — Application nodes must not directly import concrete agent implementations

```
source_modules  : genie.application.nodes
forbidden_modules: genie.agents.meter_data_agent, genie.agents.rules_engine_agent
```

**Why:** The pipeline nodes (`PlannerNode`, `ExecutorNode`) interact with agents through the `AgentRegistry` and `BaseAgent` protocol — they must not know which concrete agents exist. This keeps new agents addable without touching any node code.

**Fix applied:** `genie.agents.__init__` no longer re-exports `MeterDataAgent` / `RulesEngineAgent`. They are imported only by the composition root (`interface/bootstrap.py`).

---

### Contract 3 — API routers must not bypass the application layer

```
source_modules  : genie.interface.routers
forbidden_modules: genie.domain, genie.llm, genie.rag.adapters,
                   genie.agents.meter_data_agent, genie.agents.rules_engine_agent
```

**Why:** HTTP handlers should invoke the graph (`create_app` wires it onto `app.state.graph`) and return results. They must not directly call domain services or construct agents — that would duplicate wiring logic and bypass the pipeline (HITL, tracking, retry, etc.).

**Note:** `genie.interface.bootstrap` (the composition root) is intentionally excluded from this contract — it is the only place allowed to import from every layer.

---

### Contract 4 — Platform must not import RAG service internals

```
source_modules  : genie
forbidden_modules: rag_service
```

**Why:** `services/rag_service/` is an independently deployable microservice. The control plane communicates with it only over HTTP via `RemoteRAGAdapter`. A direct Python import would couple the two deployable units, defeating the entire purpose of extracting the service.

---

## 4. Triggering Chat — Meter Data and Rules Engine

### 4.1 Start the Platform

```bash
cd /home/pankajg/genie-platform

# Start in local mode (zero external services needed)
GENIE_RAG_MODE=local \
GENIE_ENABLE_HITL=false \
GENIE_ENABLE_TRACKING=false \
python3.11 -m uvicorn "genie.interface.bootstrap:create_app" \
  --factory --host 0.0.0.0 --port 8000 --reload
```

The startup log confirms both agents registered:

```json
{"event": "agent_registered", "agent_id": "meter_data_agent_v1", ...}
{"event": "agent_registered", "agent_id": "rules_engine_agent_v1", ...}
{"event": "genie_platform_started", "mode": "local", ...}
```

---

### 4.2 Meter Data Availability Query

**curl command:**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Check ERCOT meter availability for last week",
    "user_id": "operator-1"
  }' | python3.11 -m json.tool
```

**Sample response:**

```json
{
  "conversation_id": "8b0903fd-dad0-4291-9a68-4c784843c56d",
  "response": "Meter data availability analysis: ERCOT market shows 98.5% availability for the requested period. All metering systems operational.",
  "request_type": "agent_task",
  "agents_used": ["meter_data_agent_v1"],
  "rag_used": false,
  "correlation_id": "7fc6c352-d460-4979-b9be-4db252abafb6"
}
```

**Other meter query examples:**

```bash
# PJM market
"Check PJM meter availability for last month"

# CAISO last day
"What is the CAISO meter availability for last day?"

# MISO last hour
"Show MISO meter data for last hour"
```

**Workflow trace — step by step:**

```
POST /api/v1/chat
        │
        ▼
[CorrelationMiddleware]         Generates X-Correlation-ID and sets it in contextvars
        │
        ▼
[chat router]                   Calls sanitize_user_input() → checks for prompt injection
        │
        ▼  GraphState initial
           messages = [Message(role="user", content="Check ERCOT meter...")]
        │
        ▼
[RouterNode]                    LLM classifies → "agent_task"
                                Heuristic detects "ercot", "meter" → routing_hint = "meter_data"
        │
        ▼
[PlannerNode]                   Looks up capability METER_DATA_AVAILABILITY in AgentRegistry
                                → selected_agents = ["meter_data_agent_v1"]
        │
        ▼
[OrchestratorNode]              HITL disabled → requires_approval = False
        │
        ▼ (no HITL pause)
[ExecutorNode]                  Calls MeterDataAgent.execute(task)
                                  └─ Parses "ERCOT" from instruction
                                  └─ Parses "last_week" from instruction
                                  └─ Creates ToolCall(tool_id="meter_data_availability")
                                  └─ ConcreteToolGateway.execute(call)
                                       └─ meter_data_availability_handler(params)
                                            └─ returns {total_meters:12450, online:12227, pct:98.2}
                                  └─ Formats output string
                                  └─ Publishes TOPIC_AGENT_EXECUTED to EventBus
                                  └─ Returns AgentResult(success=True, output="Meter data...")
        │
        ▼
[CompletionGateNode]            tool_results non-empty → gate_passed = True
        │
        ▼
[SynthesizerNode]               Builds system prompt with agent result text
                                Calls LLM.complete(messages)  [MockLLM in local mode]
                                Returns final_response
        │
        ▼
[chat router]                   Returns ChatResponse with response, agents_used, correlation_id
```

---

### 4.3 Rules Engine (Deal Validation) Query

**curl command:**

```bash
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Validate deals for San Jose from May 1 to May 15",
    "user_id": "analyst-1"
  }' | python3.11 -m json.tool
```

**Sample response:**

```json
{
  "conversation_id": "5a1090ee-e3a8-46aa-9bcc-7653d4c90ca5",
  "response": "Deal validation complete: 3 deals validated for San Jose, May 1-15. 2 passed weather-based rules, 1 flagged for temperature threshold review.",
  "request_type": "agent_task",
  "agents_used": ["rules_engine_agent_v1"],
  "rag_used": false,
  "correlation_id": "a466098e-dabb-4074-b891-e0a2b94ac93f"
}
```

**Other rules engine examples:**

```bash
# Different location
"Validate deals for Houston for the period 2024-06-01 to 2024-06-15"

# Explicit validation request
"Run deal validation rules for Los Angeles"

# Check what failed
"Which deals failed validation in Chicago?"
```

**Workflow trace — step by step:**

```
POST /api/v1/chat
        │
        ▼
[CorrelationMiddleware]         Sets X-Correlation-ID
        │
        ▼
[chat router]                   sanitize_user_input() → clean, no injection patterns
        │
        ▼  GraphState initial
           messages = [Message(role="user", content="Validate deals for San Jose...")]
        │
        ▼
[RouterNode]                    LLM classifies → "agent_task"
                                Heuristic detects "deal", "validate", "san jose"
                                → routing_hint = "rules_engine"
        │
        ▼
[PlannerNode]                   Looks up capability RULES_ENGINE in AgentRegistry
                                → selected_agents = ["rules_engine_agent_v1"]
        │
        ▼
[OrchestratorNode]              HITL disabled → requires_approval = False
        │
        ▼
[ExecutorNode]                  Calls RulesEngineAgent.execute(task)
                                  └─ Parses "San Jose" from instruction
                                  └─ start_date/end_date from context (defaults 2024-05-01 / 2024-05-15)
                                  └─ WeatherDataService.get_conditions("San Jose", "2024-05-01", "2024-05-15")
                                       └─ Returns WeatherConditions(avg_temp=57°F, max_wind=13mph, ...)
                                  └─ DealsService.validate_deals(location, dates, weather)
                                       └─ SJ-001: temp 57°F < threshold 75°F → PASSED
                                       └─ SJ-002: temp 57°F < threshold 70°F → PASSED
                                       └─ SJ-003: temp 57°F < threshold 80°F, wind check → PASSED or FLAGGED
                                  └─ Formats DealsValidationResult → output string
                                  └─ Returns AgentResult(success=True, output="Deal validation...")
        │
        ▼
[CompletionGateNode]            tool_results non-empty → gate_passed = True
        │
        ▼
[SynthesizerNode]               Builds system prompt from agent result
                                Calls LLM.complete() → final_response
        │
        ▼
[chat router]                   Returns ChatResponse
```

---

### 4.4 Multi-Turn Conversation (Carry Context)

Pass the same `conversation_id` to continue a thread. LangGraph's `MemorySaver` checkpointer persists state per thread:

```bash
# First turn — meter query
CONV_ID=$(curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Check ERCOT meter availability for last week"}' \
  | python3.11 -c "import sys,json; print(json.load(sys.stdin)['conversation_id'])")

echo "conversation_id: $CONV_ID"

# Second turn — follow-up in same conversation
curl -s -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What was the offline count?\", \"conversation_id\": \"$CONV_ID\"}" \
  | python3.11 -m json.tool
```

---

### 4.5 API Reference Summary

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness probe — returns `{"status":"ok"}` |
| `/ready` | GET | Readiness probe |
| `/api/v1/chat` | POST | Send a message; receive pipeline response |
| `/api/v1/agents` | GET | List all registered agents with capabilities |
| `/api/v1/agents/{id}` | GET | Get a specific agent's details |
| `/docs` | GET | Swagger UI (interactive) |
| `/redoc` | GET | ReDoc documentation |

**Chat request body:**

```json
{
  "message":         "string — required",
  "conversation_id": "string — optional; omit to start new conversation",
  "user_id":         "string — optional; default 'anonymous'",
  "metadata":        "object — optional; passed to graph state metadata"
}
```

**Chat response body:**

```json
{
  "conversation_id": "UUID of the conversation thread",
  "response":        "Natural language response from the synthesizer",
  "request_type":    "agent_task | rag_query | domain_query | general_chat",
  "agents_used":     ["list of agent IDs that executed"],
  "rag_used":        "true if RAG context was retrieved",
  "correlation_id":  "UUID for distributed tracing (echoed in X-Correlation-ID header)"
}
```

---

### 4.6 Routing Keywords Reference

The `RouterNode` uses keyword heuristics as fallback when LLM classification fails:

| Keywords detected | `request_type` | `routing_hint` | Agent dispatched |
|------------------|---------------|---------------|-----------------|
| `meter`, `availability`, `market`, `duration`, `ercot` | `agent_task` | `meter_data` | `meter_data_agent_v1` |
| `deal`, `validate`, `rules`, `san jose` | `agent_task` | `rules_engine` | `rules_engine_agent_v1` |
| `document`, `search`, `find in`, `retrieve` | `rag_query` | — | (RAG retrieval) |
| `weather` | `domain_query` | `weather` | general agent |
| _(none matched)_ | `general_chat` | — | none (synthesizer answers directly) |
