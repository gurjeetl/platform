# Agent Communication and Memory — Architecture Reference

**Date:** 2026-06-07  
**Audience:** Engineering leads, solution architects, platform reviewers

This document answers seven questions raised in the architecture review meeting. Every answer is grounded in the current codebase; future recommendations are explicitly labelled.

---

## Question 1 — How will agents communicate with each other?

### Current behaviour (Phase 1)

Agents do **not** communicate directly. The `ExecutorNode` is the only caller of agents. It iterates `GraphState.selected_agents` and calls each agent's `execute()` method independently.

```
                        ┌─────────────────────────────────┐
                        │          ExecutorNode            │
                        │                                  │
   GraphState ────────► │  for agent_id in selected_agents │
   .selected_agents     │      agent.execute(task)         │
                        │          │                       │
                        │          ▼                       │
                        │   tool_gateway.execute()         │  ← REST / MCP call
                        │          │                       │
                        │          ▼                       │
                        │   GraphState.tool_results.append │  ← result written to shared state
                        └─────────────────────────────────┘
```

Agents are **isolated units** — they receive a task instruction and return a result. They cannot call each other.

### Planned behaviour (Phase 2 — A2A protocol)

The selected standard is **Agent-to-Agent (A2A)**, as documented in [ADR 0007](../adr/0007-technology-stack.md). Under A2A:

- Each agent publishes an `agent-card.json` manifest that advertises its capabilities and endpoint.
- An orchestrating agent calls a sub-agent via HTTP using the standardised A2A schema (`Task`, `Artifact`, `Message`).
- The called agent can be written in any language and deployed independently.

```
  Orchestrating Agent (LangGraph)
        │
        │  POST /a2a/task
        ▼
  ┌─────────────────┐        ┌─────────────────┐
  │  conductor_agent │  A2A  │  dlr_calc_agent  │
  │  (Python)        │──────►│  (Java / C++)    │
  └─────────────────┘        └─────────────────┘
        │                           │
        │◄──────────────────────────┘
        │  TaskResult (JSON)
        ▼
   GraphState.tool_results
```

### Current workaround — passing results between agents

If one agent needs another's output **within the same request**, the executor can be configured to pass prior results forward. As of the current code, each agent receives `rag_context` in its task context. To pass agent-A's result to agent-B, the executor would need to accumulate `tool_results` and pass them in the task context:

```python
# executor.py — forward previous results to each successive agent
task = AgentTask(
    ...
    context={
        "rag_context": [...],
        "prior_results": [tr.model_dump() for tr in tool_results],  # add this
    },
)
```

This is a **pending enhancement** — not yet implemented. Today, agents within the same request do not see each other's outputs.

---

## Question 2 — How will memory be used? What types of memory will we have?

There are four distinct memory layers, each with a different scope and lifetime.

```
┌──────────────────────────────────────────────────────────────────┐
│                         Memory Layers                            │
│                                                                  │
│  ┌──────────────────┐  scope: one request   lifetime: seconds   │
│  │  Working Memory  │  GraphState (Pydantic model)              │
│  │                  │  tool_calls, tool_results, rag_context     │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌──────────────────┐  scope: one session   lifetime: hours*    │
│  │  Session Memory  │  MemorySaver (LangGraph checkpointer)     │
│  │                  │  Full GraphState per turn, keyed by       │
│  │                  │  conversation_id (thread_id)              │
│  └──────────────────┘  * until server restart in current build  │
│                                                                  │
│  ┌──────────────────┐  scope: per user      lifetime: weeks     │
│  │  Long-term       │  InMemoryLongTermStore (dev)              │
│  │  Memory          │  key/value store keyed by user_id         │
│  │                  │  production: PostgreSQL or Redis           │
│  └──────────────────┘                                           │
│                                                                  │
│  ┌──────────────────┐  scope: domain-wide   lifetime: permanent │
│  │  RAG / Knowledge │  Milvus vector DB (production)            │
│  │  Base            │  LocalRAGAdapter keyword index (dev)      │
│  │                  │  retrieved at query time, not per-user    │
│  └──────────────────┘                                           │
└──────────────────────────────────────────────────────────────────┘
```

| Layer | Stores | Read by | Write by | Backed by (dev) | Backed by (prod) |
|-------|--------|---------|----------|-----------------|-----------------|
| Working | tool_calls, tool_results, rag_context, messages | All nodes | executor, synthesizer | Python object (`GraphState`) | Same — ephemeral |
| Session | Full conversation state per turn | chat router, synthesizer | LangGraph checkpoint | `MemorySaver` (in-process) | `PostgresCheckpointer` or Redis |
| Long-term | User preferences, learned facts | Synthesizer (future) | Application agents (future) | `InMemoryLongTermStore` | PostgreSQL table |
| RAG | Domain documents, specs, standards | executor (RAG path) | Ingestion endpoint | `LocalRAGAdapter` | Milvus + embedding model |

### How session memory enables multi-turn conversation

Every turn in a conversation uses the same `conversation_id` as the LangGraph `thread_id`. `MemorySaver` checkpoints the full `GraphState` after each turn. On the next turn, the graph reloads the checkpoint — so `messages` already contains the full conversation history and agents get context without any explicit history management.

```
Turn 1: "Give me ACSR conductors"
  → GraphState saved to checkpoint [thread=conv-001]

Turn 2: "Show me only active ones"
  → checkpoint [thread=conv-001] loaded
  → GraphState.messages already contains turn-1 exchange
  → agents receive full context
```

---

## Question 3 — Who is responsible for tracking the outputs from the agents?

Responsibility is distributed across four components — each layer serves a different purpose.

```
                          agent.execute()
                               │
                               ▼
┌─────────────────────────────────────────────────────────┐
│                      ExecutorNode                        │
│                                                          │
│  1. Records ToolCallRecord  ──► GraphState.tool_calls   │  ← what was asked
│  2. Records ToolResultRecord ──► GraphState.tool_results│  ← what came back
│  3. Publishes EventBus event ──► TOPIC_AGENT_EXECUTED   │  ← async subscribers
└─────────────────────────────────────────────────────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌─────────────────────┐
│ CompletionGate   │         │   MLflowTracker      │
│                  │         │                      │
│ Checks if any    │         │ Logs per-request run │
│ results exist.   │         │ params: agents_used  │
│ Signals retry if │         │ metrics: latency,    │
│ empty or error.  │         │ response_length      │
└──────────────────┘         │ traces: LangChain    │
         │                   │ node waterfall        │
         ▼                   └─────────────────────┘
┌──────────────────┐
│  SynthesizerNode │
│                  │
│ Reads ALL        │
│ tool_results and │
│ rag_context from │
│ GraphState to    │
│ compose the      │
│ final response   │
└──────────────────┘
```

**Summary of ownership:**

| Component | Tracks | Where |
|-----------|--------|-------|
| `ExecutorNode` | Every agent call and its result | `GraphState.tool_calls` + `tool_results` |
| `CompletionGateNode` | Whether results meet quality threshold | `GraphState.metadata.gate_passed` |
| `SynthesizerNode` | Consumes results to produce the final answer | `GraphState.final_response` |
| `MLflowTracker` | Request-level metrics and LangChain trace waterfall | MLflow runs + traces UI |
| `EventBus` | Async event for downstream consumers (logging, alerting) | `TOPIC_AGENT_EXECUTED` topic |

---

## Question 4 — How long will memory be retained for a conversation?

### Current state (in-process)

| Memory type | Retained until |
|-------------|---------------|
| Working memory (`GraphState` for one request) | End of HTTP response |
| Session memory (`MemorySaver` checkpoint) | Server process restart (no persistence to disk) |
| Long-term memory (`InMemoryLongTermStore`) | Server process restart |
| RAG index (`LocalRAGAdapter`) | Server process restart (re-seeded at startup) |

In the current development setup, a `--reload` or server restart clears all conversation history.

### Production targets

| Memory type | Backend | Retention policy |
|-------------|---------|-----------------|
| Session checkpoints | `PostgresCheckpointer` (langgraph-checkpoint-postgres) | Configurable TTL, e.g. 7 days of inactivity |
| Long-term memory | PostgreSQL table with `last_accessed` timestamp | 90 days default, user-deletable |
| RAG index | Milvus persistent storage | Until explicit deletion via ingestion API |
| MLflow runs | MLflow + PostgreSQL | Configured per experiment, default unlimited |

### Planned configuration knob

```yaml
# config/default.yaml
session_ttl_days: 7          # conversation checkpoints expire after N days of inactivity
long_term_memory_ttl_days: 90
```

These settings do not exist yet — they are the target state for the persistence milestone.

---

## Question 5 — How are agents keeping track of each other's responses?

### Current behaviour

They are **not**. This is a known Phase 1 limitation.

Within a single request, the executor calls agents in sequence. Each agent receives:
- The user's instruction
- The RAG context (document chunks retrieved for this query)

It does **not** receive the results from agents called earlier in the same request. Agent A and Agent B run as independent units — they cannot see each other's outputs until after the executor finishes and the synthesizer reads `GraphState.tool_results`.

```
Executor loop (current)

  Agent A: execute(instruction, rag_context)
              └─► returns result_A

  Agent B: execute(instruction, rag_context)    ← does NOT see result_A
              └─► returns result_B

  GraphState.tool_results = [result_A, result_B]

  Synthesizer: reads BOTH results and composes final response
```

### Planned enhancement — prior_results forwarding

When a workflow requires chained agents (e.g., conductor agent fetches data → DLR calculation agent uses that data), the executor will pass accumulated results forward:

```
Executor loop (planned)

  Agent A: execute(instruction, rag_context, prior_results=[])
              └─► returns result_A

  Agent B: execute(instruction, rag_context, prior_results=[result_A])
              └─► returns result_B  (can reference result_A)
```

This requires:
1. Updating `AgentTask.context` to include `prior_results`
2. Agents that need upstream data to read from `context["prior_results"]`

For fully dynamic agent-to-agent calls (not just sequential), the A2A protocol (see Q1) is the longer-term path.

---

## Question 6 — What is the size limit of shared memory, and how are agents limited?

### Current limits

There are no enforced size limits today. The practical ceiling is the **LLM context window** used by the synthesizer — all `tool_results` and `rag_context` are concatenated into the system prompt. If this exceeds the model's context window, the LLM call will fail with a token limit error.

Typical model context limits:

| Model | Context window | Practical system prompt budget |
|-------|---------------|-------------------------------|
| Llama-3-8B | 8 192 tokens | ~4 000 tokens |
| Llama-3-70B | 8 192 tokens | ~4 000 tokens |
| Llama-3.1-70B | 131 072 tokens | ~60 000 tokens |
| Llama-3.1-405B | 131 072 tokens | ~60 000 tokens |

At ~4 tokens per word, a single agent output of 1 000 words ≈ 4 000 tokens. With multiple agents, budget fills quickly on smaller models.

### Recommended limits to implement

**Per-agent output cap** — truncate `AgentResult.output` before storing in `ToolResultRecord`:

```python
MAX_AGENT_OUTPUT_CHARS = 4_000  # ≈ 1 000 tokens

class ToolResultRecord(BaseModel):
    output: Any = None

    @validator("output", pre=True)
    def _truncate(cls, v):
        if isinstance(v, str) and len(v) > MAX_AGENT_OUTPUT_CHARS:
            return v[:MAX_AGENT_OUTPUT_CHARS] + " … [truncated]"
        return v
```

**RAG chunk limit** — already parameterised in `RetrievalRequest.top_k`. Keep at ≤ 5 chunks for small models.

**Total context budget check in synthesizer** — before calling the LLM, estimate token count and drop the lowest-priority tool results if over budget:

```
total_tokens = estimate(system_prompt + all tool_results + all rag_context)
if total_tokens > MAX_CONTEXT_TOKENS:
    drop least-relevant tool_results until within budget
```

**Proposed configuration:**

```yaml
# config/default.yaml
max_agent_output_chars: 4000   # per agent, per request
max_rag_chunks: 5              # retrieval top_k
max_context_tokens: 6000       # synthesizer budget before truncation
```

These controls are **not yet implemented** — they are the recommended next step for production hardening.

---

## Question 7 — Access to local file system / database for intermediate agent-level results

### Current behaviour

Agents have **no file system writes**. Intermediate results live only in `GraphState` (in-memory Python object). Agents read static data from JSON fixture files at startup (e.g., `rules_engine/fixtures/deals.json`) but never write back.

```
Agent                        Storage
─────                        ───────
reads  ──────────────────►  JSON fixture (read-only, loaded at import time)
writes ──────────────────►  AgentResult (in-memory, returned to executor)
                             │
                             ▼
                         GraphState.tool_results  (in-memory)
```

### Options for intermediate persistence (by use case)

| Use case | Recommended approach | Notes |
|----------|---------------------|-------|
| Large agent outputs (> 10 MB) | Write to a temp file; store file path in `ToolResultRecord.output` | Platform cleans up temp files after request completes |
| Agent needs to resume a multi-step computation | Store progress in `InMemorySessionStore` keyed by `conversation_id` | Survives within a session; lost on restart in dev |
| Agent needs to read/write a shared dataset | Mount a shared volume in Kubernetes; access via file path from config | Not agent-specific; available to all agents on the node |
| Durable intermediate results across server restarts | Write to PostgreSQL via a simple repository class | Recommended for production workflows that span multiple turns |
| High-throughput transient data | Redis | Sub-millisecond read/write; TTL-based cleanup |

### Recommended pattern for agents that need intermediate storage

Rather than giving agents direct database access, the platform provides a **repository protocol** that agents receive via their tool gateway or task context. This keeps infrastructure concerns out of agent code:

```python
# Future pattern (not yet implemented)
class IntermediateStore(Protocol):
    async def save(self, key: str, value: Any, ttl_seconds: int = 3600) -> None: ...
    async def load(self, key: str) -> Any | None: ...
    async def delete(self, key: str) -> None: ...

# Agent uses it without knowing the backend
class ConductorAgent:
    async def execute(self, task, context):
        store = context.get("intermediate_store")
        cached = await store.load(f"conductor:{task.task_id}") if store else None
        if cached:
            return AgentResult(..., output=cached)
        result = await self._fetch(...)
        if store:
            await store.save(f"conductor:{task.task_id}", result.output)
        return result
```

The platform would wire either `RedisIntermediateStore` or `InMemoryIntermediateStore` depending on the deployment environment, following the same port+adapter pattern used for RAG (see [ADR 0003](../adr/0003-rag-port-adapter.md)).

---

## Summary Table

| Question | Current state | Production target |
|----------|--------------|------------------|
| Agent-to-agent communication | None; all routed through ExecutorNode | A2A HTTP protocol; `prior_results` forwarding for chained calls |
| Memory types | Working (GraphState), session (MemorySaver), long-term (InMemoryLongTermStore), RAG (LocalRAGAdapter) | Same types; all backed by persistent stores (Postgres, Redis, Milvus) |
| Output tracking | ExecutorNode → `tool_calls` + `tool_results` in GraphState; MLflow traces | Same; add per-agent token usage metrics |
| Memory retention | Until server restart | Configurable TTL per memory layer (7d session, 90d long-term) |
| Inter-agent result visibility | None within a request; synthesizer sees all after executor finishes | `prior_results` forwarding in executor for sequential chains |
| Shared memory size limits | None enforced; LLM context window is the practical ceiling | Per-agent output cap (4 000 chars), RAG top_k limit, synthesizer token budget |
| Intermediate file/DB storage | Not available; all results in-memory GraphState | `IntermediateStore` protocol wired to Redis or PostgreSQL via tool context |
