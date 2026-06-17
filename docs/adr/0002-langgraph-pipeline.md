# ADR 0002 — LangGraph StateGraph as the Pipeline and Agent Framework

**Status:** Accepted  
**Date:** 2026-06-06  
**Updated:** 2026-06-07

---

## Context

The platform requires a structured orchestration engine to:
- Route user requests across agents, RAG, and general chat
- Execute multiple agents in parallel or sequence
- Support human-in-the-loop (HITL) pauses and resumption without polling
- Persist conversation state across multi-turn interactions
- Be testable node-by-node without running a full LLM

Two frameworks were considered: **LangGraph** (by LangChain) and **Google Agent Development Kit (ADK)**.

---

## Agent Framework Comparison: LangGraph vs Google ADK

| Criteria | LangGraph | Google ADK | Winner |
|----------|-----------|------------|--------|
| **Learning curve** | Moderate — directed graph model is well-understood by engineers familiar with state machines | Steep — ADK introduces proprietary concepts (Runner, AgentTool, Session) with limited community examples | LangGraph |
| **Enterprise flexibility** | High — works with any LLM provider (OpenAI, Anthropic, vLLM, local models) via `BaseChatModel` | Tighter coupling to Google Vertex AI; using other LLM providers requires additional adapters | LangGraph |
| **Multi-agent orchestration** | Native — parent/child graphs, parallel node execution, graph-of-graphs composition | Supported via `SequentialAgent`, `ParallelAgent`, `LoopAgent` constructs, but less composable | LangGraph |
| **Vendor lock-in risk** | Low — LangChain ecosystem is open-source; no proprietary cloud dependency | High — ADK is designed around Google Cloud services (Vertex AI, Gemini); migrating away is costly | LangGraph |
| **Human-in-the-loop** | First-class `interrupt()` primitive; state persists via checkpointer; resume with same thread ID | Supported via `RequestEscalation` but requires additional scaffolding | LangGraph |
| **State management** | `TypedDict` or Pydantic `BaseModel`; type-safe, inspectable, serialisable via `MemorySaver` or external stores | ADK `Session` object; less type-safe, harder to inspect mid-flight | LangGraph |
| **Tool/MCP integration** | Native `@tool` decorator; MCP integration via `MCPToolAdapter`; `ToolGateway` abstraction available | Supports function tools; MCP integration is experimental | LangGraph |
| **Observability** | Deep traces via `mlflow.langchain.autolog(log_traces=True)`; node-by-node waterfall in MLflow | Limited observability outside of Google Cloud Trace | LangGraph |
| **Testing** | Each node is a plain async callable; testable with `pytest` and mock LLM providers | Harder to unit-test individual steps; integration tests require ADK runtime | LangGraph |
| **Community & docs** | Large community; thousands of examples; GitHub-first development | Smaller community; documentation primarily covers Google Cloud use cases | LangGraph |
| **Streaming support** | `astream_events()` for token-by-token streaming and node-event streaming | Supported but API is less ergonomic | LangGraph |
| **Persistence / checkpointing** | `MemorySaver` (in-process), `PostgresCheckpointer`, `RedisCheckpointer` — pluggable | Google Cloud Firestore-based; external store requires custom implementation | LangGraph |
| **Deployment model** | Any Python ASGI app (FastAPI, etc.); self-hosted or cloud-agnostic | Optimised for Cloud Run / Vertex AI deployment | LangGraph |
| **Cost model** | Open-source; no per-call platform fee | ADK itself is free; Vertex AI LLM calls are billed per token | Tie |
| **Maturity** | Production-proven at scale (LangChain serves ~100k+ developers) | Newer; released 2024; production adoption still growing | LangGraph |
| **OATI infrastructure fit** | Works on-premises with vLLM (self-hosted); no Google Cloud dependency | Requires Google Cloud for full feature set; conflict with on-premises OATI data centres | LangGraph |
| **Agentic loop control** | Fine-grained — developer controls every edge, condition, and retry | More opinionated — loop behaviour is configured rather than programmed | LangGraph |
| **A2A compatibility** | Agent-to-agent calls can be implemented as sub-graph invocations | ADK has native A2A concepts but they are Google-proprietary | LangGraph |

**Decision: LangGraph wins on 16 of 18 criteria.**

The two most decisive factors for OATI are:
1. **Vendor lock-in risk** — OATI operates its own data centres and cannot accept a hard dependency on Google Cloud for the core AI pipeline.
2. **HITL as a first-class primitive** — grid operations require human approval before high-impact automated actions; LangGraph `interrupt()` is the cleanest implementation of this requirement.

---

## Pipeline Design Decision

Use **LangGraph `StateGraph`** with a typed **Pydantic `GraphState`** as the pipeline engine.

### 7-Node Pipeline

```
router → planner → orchestrator → human_approval → executor → completion_gate → synthesizer
                                        ↑                             |
                                        └─────── retry (max 2) ───────┘
```

| Node | Responsibility |
|------|----------------|
| `router` | Classify intent: `rag_query`, `agent_task`, `domain_query`, `general_chat` |
| `planner` | Select agents from registry based on classified intent and routing hint |
| `orchestrator` | Evaluate HITL policy; set `requires_approval` if task is high-risk |
| `human_approval` | LangGraph `interrupt()` — pipeline pauses here until the client resumes |
| `executor` | Run selected agents and/or RAG retrieval; record results in state |
| `completion_gate` | Validate output quality; signal retry if result is empty or incomplete |
| `synthesizer` | Generate final natural-language response from LLM with full context |

### State Design

All pipeline state flows through `GraphState` (Pydantic `BaseModel`):

```python
class GraphState(BaseModel):
    messages: list[Message]
    conversation_id: str
    request_type: str | None = None    # set by router
    selected_agents: list[str] = []    # set by planner
    requires_approval: bool = False    # set by orchestrator
    agent_results: list[AgentResult] = []  # set by executor
    rag_results: list[RagResult] = []      # set by executor
    final_response: str | None = None      # set by synthesizer
    retry_count: int = 0
    metadata: dict[str, Any] = {}
```

No mutable dicts, no hidden side-channels. Every field is type-annotated and validated by Pydantic.

### Multi-turn Conversation

`MemorySaver` checkpointer persists `GraphState` per `thread_id` (= `conversation_id`). Subsequent requests with the same `conversation_id` resume from the last checkpoint — HITL pause/resume and multi-turn context are both handled by the same mechanism.

---

## Consequences

**Positive**
- Each node is a plain async callable; unit-testable with mock LLM providers in isolation.
- HITL is a first-class interrupt — no polling, no callbacks, no extra infrastructure.
- MLflow `langchain.autolog` provides node-by-node trace waterfall with zero additional code.
- LangGraph's `StateGraph` handles retry logic, conditional edges, and parallel branches natively.
- Self-hosted on OATI infrastructure with vLLM — no Google Cloud dependency.

**Negative**
- `MemorySaver` is in-process; multi-replica deployments require an external checkpointer (Redis, `langgraph-checkpoint-postgres`).
- LangGraph API stability: minor breaking changes between minor versions; pin `langgraph` version in `pyproject.toml`.
- More verbose than a simple `asyncio` pipeline for trivial single-agent use cases.

---

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Google ADK | Hard Google Cloud dependency; poor fit for OATI on-premises infrastructure; weaker HITL model |
| Custom async pipeline | Re-implements graph routing, retry, checkpointing, and HITL from scratch; no observability |
| LangChain `Chain` | Less structured state; HITL requires significant custom code; no graph topology |
| Prefect / Airflow | Designed for batch data pipelines, not interactive AI request handling |
| AutoGen | Multi-agent chat loop model; harder to embed in a request/response API |
