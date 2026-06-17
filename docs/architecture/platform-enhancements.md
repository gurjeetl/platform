# Platform Robustness Enhancements

**Date:** 2026-06-07  
**Audience:** Platform team, engineering leads

This document catalogues enhancements ranked by impact and implementation effort, grounded in the current codebase state. Each item notes the exact file and line where the change belongs.

---

## What is already in place

Before listing gaps, here is what the platform already implements correctly so the team has the full picture.

| Feature | Status | Location |
|---------|--------|----------|
| Prompt injection detection (input) | ✅ | `genie/security/auth.py` — regex patterns + max length 8 192 chars |
| API key authentication (HMAC constant-time) | ✅ | `genie/security/auth.py:61` |
| Per-component timeouts | ✅ | RAG 30 s · MCP 15 s · App services 5 s |
| RAG retry (3×, 5xx-discriminating) | ✅ | `genie/rag/adapters/remote.py:47` |
| Executor retry (2×, via completion gate) | ✅ | `genie/application/nodes/completion_gate.py:11` |
| LLM streaming protocol | ✅ | `genie/llm/base.py:32` — `stream()` method |
| LLM graceful degradation (returns placeholder) | ✅ | `genie/llm/openai_compat.py:79` |
| RAG graceful degradation (`retrieval_available=False`) | ✅ | `genie/rag/adapters/remote.py:82` |
| MCP startup failure isolation | ✅ | `genie/interface/bootstrap.py:226` |
| Per-model token cap | ✅ | `genie/llm/openai_compat.py:63` |
| Event bus with error isolation per subscriber | ✅ | `genie/platform/event_bus.py:94` |
| Correlation IDs on every request | ✅ | `genie/observability/correlation.py` |
| MLflow experiment tracking + LangChain trace waterfall | ✅ | `genie/tracking/mlflow_tracker.py` |
| In-memory metrics counters / histograms | ✅ | `genie/observability/metrics.py` |
| HITL approval gate | ✅ | `genie/application/nodes/orchestrator.py` |

---

## Enhancement 1 — Parallel Agent Execution (High Impact · Low Effort)

### Gap

`ExecutorNode` calls agents in a sequential `for` loop. If three agents are selected (e.g. conductor + meter_data + rules_engine), they execute one after another. Total latency = sum of all agent latencies.

**File:** [src/genie/application/nodes/executor.py:68](../../src/genie/application/nodes/executor.py#L68)

```python
# Current — sequential
for agent_id in state.selected_agents:
    result = await agent.execute(task, ...)
```

### Fix

Replace the loop with `asyncio.gather` so agents run concurrently. Total latency = slowest single agent.

```python
# Enhanced — concurrent
import asyncio

async def _run_one(agent_id):
    agent = self._registry.get(agent_id)
    if agent is None:
        return None
    task = AgentTask(agent_id=agent_id, ...)
    return agent_id, await agent.execute(task, context={"rag_context": rag_context})

pairs = await asyncio.gather(*[_run_one(aid) for aid in state.selected_agents], return_exceptions=True)
for pair in pairs:
    if isinstance(pair, Exception):
        error = str(pair)
    elif pair is not None:
        agent_id, result = pair
        tool_results.append(ToolResultRecord(...))
```

```
Before:  [agent_A: 800ms] ──► [agent_B: 600ms] ──► [agent_C: 400ms]  Total: 1 800ms
After:   [agent_A: 800ms]
         [agent_B: 600ms]  ◄── concurrent ──►  Total: 800ms
         [agent_C: 400ms]
```

**Caveat:** Only safe when agents are independent. If agent B needs agent A's result (chained workflow), keep them sequential. A `depends_on` field in `AgentInfo` can express this dependency.

---

## Enhancement 2 — Exponential Backoff on Retries (High Impact · Low Effort)

### Gap

`config/default.yaml` already has `rag_retry_backoff_factor: 0.5` but the remote RAG adapter never uses it — the retry loop has no sleep between attempts. Hitting a recovering service with three immediate retries causes a retry storm.

**File:** [src/genie/rag/adapters/remote.py:47](../../src/genie/rag/adapters/remote.py#L47)

### Fix

```python
import asyncio

for attempt in range(self._max_retries + 1):
    try:
        response = await self._client.post(...)
        if response.status_code < 500:
            break
    except Exception:
        pass
    if attempt < self._max_retries:
        backoff = self._retry_backoff_factor * (2 ** attempt)   # 0.5s, 1s, 2s
        await asyncio.sleep(backoff)
```

Apply the same pattern in `genie/llm/openai_compat.py` for LLM call retries.

---

## Enhancement 3 — Request-Level Pipeline Timeout (High Impact · Low Effort)

### Gap

Individual components have timeouts (RAG 30 s, MCP 15 s) but the overall chat endpoint has no timeout. A slow LLM call or hung agent can hold a connection open indefinitely.

**File:** [src/genie/interface/routers/chat.py](../../src/genie/interface/routers/chat.py)

### Fix

```python
# config/default.yaml
pipeline_timeout_seconds: 60.0

# chat.py
try:
    result = await asyncio.wait_for(
        graph.ainvoke(initial_state.model_dump(), config=thread_config),
        timeout=settings.pipeline_timeout_seconds,
    )
except asyncio.TimeoutError:
    raise GenieError(ErrorCode.TIMEOUT, "Pipeline did not complete within the allowed time")
```

Add `pipeline_timeout_seconds: 60.0` to `Settings` in `genie/platform/config.py`.

---

## Enhancement 4 — Streaming Responses via SSE (High Impact · Medium Effort)

### Gap

`BaseLLMProvider.stream()` is implemented in both `MockLLMProvider` and `OpenAICompatibleLLMProvider`, but the chat endpoint never calls it — it always calls `complete()` and returns a single JSON body. Users wait for the full LLM response before seeing anything.

**File:** [src/genie/interface/routers/chat.py](../../src/genie/interface/routers/chat.py)

### Fix

Add a `/api/v1/chat/stream` endpoint that returns `text/event-stream`:

```python
from fastapi.responses import StreamingResponse

@router.post("/api/v1/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    graph = request.app.state.graph
    # Run pipeline up to synthesizer, then stream the LLM response
    async def event_generator():
        async for token in llm.stream(messages, max_tokens=1024):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

The non-streaming `/api/v1/chat` endpoint remains unchanged — streaming is an additive endpoint.

---

## Enhancement 5 — Rate Limiting Per User / API Key (High Impact · Medium Effort)

### Gap

`ErrorCode.RATE_LIMITED` is defined in `genie/platform/errors.py:20` but is never raised. There is no middleware or token-bucket logic. A single client can send unlimited parallel requests.

**File:** `genie/security/auth.py` — add alongside `ApiKeyMiddleware`

### Fix

```python
# Sliding-window rate limiter stored in-process (Redis for multi-replica)
from collections import defaultdict, deque
import time

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self._rpm = requests_per_minute
        self._windows: dict[str, deque] = defaultdict(deque)

    async def dispatch(self, request, call_next):
        key = request.headers.get("X-API-Key", request.client.host)
        now = time.monotonic()
        window = self._windows[key]
        # Remove entries older than 60s
        while window and window[0] < now - 60:
            window.popleft()
        if len(window) >= self._rpm:
            return JSONResponse({"error": "rate_limited"}, status_code=429)
        window.append(now)
        return await call_next(request)
```

Wire in `bootstrap.py`:

```python
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_rpm)
```

Add `rate_limit_rpm: 60` to `config/default.yaml` and `Settings`.

---

## Enhancement 6 — Circuit Breaker for External Services (Medium Impact · Medium Effort)

### Gap

When the RAG service or LLM is down, every incoming request attempts to connect, waits for the timeout, then retries. Under load this wastes threads and delays responses.

A circuit breaker fast-fails requests immediately when a service is known to be unhealthy, and periodically probes to detect recovery.

```
State machine:
  CLOSED (normal) ──[N failures]──► OPEN (fast-fail)
  OPEN             ──[cooldown]──► HALF_OPEN (probe)
  HALF_OPEN        ──[success]──► CLOSED
  HALF_OPEN        ──[failure]──► OPEN
```

### Fix

Implement a lightweight `CircuitBreaker` class and wrap `RemoteRAGAdapter._call()` and `OpenAICompatibleLLMProvider.complete()`:

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, cooldown_seconds=30):
        self._failures = 0
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._opened_at: float | None = None
        self._state = "closed"  # closed | open | half_open

    def record_failure(self):
        self._failures += 1
        if self._failures >= self._threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    def record_success(self):
        self._failures = 0
        self._state = "closed"

    def allow_request(self) -> bool:
        if self._state == "closed":
            return True
        if self._state == "open":
            elapsed = time.monotonic() - (self._opened_at or 0)
            if elapsed > self._cooldown:
                self._state = "half_open"
                return True
            return False
        return True  # half_open: allow probe
```

Place in `genie/resilience/circuit_breaker.py`. Each adapter holds its own instance.

---

## Enhancement 7 — Output Content Filtering (High Impact · Low Effort)

### Gap

`auth.py` detects prompt injection on **input** before the LLM sees it, but the LLM **output** is returned to the user unfiltered. A jailbroken or confused model could return harmful content, PII from training data, or internal system prompt fragments.

**File:** `genie/security/auth.py` — add alongside `sanitize_user_input()`

### Fix

```python
_OUTPUT_BLOCK_PATTERNS = [
    re.compile(r"(?i)you are (?:a|an) (?:gpt|llama|genie).*?your instructions"),
    re.compile(r"(?i)<\|system\|>|<\|assistant\|>|<\|endoftext\|>"),  # model artefacts
    re.compile(r"\b(?:\d{3}-\d{2}-\d{4}|\d{4}[-\s]\d{4}[-\s]\d{4}[-\s]\d{4})\b"),  # SSN, CC
]

def sanitize_llm_output(text: str) -> str:
    for pattern in _OUTPUT_BLOCK_PATTERNS:
        if pattern.search(text):
            logger.warning("output_filter_triggered", pattern=pattern.pattern)
            text = pattern.sub("[REDACTED]", text)
    return text
```

Call in `SynthesizerNode` before returning `final_response`.

---

## Enhancement 8 — Dependency-Aware Health Checks (Medium Impact · Low Effort)

### Gap

The current `/health` endpoint likely returns `{"status": "ok"}` regardless of whether the LLM, RAG service, or database are reachable. Kubernetes liveness/readiness probes based on this endpoint will not detect a degraded-but-running instance.

**File:** `genie/interface/routers/health.py`

### Fix

Add a `/ready` endpoint that actively probes each dependency:

```python
@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    checks = {}

    # LLM probe — cheap classification call
    try:
        llm = request.app.state.llm_provider
        await asyncio.wait_for(llm.complete([Message(role="user", content="ping")], max_tokens=1), timeout=3.0)
        checks["llm"] = "ok"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    # RAG probe
    rag = request.app.state.rag_adapter
    if rag is not None:
        try:
            await asyncio.wait_for(rag.retrieve(RetrievalRequest(query="ping")), timeout=3.0)
            checks["rag"] = "ok"
        except Exception as exc:
            checks["rag"] = f"error: {exc}"

    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse({"status": "ready" if all_ok else "degraded", "checks": checks},
                        status_code=200 if all_ok else 503)
```

Kubernetes readiness probe should point at `/ready`; liveness probe at `/health`.

---

## Enhancement 9 — Completion Gate Quality Scoring (Medium Impact · Low Effort)

### Gap

The current `CompletionGateNode` only checks two conditions: (a) was there an error, (b) is the result non-empty. It cannot distinguish a partial result (agent returned "no data found") from a successful result.

**File:** [src/genie/application/nodes/completion_gate.py](../../src/genie/application/nodes/completion_gate.py)

### Fix

Add a minimum result quality check before passing to the synthesizer:

```python
_EMPTY_RESULT_PHRASES = [
    "no conductors found",
    "no data found",
    "no results",
    "error retrieving",
    "currently unavailable",
]

def _is_low_quality(tool_results) -> bool:
    for tr in tool_results:
        if tr.success and tr.output:
            output_lower = str(tr.output).lower()
            if any(phrase in output_lower for phrase in _EMPTY_RESULT_PHRASES):
                return True  # agent succeeded but found nothing meaningful
    return False
```

If quality is low and retries remain, the gate signals retry. This prevents the synthesizer from receiving empty-but-successful results and generating hallucinated responses.

---

## Enhancement 10 — Per-Agent Output Size Cap (Medium Impact · Low Effort)

### Gap

As documented in [agent-communication-and-memory.md](agent-communication-and-memory.md#question-6), there are no enforced size limits on agent outputs. A single agent returning a 50 000-character response will exceed the LLM's context window when the synthesizer assembles the system prompt.

**File:** `genie/application/state.py` (ToolResultRecord) + `genie/application/nodes/executor.py`

### Fix

```python
# config/default.yaml
max_agent_output_chars: 4000
max_rag_chunks: 5

# executor.py — truncate before storing
MAX_OUTPUT = getattr(settings, "max_agent_output_chars", 4000)

output = result.output
if isinstance(output, str) and len(output) > MAX_OUTPUT:
    output = output[:MAX_OUTPUT] + f" … [truncated, {len(result.output) - MAX_OUTPUT} chars omitted]"

tool_results.append(ToolResultRecord(..., output=output))
```

---

## Enhancement 11 — Conversation Memory Summarization (Medium Impact · Medium Effort)

### Gap

`GraphState.messages` grows indefinitely as a conversation progresses. On turn 20, the synthesizer LLM call includes 19 previous turns as context, which may overflow the context window and costs tokens on every turn.

### Fix

Add a summarization step in the chat router or a dedicated `memory_manager` component:

```python
MAX_TURNS_BEFORE_SUMMARY = 10
SUMMARY_PROMPT = "Summarize this conversation in under 200 words, retaining key facts and decisions."

async def maybe_summarize(messages, llm, threshold=MAX_TURNS_BEFORE_SUMMARY):
    turns = sum(1 for m in messages if m.role == "user")
    if turns < threshold:
        return messages
    summary = await llm.complete([Message(role="user", content=SUMMARY_PROMPT), *messages], max_tokens=300)
    return [
        Message(role="system", content=f"[Conversation summary]: {summary.content}"),
        *messages[-4:],  # keep last 2 turns verbatim for immediate context
    ]
```

Call before building the graph state for each request.

---

## Enhancement 12 — Distributed Tracing with OpenTelemetry (Medium Impact · High Effort)

### Gap

Correlation IDs are propagated through structured logs, and MLflow captures node-level LangChain traces — but there is no OpenTelemetry trace that spans from the HTTP request through to the RAG service or MCP servers. This makes it impossible to see cross-service latency in a single trace viewer.

### Fix

Install `opentelemetry-sdk`, `opentelemetry-instrumentation-fastapi`, `opentelemetry-instrumentation-httpx`:

```python
# genie/observability/tracing.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def configure_tracing(endpoint: str):
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
```

Wire in `bootstrap.py` before app creation. FastAPI and httpx auto-instrumentation then propagate the `traceparent` header to the RAG service and MCP servers automatically.

```
HTTP Request
  └─ FastAPI span (traceparent: 00-abc-001)
       ├─ RouterNode span
       ├─ PlannerNode span
       ├─ ExecutorNode span
       │    ├─ ConductorAgent span
       │    │    └─ httpx call to conductor REST API span
       │    └─ RAG retrieval span
       │         └─ httpx call to RAG service span  ← cross-service!
       └─ SynthesizerNode span
            └─ httpx call to vLLM span              ← cross-service!
```

---

## Enhancement 13 — LLM Response Caching (Low Impact now · High Impact at scale)

### Gap

Identical or near-identical prompts (e.g., repeated "give me 3 ACSR conductors" in load tests or demos) hit the full pipeline every time.

### Fix

Hash the system prompt + user message. Cache LLM responses with a configurable TTL:

```python
import hashlib, json

def _cache_key(messages, max_tokens, temperature) -> str:
    payload = json.dumps([m.model_dump() for m in messages], sort_keys=True)
    return hashlib.sha256(f"{payload}{max_tokens}{temperature}".encode()).hexdigest()

# In OpenAICompatibleLLMProvider.complete():
key = _cache_key(messages, max_tokens, temperature)
if (cached := self._cache.get(key)):
    return cached
response = await self._client.post(...)
self._cache.set(key, response, ttl=300)  # 5-minute cache
```

Use `InMemoryLRUCache` for development, Redis for production.

**Important:** Only cache when `temperature=0.0`. Stochastic responses must not be cached.

---

## Prioritised Roadmap

### Tier 1 — Implement now (days, high return)

| # | Enhancement | Effort | Benefit |
|---|-------------|--------|---------|
| 2 | Exponential backoff | 1 hour | Prevent retry storms on RAG/LLM outages |
| 3 | Request-level pipeline timeout | 1 hour | Prevent hung HTTP connections |
| 7 | Output content filtering | 2 hours | Security — filter PII and model artefacts |
| 9 | Completion gate quality scoring | 2 hours | Fewer hallucinated "no data" responses |
| 10 | Per-agent output size cap | 1 hour | Prevent LLM context overflow |
| 8 | Dependency-aware `/ready` endpoint | 3 hours | Kubernetes probes actually detect degradation |

### Tier 2 — Next sprint (weeks, medium effort)

| # | Enhancement | Effort | Benefit |
|---|-------------|--------|---------|
| 1 | Parallel agent execution | 3 hours | Latency cut proportional to agent count |
| 4 | Streaming responses (SSE) | 1 day | Perceived responsiveness; table-stakes for chat UX |
| 5 | Rate limiting middleware | 1 day | API quota management; prevent flood |
| 11 | Conversation memory summarization | 2 days | Reliable multi-turn conversations past 10 turns |

### Tier 3 — Platform hardening (months, architectural)

| # | Enhancement | Effort | Benefit |
|---|-------------|--------|---------|
| 6 | Circuit breaker | 2 days | Fast-fail under sustained service outage |
| 12 | OpenTelemetry distributed tracing | 1 week | Cross-service latency visibility |
| 13 | LLM response caching | 3 days | Cost and latency reduction at scale |
| — | Persistent session checkpointer (Postgres) | 3 days | Conversation history survives restarts |
| — | Per-conversation token budget | 2 days | Cost control per user / session |
