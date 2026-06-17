# ADR 0006 — Human-in-the-Loop via LangGraph Interrupt

**Status:** Accepted  
**Date:** 2026-06-06

## Context

Some agent tasks (e.g., high-risk deal validation, large-volume market commands) require human review before execution. The approval mechanism must be asynchronous — the pipeline pauses, an operator reviews the pending action, and the pipeline resumes with a binary approved/rejected decision. No polling loops should exist in the application code.

## Decision

Use **LangGraph's `interrupt_before` compile option** to pause the graph at the `human_approval` node.

Flow:

1. `OrchestratorNode` evaluates `(enable_hitl AND NOT hitl_auto_approve AND request_type in high_risk AND risk_level in [high, medium])`. If true, it sets `requires_approval=True` and writes a `hitl_prompt` string to state.
2. The compiled graph has `interrupt_before=[NODE_HUMAN_APPROVAL]`. LangGraph serialises state to the `MemorySaver` checkpointer and returns a partial result to the caller.
3. The caller (API handler or CLI) presents `hitl_prompt` to the operator.
4. The operator resumes the graph by invoking `graph.ainvoke({"approved": True/False}, config=thread_config)`. LangGraph reloads the checkpointed state and continues from `human_approval`.
5. `human_approval` is a no-op node — it exists only as an interrupt anchor. The actual approval flag flows through `GraphState.approved`.

**Config flags:**
- `enable_hitl: bool` — master on/off switch.
- `hitl_auto_approve: bool` — when `True`, HITL checks run but approvals are automatic (safe for CI).

## Consequences

**Positive**
- No polling, no callbacks, no webhooks — HITL is native to LangGraph.
- State is persisted by the checkpointer; the pause can survive server restarts (if `MemorySaver` is replaced with a persistent store).
- `hitl_auto_approve=True` makes tests deterministic without disabling the HITL code path.

**Negative**
- `MemorySaver` is in-process only — a persistent checkpointer (Redis, Postgres) is needed for multi-replica deployments.
- The HITL pause is not visible via HTTP (the `/chat` endpoint blocks); a webhook or SSE pattern is needed for long-lived approvals.

## Alternatives considered

- **Separate approval endpoint with a database:** more flexible but requires significant extra infrastructure.
- **Synchronous approval via timeout:** rejected — approval latency is unpredictable; blocking a thread is wasteful.
