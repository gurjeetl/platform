# 9. Multi-store memory behind the memory ports

Date: 2026-06-17

## Status

Accepted

## Context

BaseAgentFramework persists conversation memory, extracted facts, durable agent-output
commits, and semantic long-term memory across MongoDB (messages 24h TTL, facts,
conversations, commits), Redis (hot blackboard mirror), and Milvus (embeddings).
Genie shipped only in-memory `SessionStore`/`LongTermStore`. We need the durable
behavior without coupling the platform to those drivers in its lean core.

## Decision

- Keep Genie's `genie.memory.ports` protocols and `in_memory` implementations as the
  zero-dependency default.
- Add adapters mirroring the `genie.rag` port+adapter layout:
  `genie.memory.adapters.{mongo,redis,vector}` — each imports its driver **lazily**
  and degrades to a disabled no-op when the driver or config is absent.
- Add `genie.memory.facade.MemoryFacade` exposing the three operations the pipeline
  needs — `recall` (Milvus semantic recall, used by the planner), `query_facts`
  (Mongo, used by the planner), and `writeback` (commits + embedding + LLM
  fact-extraction, used by the synthesizer).
- Add `genie.memory.factory.create_memory(settings, llm)` (returns None when
  `memory_backend == "in_memory"`) and `create_redis(settings)` (standalone blackboard
  mirror for the executor). Drivers ship in optional extras `memory-mongo`,
  `memory-redis`, `memory-vector`.

## Consequences

- `memory_backend` defaults to `in_memory`, so tests and quick local runs need no
  database. Switching to `mongo` (+ optional redis/milvus) is config-only.
- The planner/synthesizer nodes accept an optional `memory` and no-op when it is None,
  so the durable path is fully decoupled from the graph wiring.
