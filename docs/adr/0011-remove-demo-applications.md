# 11. Remove the DLR / webTrader demo applications

Date: 2026-06-17

## Status

Accepted

## Context

The platform was re-homed onto Genie's structure but is now filled with
BaseAgentFramework's functionality (distributed agents, DAG planner/wave orchestrator/
replan gate/blackboard synthesizer, mandatory guards, multi-store memory, the chat +
trace UIs). Genie's original *functional* content — the in-process `applications/dlr`
and `applications/webTrader` demo agents, the `services/dlr/mcp` demo service, and the
local keyword-search RAG default — represents the behavior we are explicitly NOT
keeping.

## Decision

- Remove `src/applications/dlr`, `src/applications/webTrader`, `services/dlr/mcp`, and
  their tests/fixtures, plus the DLR/webTrader-specific `import-linter` contracts.
- Replace them with distributed example agents on the SDK (`services/agents/`:
  weather, outage, rag) backed by an MCP tool server — these are the new reference
  implementations and exercise the full A2A path.
- The agent execution model is now distributed-first; `src/app.py` injects no
  in-process providers.

## Consequences

- The kept-from-Genie surface is purely structural (kernel layering, config system,
  import-linter, bootstrap/DI, observability, tracking, `uv`, tests, ADRs).
- Anyone needing an in-process agent can still use `agent_mode=hybrid` and the existing
  provider-injection seam in `src/app.py` / `create_app(agent_providers=[...])`.
