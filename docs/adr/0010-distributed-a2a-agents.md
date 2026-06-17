# 10. Distributed agents over A2A as the primary agent model

Date: 2026-06-17

## Status

Accepted

## Context

The merged platform keeps BaseAgentFramework's functionality, where agents are
**independent services** that self-register with a discovery service and are invoked
over A2A JSON-RPC — not in-process plugins. Genie's pipeline, however, selects agents
from an in-process `AgentRegistry` and calls `agent.execute(...)`. We must preserve
the distributed model without rewriting the planner/executor or breaking Genie's
layered import boundaries.

## Decision

- **Bridge, don't rewrite.** Add `genie.agents.remote.RemoteAgent`, which implements
  Genie's existing `BaseAgent` protocol but dispatches `execute` as an A2A JSON-RPC
  `message/send` to the agent's endpoint, mapping the reply into an `AgentResult`.
  The planner/executor are unchanged — local and remote agents look identical.
- Add `genie.a2a` (JSON-RPC message types + client) and `genie.discovery`
  (`AgentMeta` model + async `DiscoveryClient`).
- The bootstrap **discovery bridge** queries the registry on startup and registers one
  `RemoteAgent` per live agent into the in-process registry, then keeps the set fresh
  on a background loop (`agent_refresh_seconds`). `agent_mode` selects `distributed`
  (default), `hybrid` (in-process providers + discovered), or `local` (tests/dev).
- Extract the discovery service to `services/registry/` (`registry-service`, port 2005)
  and the self-registering agent harness to `packages/genie-agent-sdk` — both
  **independent deployables** that do not import `genie.*`. `import-linter` enforces
  that the platform never imports their internals.

## Consequences

- Genie's in-process registry stays the planner's single source of truth; the
  distributed registry feeds it via `RemoteAgent`.
- Agents are added by running a new service on the SDK pointed at the registry — no
  platform code change (`src/app.py` injects no providers in distributed mode).
- `AgentMeta`/A2A types are intentionally duplicated across the platform, the registry
  service, and the SDK so each deployable is independently installable; the wire
  contract (JSON) is the coupling point.
