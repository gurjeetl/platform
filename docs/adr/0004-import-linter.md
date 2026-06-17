# ADR 0004 — import-linter as Hard Architectural Boundary Gate

**Status:** Accepted  
**Date:** 2026-06-06

## Context

In a modular monolith, teams frequently bypass intended layer boundaries via direct imports. These violations are invisible until the codebase has already become entangled. We need a mechanism that fails CI when a boundary is crossed, not just a convention that relies on code review discipline.

## Decision

Use **import-linter** (`lint-imports`) with four contracts defined in `.importlinter`:

| Contract | Rule |
|----------|------|
| `no-domain-outward` | `genie.domain` may not import from `interface`, `application`, `llm`, `rag`, `tools`, `security`, or `tracking` |
| `no-cross-module-internals` | `genie.application` nodes may not import agent internals directly (must go via `genie.agents` public API) |
| `interface-not-past-application` | `genie.interface` may not import domain models, LLM internals, RAG adapters, or agent internals directly |
| `no-rag-service-internals` | The `genie` package may not import from `rag_service` (the extracted microservice) |

`lint-imports` is run as a pre-commit hook and in CI (`make lint`). Violations fail the build hard.

## Consequences

**Positive**
- Boundary violations are caught at commit time, not in production.
- New engineers get immediate, actionable feedback if they cross a layer.
- The contracts are documentation — reading `.importlinter` explains the architecture.

**Negative**
- Adds a pre-commit step that can slow down initial setup.
- Some legitimate refactors require updating contracts first (which is the intent).

## Alternatives considered

- **Code-review conventions only:** rejected — too inconsistent across contributors.
- **Separate packages per layer:** rejected — makes cross-cutting concerns (logging, config) awkward and adds packaging overhead.
