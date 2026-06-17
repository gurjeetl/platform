# 8. Mandatory content guards (input/output scanning)

Date: 2026-06-17

## Status

Accepted

## Context

The platform now carries the full BaseAgentFramework functionality, which treats
input/output content scanning as a **mandatory**, fail-closed control rather than an
optional add-on. The pipeline must block prompt-injection, toxicity, banned-topics
and code/script-injection payloads on the way in, and toxic/banned content on the way
out, while redacting PII/secrets in place. Genie previously only had a lightweight
regex `sanitize_user_input`.

The backing library, `llm-guard`, is heavy (pulls torch, transformers, spacy,
presidio). Genie's design principle is a lean core install with optional extras.

## Decision

- Add `genie.security.guard.LLMGuard` (ported from BaseAgentFramework) and two
  LangGraph nodes — `input_guard` (before the router) and `output_guard` (after the
  synthesizer) — wired by `build_graph` when a guard is present.
- `enable_guards` defaults **on** (BaseAgentFramework parity). On a block the input
  guard short-circuits straight to END with a safe refusal (`GenieError`/`ErrorCode
  .PROMPT_INJECTION` → HTTP 400 mapping is reused).
- `llm-guard` lives in the optional `guards` extra. `create_guard` is **fail-closed**:
  when guards are enabled but the extra is not installed it raises a clear startup
  error instructing `uv sync --extra guards` (rather than silently running
  unprotected). Set `GENIE_ENABLE_GUARDS=false` to opt out (e.g. tests, local dev).
- The read-only scanners run concurrently so a scan costs ~the slowest single scanner.

## Consequences

- The default *run* profile must install the `guards` extra; the default *test*
  profile disables guards (no torch dependency in CI).
- Guards are transparent (zero edges added) when disabled.
- PII redaction rewrites the user message in-flight so downstream agents never see
  raw sensitive data.
