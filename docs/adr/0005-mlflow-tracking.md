# ADR 0005 — MLflow for Experiment Tracking

**Status:** Accepted  
**Date:** 2026-06-06

## Context

Pipeline invocations need to be tracked for debugging, performance analysis, and compliance. The tracking backend should be swappable, and its absence should never crash the platform (CI runs without a tracking server).

## Decision

Use **MLflow** for experiment tracking via the `MLflowTracker` wrapper in `genie.tracking`.

Design choices:

- `MLflowTracker` lazy-imports `mlflow` at startup. If `mlflow` is not installed or the server is unreachable, it degrades to a silent no-op — the pipeline continues without tracking.
- The `start_run()` async context manager wraps each pipeline invocation. It records `pipeline_duration_ms`, sets a `status` tag (`succeeded` / `failed`), and calls `mlflow.end_run()` on exit.
- Callers log structured params and metrics via `ctx.log_params()` and `ctx.log_metrics()` — no direct `mlflow` imports outside `genie.tracking`.
- `enable_tracking: bool` in `Settings` lets CI disable tracking entirely via environment variable.

## Consequences

**Positive**
- MLflow's UI provides a searchable history of pipeline runs with params and metrics.
- The no-op fallback means CI and local development work without a tracking server.
- `MLflowTracker` can be swapped for Weights & Biases or Neptune by replacing the wrapper — callers are unaffected.

**Negative**
- MLflow adds ~20 MB to the dependency tree.
- In-process tracking is synchronous; high-frequency calls add latency.

## Alternatives considered

- **OpenTelemetry spans:** better for distributed tracing but heavier to configure; MLflow is better suited for experiment/param tracking.
- **Custom JSON log files:** no server, no query UI — rejected for production use.
