"""MLflow experiment tracker — records pipeline runs and metrics."""
from __future__ import annotations

import contextlib
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Generator

from genie.observability.logging import get_logger

logger = get_logger(__name__)


@contextmanager
def node_span(name: str) -> Generator[Any, None, None]:
    """Synchronous MLflow CHAIN span for a LangGraph node.

    Used as ``with node_span("router") as span:`` inside async node __call__
    methods.  Falls back to a no-op (yields None) when MLflow is unavailable or
    when no active trace exists — the node runs unchanged.

    Explicit node spans are cross-platform reliable.  mlflow.langchain.autolog
    also creates node spans via LangChain callbacks, but async contextvar
    propagation differences across Python versions (3.11 vs 3.13) cause the
    autolog spans to end up in a separate trace on some platforms.
    """
    try:
        import mlflow
        with mlflow.start_span(name=name, span_type="CHAIN") as span:
            yield span
    except Exception:
        yield None


class MLflowTracker:
    """Wrapper around MLflow that records Genie pipeline invocations.

    Lazy-imports mlflow so the rest of the platform works without it installed.
    When MLflow is unavailable, all methods are silent no-ops.
    """

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._enabled = False
        self._experiment_set = False
        self._setup()

    def _setup(self) -> None:
        try:
            import mlflow as _mlflow

            _mlflow.set_tracking_uri(self._tracking_uri)
            # set_experiment() makes a synchronous HTTP call to the tracking server.
            # Skip it here — call it lazily on first start_run() so startup never blocks.
            self._enabled = True
            logger.info(
                "mlflow_initialized",
                uri=self._tracking_uri,
                experiment=self._experiment_name,
            )
        except ImportError:
            logger.warning("mlflow_not_installed", detail="tracking disabled")
            return
        except Exception as exc:
            logger.warning("mlflow_init_failed", error=str(exc))
            return

        # Autolog is optional — if it fails, basic run tracking still works.
        try:
            import mlflow as _mlflow
            import mlflow.langchain
            _mlflow.langchain.autolog(log_traces=True)
            logger.info("mlflow_autolog_enabled")
        except Exception as exc:
            logger.warning("mlflow_autolog_failed", error=str(exc))

    def _ensure_experiment(self) -> None:
        """Lazily call set_experiment() on first use — avoids blocking startup."""
        if self._experiment_set:
            return
        try:
            import mlflow as _mlflow
            _mlflow.set_experiment(self._experiment_name)
            self._experiment_set = True
        except Exception:
            pass

    @asynccontextmanager
    async def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> AsyncIterator["RunContext"]:
        """Async context manager that wraps an MLflow run."""
        self._ensure_experiment()
        ctx = RunContext(
            tracker=self,
            run_name=run_name,
            tags=tags or {},
        )
        async with ctx:
            yield ctx

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._enabled:
            return
        with contextlib.suppress(Exception):
            import mlflow

            mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self._enabled:
            return
        with contextlib.suppress(Exception):
            import mlflow

            mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        if not self._enabled:
            return
        with contextlib.suppress(Exception):
            import mlflow

            mlflow.log_artifact(path)

    def set_tag(self, key: str, value: str) -> None:
        if not self._enabled:
            return
        with contextlib.suppress(Exception):
            import mlflow

            mlflow.set_tag(key, value)


class RunContext:
    """Context object returned by MLflowTracker.start_run()."""

    def __init__(
        self,
        tracker: MLflowTracker,
        run_name: str,
        tags: dict[str, str],
    ) -> None:
        self._tracker = tracker
        self._run_name = run_name
        self._tags = tags
        self._start_time: float = 0.0
        self._run: Any = None

    async def __aenter__(self) -> "RunContext":
        self._start_time = time.perf_counter()
        if self._tracker._enabled:
            with contextlib.suppress(Exception):
                import mlflow

                self._run = mlflow.start_run(run_name=self._run_name, tags=self._tags)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        elapsed = (time.perf_counter() - self._start_time) * 1000
        if self._tracker._enabled and self._run is not None:
            with contextlib.suppress(Exception):
                import mlflow

                mlflow.log_metric("pipeline_duration_ms", elapsed)
                if exc_type is not None:
                    mlflow.set_tag("status", "failed")
                    mlflow.set_tag("error", str(exc))
                else:
                    mlflow.set_tag("status", "succeeded")
                mlflow.end_run()
        logger.debug(
            "run_context_closed",
            run_name=self._run_name,
            elapsed_ms=round(elapsed, 1),
            success=exc_type is None,
        )

    def log_params(self, params: dict[str, Any]) -> None:
        self._tracker.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._tracker.log_metrics(metrics, step=step)
