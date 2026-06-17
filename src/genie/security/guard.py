"""Content guard backed by the local ``llm-guard`` library.

Ported from BaseAgentFramework ``security/llm_guard.py``. Two scanner classes:
  * BLOCKING   — a failure short-circuits the pipeline to a safe refusal
                 (prompt injection, toxicity/harmful, banned topics, regex).
  * SANITIZING — never blocks; redacts in place (PII via Anonymize/Sensitive,
                 credentials via Secrets).

On the Genie platform guards default ON (``enable_guards``) but the heavy
``llm-guard`` dependency is an optional extra — see ``create_guard``. The model
construction is eager so a mis-load surfaces at startup (fail-closed), and the
read-only classifiers run concurrently so a scan costs ~the slowest single scanner.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from genie.observability.logging import get_logger

logger = get_logger(__name__)

_BLOCKING_INPUT = {"PromptInjection", "Toxicity", "BanTopics", "Regex"}
_BLOCKING_OUTPUT = {"Toxicity", "BanTopics"}
_SANITIZING_SCANNERS = {"Anonymize", "Secrets", "Sensitive"}

_DEFAULT_BAN_TOPICS = ["violence", "self-harm", "hate speech", "illegal weapons"]

# Deterministic defense-in-depth against application-layer code/script injection.
_DEFAULT_INJECTION_PATTERNS = [
    r"(?i)<\s*script\b",
    r"(?i)\bon(error|load|click|mouseover)\s*=",
    r"(?i)javascript\s*:",
    r"(?i)<\s*iframe\b",
    r"(?i)\bunion\s+select\b",
    r"(?i);\s*drop\s+table\b",
    r"(?i)\bor\s+1\s*=\s*1\b",
    r"(?i);\s*rm\s+-rf\b",
    r"(?i)\|\s*(sh|bash)\b",
    r"\$\(",
    r"(?i)\$\{\s*jndi:",
    r"\{\{.*\}\}",
    r"(?:\.\.[\\/]){2,}",
]


class LLMGuard:
    """Eagerly-loaded local content guard. Constructed once at startup."""

    def __init__(
        self,
        *,
        ban_topics: list[str] | None = None,
        pii: bool = True,
        fail_open: bool = False,
        parallel: bool = True,
        use_onnx: bool = False,
    ) -> None:
        # Plain imports: an ImportError propagates so the caller can fail-closed.
        from llm_guard.input_scanners import (
            Anonymize,
            BanTopics,
            PromptInjection,
            Regex,
            Secrets,
        )
        from llm_guard.input_scanners import (
            Toxicity as InputToxicity,
        )
        from llm_guard.output_scanners import (
            BanTopics as OutputBanTopics,
        )
        from llm_guard.output_scanners import (
            Sensitive,
        )
        from llm_guard.output_scanners import (
            Toxicity as OutputToxicity,
        )
        from llm_guard.vault import Vault

        self._fail_open = fail_open
        self._parallel = parallel
        self._pii = pii
        topics = ban_topics or list(_DEFAULT_BAN_TOPICS)
        self._vault = Vault()

        self._input_scanners: list[Any] = [
            PromptInjection(use_onnx=use_onnx),
            InputToxicity(use_onnx=use_onnx),
            BanTopics(topics=topics, use_onnx=use_onnx),
            Regex(
                patterns=list(_DEFAULT_INJECTION_PATTERNS),
                is_blocked=True,
                match_type="search",
                redact=False,
            ),
        ]
        if self._pii:
            self._input_scanners += [Anonymize(self._vault, use_onnx=use_onnx), Secrets()]
        self._output_scanners = [
            OutputToxicity(use_onnx=use_onnx),
            OutputBanTopics(topics=topics, use_onnx=use_onnx),
        ]
        if self._pii:
            self._output_scanners.append(Sensitive(use_onnx=use_onnx))
        logger.info(
            "llm_guard_ready",
            input=[type(s).__name__ for s in self._input_scanners],
            output=[type(s).__name__ for s in self._output_scanners],
            ban_topics=topics,
            fail_open=fail_open,
        )

    # ── Sync core (CPU-bound; callers wrap in asyncio.to_thread) ────────────────
    def _scan_parallel(self, scanners: list, scan_one: Callable[[Any, str], tuple], base_text: str):
        classifiers = [s for s in scanners if type(s).__name__ not in _SANITIZING_SCANNERS]
        sanitizers = [s for s in scanners if type(s).__name__ in _SANITIZING_SCANNERS]
        valid: dict[str, bool] = {}
        scores: dict[str, float] = {}

        def run_classifier(scanner: Any):
            _, is_valid, risk = scan_one(scanner, base_text)
            return type(scanner).__name__, bool(is_valid), float(risk)

        def run_sanitizer_chain():
            sanitized = base_text
            out = []
            for scanner in sanitizers:
                sanitized, is_valid, risk = scan_one(scanner, sanitized)
                out.append((type(scanner).__name__, bool(is_valid), float(risk)))
            return sanitized, out

        sanitized = base_text
        workers = max(1, len(classifiers) + (1 if sanitizers else 0))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            cls_futures = [ex.submit(run_classifier, s) for s in classifiers]
            san_future = ex.submit(run_sanitizer_chain) if sanitizers else None
            for fut in cls_futures:
                name, is_valid, risk = fut.result()
                valid[name], scores[name] = is_valid, risk
            if san_future is not None:
                sanitized, chain = san_future.result()
                for name, is_valid, risk in chain:
                    valid[name], scores[name] = is_valid, risk
        return sanitized, valid, scores

    def _fail_result(self, text: str, stage: str, error: str) -> dict[str, Any]:
        logger.warning("llm_guard_scan_error", stage=stage, error=error)
        if self._fail_open:
            return {"valid": True, "sanitized": text, "findings": [], "scores": {}}
        return {"valid": False, "sanitized": text, "findings": ["scan_error"], "scores": {}}

    def scan_input(self, text: str) -> dict[str, Any]:
        if not text:
            return {"valid": True, "sanitized": text, "findings": [], "scores": {}}
        try:
            if self._parallel:
                sanitized, valid, scores = self._scan_parallel(
                    self._input_scanners, lambda s, t: s.scan(t), text
                )
            else:
                from llm_guard import scan_prompt

                sanitized, valid, scores = scan_prompt(self._input_scanners, text)
        except Exception as e:  # noqa: BLE001 — fail-closed unless fail_open
            return self._fail_result(text, "input", str(e))
        findings = [n for n, ok in valid.items() if not ok and n in _BLOCKING_INPUT]
        return {
            "valid": not findings,
            "sanitized": sanitized,
            "findings": findings,
            "scores": scores,
        }

    def scan_output(self, prompt: str, output: str) -> dict[str, Any]:
        if not output:
            return {"valid": True, "sanitized": output, "findings": [], "scores": {}}
        try:
            if self._parallel:
                sanitized, valid, scores = self._scan_parallel(
                    self._output_scanners, lambda s, t: s.scan(prompt or "", t), output
                )
            else:
                from llm_guard import scan_output as _scan_output

                sanitized, valid, scores = _scan_output(self._output_scanners, prompt or "", output)
        except Exception as e:  # noqa: BLE001
            return self._fail_result(output, "output", str(e))
        findings = [n for n, ok in valid.items() if not ok and n in _BLOCKING_OUTPUT]
        return {
            "valid": not findings,
            "sanitized": sanitized,
            "findings": findings,
            "scores": scores,
        }

    def warm(self) -> None:
        """Run one benign input+output scan so the models' kernels are warm.

        Construction loads the weights, but the FIRST inference still pays a
        cold-kernel penalty (hundreds of ms). Calling this once at startup keeps
        that off the first real request. Best-effort — warming never raises.
        """
        # Use a realistic-length prompt: PyTorch caches kernels per input shape, so
        # warming on a 1-token string leaves the first real (longer) request to
        # re-trace. This sentence covers the common short-prompt shape.
        sample = "What is the weather in Paris and tell me about the latest grid outage report?"
        try:
            self.scan_input(sample)
            self.scan_output(sample, "Here is the weather report and the outage summary you asked for.")
        except Exception as exc:  # noqa: BLE001 — warming is best-effort
            logger.warning("llm_guard_warm_failed", error=str(exc))

    # ── Async wrappers used by the graph nodes ──────────────────────────────────
    async def ascan_input(self, text: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.scan_input, text)

    async def ascan_output(self, prompt: str, output: str) -> dict[str, Any]:
        return await asyncio.to_thread(self.scan_output, prompt, output)


def create_guard(settings: Any) -> "LLMGuard | None":
    """Build the guard when ``enable_guards`` is on.

    Fail-closed: a missing ``llm-guard`` install raises with an install hint rather
    than silently running unprotected. Returns None only when guards are disabled.
    """
    if not getattr(settings, "enable_guards", False):
        return None
    ban = getattr(settings, "guard_ban_topics", None)
    topics = [t.strip() for t in ban.split(",") if t.strip()] if ban else None
    try:
        return LLMGuard(
            ban_topics=topics,
            pii=getattr(settings, "guard_pii", True),
            fail_open=getattr(settings, "guard_fail_open", False),
        )
    except ImportError as exc:  # mandatory but optional-extra: instruct, don't silently skip
        raise RuntimeError(
            "Content guards are enabled (enable_guards=true) but 'llm-guard' is not "
            "installed. Install the extra: `uv sync --extra guards`, or set "
            "GENIE_ENABLE_GUARDS=false to disable."
        ) from exc
