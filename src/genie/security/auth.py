"""API-key authentication and prompt-injection detection."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

from genie.platform.errors import ErrorCode, GenieError

# ── Prompt injection patterns ─────────────────────────────────────────────────
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"disregard\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+[a-z]+\s*(ai|model|assistant)?", re.I),
    re.compile(r"act\s+as\s+(if\s+you\s+were|a)\s+", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"</?[a-z]+[^>]{0,200}>", re.I),  # HTML/XML injection
    re.compile(r"system:\s*you\s+are", re.I),
]


def check_prompt_injection(text: str) -> None:
    """Raise GenieError(PROMPT_INJECTION) if *text* contains injection patterns."""
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            raise GenieError(
                ErrorCode.PROMPT_INJECTION,
                "Potential prompt injection detected in input.",
                {"pattern": pattern.pattern},
            )


def verify_api_key(provided: str | None, expected: str | None) -> bool:
    """Constant-time comparison of API keys. Returns True if valid or if no key configured."""
    if expected is None:
        return True
    if provided is None:
        return False
    return hmac.compare_digest(
        hashlib.sha256(provided.encode()).digest(),
        hashlib.sha256(expected.encode()).digest(),
    )


def require_api_key(provided: str | None, expected: str | None) -> None:
    """Raise GenieError(UNAUTHORIZED) if the provided key is invalid."""
    if not verify_api_key(provided, expected):
        raise GenieError(ErrorCode.UNAUTHORIZED, "Invalid or missing API key.")


def sanitize_user_input(text: str, max_length: int = 8_192) -> str:
    """Strip leading/trailing whitespace, enforce max length, check injection."""
    text = text.strip()
    if len(text) > max_length:
        text = text[:max_length]
    check_prompt_injection(text)
    return text


class ApiKeyMiddleware:
    """ASGI middleware that enforces Bearer API-key authentication."""

    # Probe / docs endpoints stay open so health checks and the OpenAPI UI work.
    EXEMPT_PATHS: frozenset[str] = frozenset({"/health", "/metrics", "/docs", "/openapi.json"})

    def __init__(self, app: Any, api_key: str | None) -> None:
        self._app = app
        self._api_key = api_key

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """Reject HTTP requests with a missing/invalid Bearer key (401); pass
        through when no key is configured, the path is exempt, or auth succeeds."""
        if scope["type"] == "http" and self._api_key:
            path: str = scope.get("path", "")
            if path not in self.EXEMPT_PATHS:
                headers = dict(scope.get("headers", []))
                auth_header = headers.get(b"authorization", b"").decode()
                token = auth_header.removeprefix("Bearer ").strip()
                if not verify_api_key(token or None, self._api_key):
                    body = b'{"code":"UNAUTHORIZED","message":"Invalid or missing API key."}'
                    await send(
                        {
                            "type": "http.response.start",
                            "status": 401,
                            "headers": [
                                [b"content-type", b"application/json"],
                                [b"content-length", str(len(body)).encode()],
                            ],
                        }
                    )
                    await send({"type": "http.response.body", "body": body})
                    return
        await self._app(scope, receive, send)
