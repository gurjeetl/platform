"""Correlation ID propagation via Python contextvars."""
from __future__ import annotations

import uuid
from contextvars import ContextVar, Token

from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

# Module-level ContextVar — one per async task / request
_correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


def set_correlation_id(cid: str) -> Token:
    """Set the correlation ID for the current context and return the reset token."""
    return _correlation_id_var.set(cid)


def get_correlation_id() -> str:
    """Return the correlation ID for the current context (empty string if unset)."""
    return _correlation_id_var.get()


def new_correlation_id() -> str:
    """Generate a new UUID4 correlation ID, set it in context, and return it."""
    cid = str(uuid.uuid4())
    _correlation_id_var.set(cid)
    return cid


class CorrelationMiddleware:
    """ASGI middleware that propagates X-Correlation-ID through the request context."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            headers = Headers(scope=scope)
            cid = headers.get("X-Correlation-ID") or new_correlation_id()
            set_correlation_id(cid)

            async def send_with_correlation(message: dict) -> None:
                if message["type"] == "http.response.start":
                    # Inject the correlation ID into response headers
                    existing_headers = list(message.get("headers", []))
                    existing_headers.append(
                        (b"x-correlation-id", cid.encode("latin-1"))
                    )
                    message = {**message, "headers": existing_headers}
                await send(message)

            await self.app(scope, receive, send_with_correlation)
        else:
            await self.app(scope, receive, send)
