"""Platform-wide error types and HTTP response helpers."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ErrorCode(str, Enum):
    """Stable machine-readable error codes; mapped to HTTP status in bootstrap."""

    INTERNAL_ERROR = "INTERNAL_ERROR"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    LLM_ERROR = "LLM_ERROR"
    RAG_ERROR = "RAG_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    TIMEOUT = "TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    PROMPT_INJECTION = "PROMPT_INJECTION"


class GenieError(Exception):
    """Base exception for all Genie platform errors."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict | None = None,
    ) -> None:
        """Build an error from a code, human message, and optional detail dict."""
        super().__init__(message)
        self._code = code
        self._message = message
        self._details: dict = details or {}

    @property
    def code(self) -> ErrorCode:
        """The machine-readable error code."""
        return self._code

    @property
    def message(self) -> str:
        """The human-readable error message."""
        return self._message

    @property
    def details(self) -> dict:
        """Structured extra context (may be empty)."""
        return self._details

    def to_dict(self) -> dict:
        """Serialise the error to a plain dict for logging or JSON responses."""
        return {
            "code": self._code.value,
            "message": self._message,
            "details": self._details,
        }

    def __repr__(self) -> str:
        """Developer-friendly representation showing code and message."""
        return f"GenieError(code={self._code!r}, message={self._message!r})"


class ErrorResponse(BaseModel):
    """Pydantic model used for HTTP error responses."""

    code: str
    message: str
    details: dict = {}
    correlation_id: str = ""


def error_response(exc: GenieError, correlation_id: str = "") -> ErrorResponse:
    """Convert a GenieError into an HTTP-safe ErrorResponse."""
    return ErrorResponse(
        code=exc.code.value,
        message=exc.message,
        details=exc.details,
        correlation_id=correlation_id,
    )
