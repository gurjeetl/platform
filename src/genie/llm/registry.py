"""LLM provider registry — register and resolve named providers."""

from __future__ import annotations

from genie.llm.base import LLMProvider
from genie.platform.errors import ErrorCode, GenieError


class LLMRegistry:
    """Maps provider names to LLMProvider instances."""

    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}

    def register(self, name: str, provider: LLMProvider) -> None:
        """Register *provider* under *name*."""
        self._providers[name] = provider

    def get(self, name: str) -> LLMProvider:
        """Return the provider registered under *name*.

        Raises GenieError(NOT_FOUND) if the name is not registered.
        """
        if name not in self._providers:
            raise GenieError(
                ErrorCode.NOT_FOUND,
                f"LLM provider '{name}' not found. Registered: {list(self._providers)}",
            )
        return self._providers[name]

    def list_providers(self) -> list[str]:
        """Return all registered provider names."""
        return list(self._providers.keys())
