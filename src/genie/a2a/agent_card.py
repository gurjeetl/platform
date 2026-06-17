"""A2A Agent Card helpers."""

from __future__ import annotations


def a2a_url(endpoint: str) -> str:
    """The JSON-RPC A2A URL for an agent given its advertised base endpoint.

    By convention the A2A handler is served at ``/a2a`` on the agent's endpoint.
    """
    return endpoint.rstrip("/") + "/a2a"
