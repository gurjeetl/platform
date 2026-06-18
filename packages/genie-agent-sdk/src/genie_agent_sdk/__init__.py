"""Genie Agent SDK — a self-registering A2A agent harness.

Build an agent by subclassing :class:`BaseAgent`, declaring an :class:`AgentMeta`,
and running it with :func:`serve_agent`.

The pydantic-only surface (A2A types + AgentMeta + the Events catalog) is
imported eagerly. The harness surface (BaseAgent / serve_agent / Observable /
the A2A send client) is imported lazily via module ``__getattr__`` so that
``import genie_agent_sdk.a2a`` works even when the heavier deps (langchain,
mlflow, ...) are not yet imported.
"""

from genie_agent_sdk.a2a import (
    AgentCard,
    AgentSkill,
    DataPart,
    JsonRpcError,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    TextPart,
    a2a_url,
    data_part,
    get_data,
    get_text,
    text_part,
    to_agent_card,
)
from genie_agent_sdk.agent_meta import AgentMeta, FieldSpec, Skill
from genie_agent_sdk.events import Events

_LAZY = {
    "BaseAgent": ("genie_agent_sdk.base_agent", "BaseAgent"),
    "build_task_state": ("genie_agent_sdk.base_agent", "build_task_state"),
    "make_chat_model": ("genie_agent_sdk.base_agent", "make_chat_model"),
    "serve_agent": ("genie_agent_sdk.server", "serve_agent"),
    "build_agent_app": ("genie_agent_sdk.server", "build_agent_app"),
    "AgentServer": ("genie_agent_sdk.server", "AgentServer"),
    "Observable": ("genie_agent_sdk.observable", "Observable"),
    "A2AClient": ("genie_agent_sdk.a2a_client", "A2AClient"),
    "call_agent": ("genie_agent_sdk.a2a_client", "call_agent"),
    "resolve_endpoint": ("genie_agent_sdk.a2a_client", "resolve_endpoint"),
    "A2AError": ("genie_agent_sdk.a2a_client", "A2AError"),
}


def __getattr__(name: str):
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(target[0])
    return getattr(module, target[1])


__all__ = [
    # Harness (lazy)
    "BaseAgent",
    "serve_agent",
    "build_agent_app",
    "AgentServer",
    "build_task_state",
    "make_chat_model",
    # A2A send client (lazy)
    "A2AClient",
    "call_agent",
    "resolve_endpoint",
    "A2AError",
    # Observability
    "Observable",
    "Events",
    # Metadata
    "AgentMeta",
    "FieldSpec",
    "Skill",
    # A2A types
    "Message",
    "TextPart",
    "DataPart",
    "AgentCard",
    "AgentSkill",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "JsonRpcError",
    "text_part",
    "data_part",
    "get_text",
    "get_data",
    "a2a_url",
    "to_agent_card",
]
