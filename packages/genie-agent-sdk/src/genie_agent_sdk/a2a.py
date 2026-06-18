"""Minimal, spec-aligned A2A protocol types + Agent Card helpers.

A pragmatic subset of the Agent2Agent (A2A) protocol — enough for synchronous
JSON-RPC ``message/send`` and Agent Card discovery. Field names mirror the A2A
spec (``kind``, ``parts``, ``role``, ``messageId``, ``jsonrpc``, ``method``,
``params``, ``result``, ``error``) so the wire format is interoperable.

Deliberately omitted (not in scope): Task lifecycle objects, ``message/stream``
(SSE), push notifications, and file parts.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

# --- JSON-RPC method names --------------------------------------------------
METHOD_MESSAGE_SEND = "message/send"

# --- JSON-RPC error codes ---------------------------------------------------
ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_AGENT_EXECUTION = -32001  # custom: the agent ran but returned an error


# --- Message parts ----------------------------------------------------------
class TextPart(BaseModel):
    """A2A text message part — plain-language content."""

    kind: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    """A2A structured message part — carries args (request) or a view (response)."""

    kind: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None


Part = Annotated[Union[TextPart, DataPart], Field(discriminator="kind")]


class Message(BaseModel):
    """An A2A message: an ordered list of parts with a role and free metadata.

    Invocation context (run_id, task_id, blackboard, sla_ms, ...) is carried in
    ``metadata`` and structured args / views in :class:`DataPart`s.
    """

    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: list[Part] = Field(default_factory=list)
    messageId: str
    taskId: str | None = None
    contextId: str | None = None
    metadata: dict[str, Any] | None = None


# --- Agent Card (discovery) -------------------------------------------------
class AgentCapabilities(BaseModel):
    """Optional A2A protocol features. This SDK supports neither (synchronous only)."""

    streaming: bool = False
    pushNotifications: bool = False


class AgentSkill(BaseModel):
    """One capability advertised on the Agent Card (A2A ``AgentSkill``)."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None
    inputModes: list[str] | None = None
    outputModes: list[str] | None = None


class AgentCard(BaseModel):
    """A2A discovery document served at ``/.well-known/agent.json``."""

    name: str
    description: str = ""
    url: str
    version: str = "1.0.0"
    protocolVersion: str = "0.2.5"
    preferredTransport: str = "JSONRPC"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    defaultInputModes: list[str] = Field(default_factory=lambda: ["text", "data"])
    defaultOutputModes: list[str] = Field(default_factory=lambda: ["text", "data"])
    skills: list[AgentSkill] = Field(default_factory=list)


# --- JSON-RPC envelopes -----------------------------------------------------
class JsonRpcError(BaseModel):
    """JSON-RPC 2.0 error object (see the ``ERR_*`` codes above)."""

    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcRequest(BaseModel):
    """JSON-RPC 2.0 request envelope (e.g. ``message/send``)."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    """JSON-RPC 2.0 response envelope: exactly one of ``result`` / ``error`` is set."""

    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


# --- Part / message helpers -------------------------------------------------
def text_part(text: str) -> TextPart:
    """Wrap a string in a :class:`TextPart`."""
    return TextPart(text=text)


def data_part(data: dict[str, Any]) -> DataPart:
    """Wrap a dict in a :class:`DataPart`."""
    return DataPart(data=data)


def get_text(message: Message) -> str:
    """Concatenate the text of every TextPart in the message."""
    chunks = [p.text for p in message.parts if isinstance(p, TextPart)]
    return "\n".join(c for c in chunks if c)


def get_data(message: Message) -> dict[str, Any]:
    """Merge the ``data`` of every DataPart in the message (later parts win)."""
    merged: dict[str, Any] = {}
    for p in message.parts:
        if isinstance(p, DataPart):
            merged.update(p.data)
    return merged


# --- Agent Card derivation --------------------------------------------------
def a2a_url(endpoint: str | None) -> str:
    """The A2A JSON-RPC URL for an agent base endpoint (``/a2a`` by convention)."""
    base = (endpoint or "").rstrip("/")
    return f"{base}/a2a" if base else ""


def to_agent_card(meta) -> AgentCard:
    """Map an :class:`AgentMeta` onto an A2A :class:`AgentCard`.

    The card's ``skills`` are projected directly from ``meta.skills`` so the card
    and the registry record always advertise identical skills.
    """
    skills = [
        AgentSkill(
            id=s.id,
            name=s.name,
            description=s.description,
            tags=list(s.tags),
            examples=s.examples,
        )
        for s in meta.skills
    ]
    return AgentCard(
        name=meta.agent_id,
        description=meta.description or "",
        url=a2a_url(meta.endpoint),
        version=meta.version,
        capabilities=AgentCapabilities(streaming=False, pushNotifications=False),
        skills=skills,
    )
