"""Minimal, spec-aligned A2A protocol types (ported from BaseAgentFramework).

A pragmatic subset of the Agent2Agent (A2A) protocol — enough for synchronous
JSON-RPC ``message/send`` and Agent Card discovery. Field names mirror the A2A
spec so the wire format is interoperable with the distributed agent SDK.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

METHOD_MESSAGE_SEND = "message/send"

ERR_PARSE = -32700
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_INTERNAL = -32603
ERR_AGENT_EXECUTION = -32001


class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: dict[str, Any] | None = None


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] | None = None


Part = Annotated[Union[TextPart, DataPart], Field(discriminator="kind")]


class Message(BaseModel):
    kind: Literal["message"] = "message"
    role: Literal["user", "agent"]
    parts: list[Part] = Field(default_factory=list)
    messageId: str
    taskId: str | None = None
    contextId: str | None = None
    metadata: dict[str, Any] | None = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: dict[str, Any] | None = None


class JsonRpcRequest(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcResponse(BaseModel):
    jsonrpc: Literal["2.0"] = "2.0"
    id: str | int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None


def text_part(text: str) -> TextPart:
    return TextPart(text=text)


def data_part(data: dict[str, Any]) -> DataPart:
    return DataPart(data=data)


def get_text(message: Message) -> str:
    chunks = [p.text for p in message.parts if isinstance(p, TextPart)]
    return "\n".join(c for c in chunks if c)


def get_data(message: Message) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for p in message.parts:
        if isinstance(p, DataPart):
            merged.update(p.data)
    return merged
