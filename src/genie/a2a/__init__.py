"""Minimal A2A (Agent2Agent) protocol surface used by the platform.

Synchronous JSON-RPC ``message/send`` over HTTP plus Agent Card types — enough
for the Executor (via ``RemoteAgent``) to dispatch tasks to distributed agents.
Ported from BaseAgentFramework ``a2a/``.
"""

from genie.a2a.agent_card import a2a_url
from genie.a2a.client import A2AClient, A2AError
from genie.a2a.types import (
    METHOD_MESSAGE_SEND,
    DataPart,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    TextPart,
    data_part,
    get_data,
    get_text,
    text_part,
)

__all__ = [
    "A2AClient",
    "A2AError",
    "DataPart",
    "JsonRpcRequest",
    "JsonRpcResponse",
    "METHOD_MESSAGE_SEND",
    "Message",
    "TextPart",
    "a2a_url",
    "data_part",
    "get_data",
    "get_text",
    "text_part",
]
