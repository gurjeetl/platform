"""MCP transport / tool loading.

Bundles the MCP config models (previously ``mcpconfig.mcp_config``) so the SDK
is self-contained. Observability hooks are replaced with stdlib logging.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from genie_agent_sdk.permissions import filter_tools_by_permission

_log = logging.getLogger("genie_agent_sdk.mcp")


# --- MCP config models ------------------------------------------------------
class MCPTransport(str, Enum):
    SSE = "sse"
    STDIO = "stdio"
    WEBSOCKET = "websocket"
    STREAMABLE_HTTP = "streamable_http"


@dataclass
class MCPServerConfig:
    name: str
    url: str
    transport: MCPTransport = MCPTransport.SSE
    timeout: float = 30.0
    retries: int = 3
    headers: dict = field(default_factory=dict)
    env: dict = field(default_factory=dict)


@dataclass
class MCPAgentConfig:
    servers: list[MCPServerConfig]
    default_timeout: int = 30
    allowed_roles: list[str] = field(default_factory=list)


# --- Client -----------------------------------------------------------------
class MCPClient:
    """Builds MCP configuration from env, loads tools, and unwraps results."""

    def build_config_from_env(self) -> MCPAgentConfig | None:
        """Construct an MCPAgentConfig from environment variables.

        Reads MCP_SERVER_URL (required), MCP_SERVER_NAME, MCP_TRANSPORT,
        MCP_AUTH_TOKEN, MCP_TIMEOUT.
        """
        url = os.getenv("MCP_SERVER_URL")
        if not url:
            return None

        transport_str = os.getenv("MCP_TRANSPORT", "sse")
        try:
            transport = MCPTransport(transport_str)
        except ValueError:
            _log.warning("mcp.unknown_transport transport=%s fallback=sse", transport_str)
            transport = MCPTransport.SSE

        token = os.getenv("MCP_AUTH_TOKEN", "")
        server = MCPServerConfig(
            name=os.getenv("MCP_SERVER_NAME", "default"),
            url=url,
            transport=transport,
            timeout=float(os.getenv("MCP_TIMEOUT", "30.0")),
            headers={"Authorization": f"Bearer {token}"} if token else {},
        )
        return MCPAgentConfig(servers=[server])

    async def load_tools(
        self,
        config: MCPAgentConfig,
        tool_names: list[str] | None,
    ) -> list[BaseTool]:
        server_map = {
            s.name: {
                "transport": s.transport.value,
                "url": s.url,
                "headers": s.headers,
                "timeout": s.timeout,
            }
            for s in config.servers
        }
        client = MultiServerMCPClient(server_map)
        tools = await client.get_tools()
        if tool_names is not None:
            allowed = set(tool_names)
            tools = [t for t in tools if t.name in allowed]
        return filter_tools_by_permission(tools)

    @staticmethod
    def unwrap_result(result) -> str:
        """MCP tool results may come back as a list of content blocks; flatten to text."""
        if isinstance(result, str):
            return result
        if isinstance(result, list):
            parts: list[str] = []
            for item in result:
                if isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            return " ".join(parts)
        return str(result)
