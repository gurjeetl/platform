"""MCP (Model Context Protocol) client and tool-gateway adapter."""

from genie.mcp.client import MCPClient
from genie.mcp.tool_adapter import MCPToolAdapter

__all__ = ["MCPClient", "MCPToolAdapter"]
