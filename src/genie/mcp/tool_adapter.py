"""MCPToolAdapter — bridges MCP tool discovery into ConcreteToolGateway."""

from __future__ import annotations

from typing import Any

from genie.mcp.client import MCPClient
from genie.observability.logging import get_logger
from genie.tools.gateway import ConcreteToolGateway

logger = get_logger(__name__)


class MCPToolAdapter:
    """Connects an MCPClient and registers its tools into a ConcreteToolGateway.

    Usage (in bootstrap lifespan):
        adapter = MCPToolAdapter("meter_service", client, gateway)
        await adapter.register()     # connect + discover tools
        ...
        await adapter.unregister()   # disconnect
    """

    def __init__(
        self,
        service_name: str,
        client: MCPClient,
        gateway: ConcreteToolGateway,
    ) -> None:
        self._service_name = service_name
        self._client = client
        self._gateway = gateway
        self._registered_tools: list[str] = []

    @property
    def connected(self) -> bool:
        """True while the underlying MCP client holds a live connection."""
        return self._client.connected

    async def register(self) -> bool:
        """Connect to MCP and register all discovered tools into the gateway.

        Returns True if at least one tool was registered.
        """
        ok = await self._client.connect()
        if not ok:
            logger.warning(
                "mcp_adapter_skipped",
                service=self._service_name,
                reason="connection failed",
            )
            return False

        tools = await self._client.list_tools()
        for tool in tools:
            # Namespace tool ids by service so two servers can expose the same name.
            tool_id = f"{self._service_name}.{tool.name}"
            client = self._client  # capture for closure
            tool_name = tool.name  # capture for closure

            async def _handler(
                params: dict[str, Any],
                _client: MCPClient = client,
                _tool: str = tool_name,
            ) -> Any:
                return await _client.call_tool(_tool, params)

            self._gateway.register(tool_id, _handler)
            self._registered_tools.append(tool_id)
            logger.info(
                "mcp_tool_registered",
                service=self._service_name,
                tool=tool_id,
            )

        logger.info(
            "mcp_adapter_ready",
            service=self._service_name,
            tool_count=len(self._registered_tools),
        )
        return len(self._registered_tools) > 0

    async def unregister(self) -> None:
        """Disconnect and clear registered tools from the gateway."""
        await self._client.disconnect()
        # No public deregister on the gateway, so pop directly from its registry.
        for tool_id in self._registered_tools:
            self._gateway._tools.pop(tool_id, None)
        self._registered_tools.clear()

    @property
    def registered_tools(self) -> list[str]:
        """Namespaced ids of the tools this adapter has registered."""
        return list(self._registered_tools)
