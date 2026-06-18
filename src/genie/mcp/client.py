"""Low-level async MCP client — persistent background-task lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

from genie.observability.logging import get_logger

logger = get_logger(__name__)


class MCPClient:
    """Manages a persistent connection to one MCP server.

    The connection is held alive inside a private asyncio Task so that
    ``streamable_http_client``'s internal anyio cancel scopes are always
    entered *and* exited in the same task — avoiding the
    "Attempted to exit cancel scope in a different task" RuntimeError that
    occurs when the server is unreachable and cleanup crosses task boundaries.

    Usage::

        client = MCPClient("meter_service", "http://host/mcp")
        ok = await client.connect()          # returns False if server is down
        tools = await client.list_tools()
        result = await client.call_tool("get_meter_data", {...})
        await client.disconnect()
    """

    def __init__(
        self, name: str, url: str, transport: str = "streamable_http", connect_timeout: float = 30.0
    ) -> None:
        """Configure (but don't open) a connection to the MCP server at ``url``.
        Events/session/task are created lazily in ``connect`` so they bind to the
        current event loop."""
        self.name = name
        self.url = url
        self.transport = transport
        self.connect_timeout = connect_timeout
        self.connected = False
        self._session: Any = None
        self._task: asyncio.Task[None] | None = None
        self._ready_event: asyncio.Event | None = None
        self._stop_event: asyncio.Event | None = None
        self._connect_error: BaseException | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def connect(self, timeout: float | None = None) -> bool:
        """Establish a persistent connection to the MCP server.

        Starts a background task that holds the ``streamable_http_client``
        context open for the lifetime of the connection.

        Returns True when connected and ``initialize()`` has succeeded.
        Returns False on any connection failure or timeout.
        """
        # Always create fresh events in the current event-loop context.
        self._ready_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._connect_error = None

        self._task = asyncio.create_task(self._run_forever(), name=f"mcp-{self.name}")

        effective_timeout = timeout if timeout is not None else self.connect_timeout
        # Wait for _run_forever to signal readiness (or failure).
        try:
            await asyncio.wait_for(
                asyncio.shield(self._ready_event.wait()),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "mcp_connect_timeout",
                service=self.name,
                timeout=effective_timeout,
            )
            await self._cancel_task()
            return False

        return self.connected

    async def disconnect(self) -> None:
        """Signal the background task to close the connection gracefully."""
        if self._stop_event is not None:
            self._stop_event.set()
        await self._cancel_task(graceful=True)
        logger.info("mcp_disconnected", service=self.name)

    # ── Tool discovery ────────────────────────────────────────────────────────

    async def list_tools(self) -> list[Any]:
        """Tools advertised by the server; empty list when not connected or on error."""
        if not self._session:
            return []
        try:
            result = await self._session.list_tools()
            return result.tools
        except Exception as exc:
            logger.warning("mcp_list_tools_failed", service=self.name, error=str(exc))
            return []

    # ── Tool invocation ───────────────────────────────────────────────────────

    async def call_tool(self, tool_name: str, params: dict[str, Any]) -> Any:
        """Invoke a server tool and normalise its content blocks into one value:
        a single text block as-is, multiple blocks re-assembled into a JSON array
        string, or model-dumped non-text content. Raises if not connected."""
        import json as _json

        if not self._session:
            raise RuntimeError(f"MCP client '{self.name}' is not connected")
        result = await self._session.call_tool(tool_name, params)

        text_blocks = [c.text for c in result.content if hasattr(c, "text") and c.text]

        if not text_blocks:
            # No text content (empty list result or non-text content)
            return [c.model_dump() if hasattr(c, "model_dump") else str(c) for c in result.content]

        if len(text_blocks) == 1:
            # Single block — return as-is (backward compat for plain strings and
            # single-value results; consumers handle JSON-parsing themselves)
            return text_blocks[0]

        # Multiple blocks: FastMCP serialises each list item as a separate
        # TextContent.  Re-assemble into a JSON array string so all consumers
        # get a consistent encoding regardless of result count.
        items = []
        for text in text_blocks:
            try:
                items.append(_json.loads(text))
            except (_json.JSONDecodeError, ValueError):
                items.append(text)
        return _json.dumps(items)

    # ── Background task ───────────────────────────────────────────────────────

    async def _run_forever(self) -> None:
        """Hold the MCP context managers open inside a single asyncio task.

        All anyio cancel scopes created by ``streamable_http_client`` are
        entered and exited here — never crossing into a different task.
        """
        assert self._ready_event is not None
        assert self._stop_event is not None

        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            async with streamable_http_client(self.url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    self._session = session
                    self.connected = True
                    logger.info("mcp_connected", service=self.name, url=self.url)
                    self._ready_event.set()  # unblock connect()

                    # Stay alive until disconnect() signals us.
                    await self._stop_event.wait()

        except Exception as exc:
            self._connect_error = exc
            logger.warning(
                "mcp_connect_failed",
                service=self.name,
                url=self.url,
                error=type(exc).__name__,
                detail=str(exc),
            )
        finally:
            self._session = None
            self.connected = False
            # Always unblock connect() so it doesn't hang.
            if not self._ready_event.is_set():
                self._ready_event.set()

    async def _cancel_task(self, *, graceful: bool = False) -> None:
        """Await the background task to completion, cancelling it unless ``graceful``
        (in which case ``_stop_event`` was already set to let it unwind cleanly)."""
        if self._task is None or self._task.done():
            return
        if not graceful:
            self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            pass
        self._task = None
