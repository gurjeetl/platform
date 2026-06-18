"""Catalog of structured event names emitted by SDK agents.

Centralizing the names (rather than scattering ad-hoc strings) keeps log/span
events consistent and greppable across the SDK and any observer that consumes
them. Values are plain strings, used as the ``event`` / ``name`` argument to an
:class:`~genie_agent_sdk.observable.Observable`'s ``log`` / ``log_event``.
"""
from __future__ import annotations


class Events:
    """Event-name constants for agent observability.

    Reference them as ``Events.MCP_TOOL_CALL`` etc.; never type the raw string.
    """

    # Agent run / output lifecycle
    FINAL_OUTPUT_SET = "agent.final_output_set"
    AGENT_ERROR_SET = "agent.error_set"
    AGENT_RUN_FAILED = "agent.run_failed"
    AGENT_SCRATCHPAD = "agent.scratchpad"

    # Message shaping
    FORMAT_MESSAGES = "agent.format_messages"

    # MCP tool lifecycle
    MCP_TOOLS_LOADED = "mcp.tools_loaded"
    MCP_LOAD_FAILED = "mcp.load_failed"
    MCP_TOOL_CALL = "mcp.tool_call"

    # LLM tool loop
    LLM_TOOL_CALLS = "llm.tool_calls"
    LLM_TOOL_RESULTS = "llm.tool_results"
    LLM_INVOKE_FAILED = "llm.invoke_failed"
    TOOL_INVOKE_FAILED = "tool.invoke_failed"

    # A2A peer delegation
    PEER_CALL = "a2a.peer_call"
    PEER_CALL_FAILED = "a2a.peer_call_failed"
