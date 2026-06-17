"""BaseAgent composition: LLM + MCP + memory, with a tool-calling run loop.

Trimmed from the framework's BaseAgent: the mlflow tracing and the
``observability.Observable`` base class are removed and replaced with stdlib
logging, so an agent built on this SDK depends on neither mlflow nor genie.

State shape
-----------
An agent runs over a plain ``dict`` (``AgentState``). The harness seeds it via
:func:`build_task_state`; subclasses read their inputs as top-level keys (the
task's ``args`` are spread in) and the agent records its answer on
``final_output`` / ``view`` / ``error``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from genie_agent_sdk.llm_client import LLMClient
from genie_agent_sdk.mcp_client import MCPClient
from genie_agent_sdk.memory import AgentMemory

load_dotenv()

_log = logging.getLogger("genie_agent_sdk.agent")

# AgentState is an open dict: a known set of control keys plus arbitrary args
# spread in by the harness. Kept as a plain alias so the SDK carries no rigid
# platform-specific TypedDict.
AgentState = dict[str, Any]


def patch(state: AgentState, **changes) -> AgentState:
    """Return a new state with the given keys overwritten."""
    return {**state, **changes}


def build_task_state(
    *,
    task_id: str = "",
    agent_id: str = "",
    args: dict | None = None,
    thread_id: str = "",
    run_id: str = "",
    blackboard: dict | None = None,
    max_iterations: int = 5,
) -> AgentState:
    """Seed a clean per-task state with identifiers + the task args spread in."""
    state: AgentState = {
        "user_input": "",
        "current_task": task_id,
        "thread_id": thread_id,
        "run_id": run_id,
        "messages": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "tool_calls": [],
        "tool_results": [],
        "short_term_memory": [],
        "long_term_memory_keys": [],
        "active_agent": agent_id,
        "blackboard": blackboard or {},
        "final_output": None,
        "view": None,
        "is_complete": False,
        "error": None,
    }
    for k, v in (args or {}).items():
        state[k] = v
    return state


def make_chat_model(model: str | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI from env, with an optional per-component model override.

    Honors OPENAI_MODEL / OPENAI_API_KEY / OPENAI_BASE_URL, plus an optional
    OPENAI_TEMPERATURE (set to 0 for deterministic calls).
    """
    kwargs: dict = {}
    temperature = os.getenv("OPENAI_TEMPERATURE")
    if temperature not in (None, ""):
        kwargs["temperature"] = float(temperature)
    return ChatOpenAI(
        model=model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        **kwargs,
    )


class BaseAgent:
    """Single composed agent: orchestrates an LLMClient, MCPClient, and AgentMemory.

    Subclasses set ``system_prompt`` and ``tool_names``, then either override
    ``run()`` or call ``answer_with_tool()`` from inside their own ``run()``.
    """

    system_prompt: str = ""

    # None  → load all permitted MCP tools (default for generic agents).
    # []    → skip MCP connection entirely (hardcoded agents that never call tools).
    # [...] → load only the named tools.
    tool_names: list[str] | None = None

    def __init__(self) -> None:
        self.llm_client = LLMClient(make_chat_model())
        self.mcp_client = MCPClient()
        self.memory = AgentMemory()
        self.tools: list[BaseTool] = []
        if os.getenv("MCP_SERVER_URL"):
            self._load_mcp_from_env()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _increment(self, state: AgentState) -> AgentState:
        return patch(state, iteration_count=state.get("iteration_count", 0) + 1)

    def append_response(self, state: AgentState, text: str) -> AgentState:
        return patch(state, messages=[AIMessage(content=text)])

    def set_final_output(self, state: AgentState, text: str) -> AgentState:
        return patch(
            state,
            final_output=text,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_final_view(self, state: AgentState, text: str, view: dict) -> AgentState:
        return patch(
            state,
            final_output=text,
            view=view,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_error(self, state: AgentState, msg: str) -> AgentState:
        _log.error("agent.error_set agent=%s error=%s", type(self).__name__, msg)
        return patch(state, error=msg, is_complete=True)

    def _append_trace(self, state: AgentState, **kwargs) -> AgentState:
        cls_name = type(self).__name__
        entry = f"[{cls_name}] " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        existing = list(state.get("short_term_memory") or [])
        return patch(state, short_term_memory=existing + [entry])

    # ------------------------------------------------------------------
    # LLM / message helpers (used by subclasses with custom run())
    # ------------------------------------------------------------------
    def format_messages(self, state: AgentState) -> list[BaseMessage]:
        raw: list[BaseMessage] = state.get("messages") or []
        trimmed = self.memory.trim(raw)
        facts = state.get("long_term_memory_keys") or []
        return LLMClient.build_messages(
            self.system_prompt, trimmed, self.memory.facts_block(facts)
        )

    def call_llm(self, messages: list[BaseMessage]) -> str:
        return self.llm_client.call(messages)

    # ------------------------------------------------------------------
    # MCP loading + single-tool invocation
    # ------------------------------------------------------------------
    def _load_mcp_from_env(self) -> None:
        # Empty list means the subclass explicitly opted out — skip the connection.
        if self.tool_names is not None and not self.tool_names:
            return
        config = self.mcp_client.build_config_from_env()
        if not config:
            return
        try:
            self._run_async(self._async_load_mcp_tools(config))
            _log.info("mcp.tools_loaded agent=%s count=%s", type(self).__name__, len(self.tools))
        except Exception as e:
            _log.error("mcp.load_failed agent=%s error=%s", type(self).__name__, e)

    async def _async_load_mcp_tools(self, config) -> None:
        self.tools = await self.mcp_client.load_tools(config, self.tool_names)
        self.llm_client.bind_tools(self.tools)

    @staticmethod
    def _run_async(coro):
        """Run a coroutine from sync code, creating a loop if none is running."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        # Already inside a loop — run on a fresh one in this thread.
        return loop.run_until_complete(coro)

    def call_mcp_tool(self, name: str, args: dict) -> str:
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            raise LookupError(f"MCP tool '{name}' not available")
        raw = self._run_async(tool.ainvoke(args))
        report = MCPClient.unwrap_result(raw)
        _log.debug("mcp.tool_call tool=%s result=%s", name, report[:200])
        return report

    def answer_with(
        self,
        state: AgentState,
        work: Callable[[], "str | tuple[str, dict | None]"],
        **trace_kwargs,
    ) -> AgentState:
        """Run a unit of agent work and capture its outcome on state.

        ``work`` is a zero-arg callable returning either a plain text reply or a
        ``(text, view)`` tuple where ``view`` is a structured dict for a renderer.
        """
        updated = self._increment(state)
        try:
            result = work()
        except LookupError as e:
            return self.set_error(updated, str(e))
        except Exception as e:
            _log.error("agent.run_failed agent=%s error=%s", type(self).__name__, e)
            return self.set_error(updated, str(e))

        if isinstance(result, tuple):
            text, view = result
        else:
            text, view = result, None

        updated = self._append_trace(updated, **trace_kwargs)
        if view is not None:
            return self.set_final_view(updated, text, view)
        return self.set_final_output(updated, text)

    def answer_with_tool(
        self,
        state: AgentState,
        tool_name: str,
        args: dict,
        format_text: Callable[[str], str],
        **trace_kwargs,
    ) -> AgentState:
        """Shorthand for the single-MCP-tool case: call one tool, format its result."""
        def work() -> str:
            return format_text(self.call_mcp_tool(tool_name, args))

        return self.answer_with(state, work, source=f"mcp:{tool_name}", **trace_kwargs)

    # ------------------------------------------------------------------
    # Main agent loop
    # ------------------------------------------------------------------
    def run(self, state: AgentState) -> AgentState:
        state = self._increment(state)
        messages = self.format_messages(state)

        if not self.tools:
            return self.set_final_output(state, self.call_llm(messages))

        return self._run_tool_loop(state, messages)

    def _run_tool_loop(self, state: AgentState, messages: list[BaseMessage]) -> AgentState:
        max_iters = state.get("max_iterations") or 10
        total_tool_calls = 0

        for iteration in range(max_iters):
            response = self.llm_client.invoke(messages)

            if not response.tool_calls:
                return self.set_final_output(state, response.content)

            messages, state = self._step_tools(response, messages, state, iteration + 1)
            total_tool_calls += len(response.tool_calls)

        return self.set_error(
            state, f"exceeded max_iterations ({max_iters}) without a final answer"
        )

    def _step_tools(
        self,
        response: AIMessage,
        messages: list[BaseMessage],
        state: AgentState,
        iteration: int,
    ) -> tuple[list[BaseMessage], AgentState]:
        messages.append(response)
        tool_messages = asyncio.run(self.llm_client.execute_tool_calls(response.tool_calls))
        messages.extend(tool_messages)
        state = self._append_trace(state, tool_calls=len(response.tool_calls))
        return messages, state
