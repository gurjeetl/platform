"""BaseAgent composition: LLM + MCP + memory, with a tool-calling run loop.

``BaseAgent`` inherits :class:`~genie_agent_sdk.observable.Observable`, so every
agent's ``run()`` is automatically wrapped in an observability span (an MLflow
``AGENT`` span when mlflow is configured, a stdlib-logging timing span otherwise).
mlflow stays an *optional* dependency — an agent built on this SDK still depends
on neither mlflow nor genie at import time.

State shape
-----------
An agent runs over a plain ``dict`` (``AgentState``). The harness seeds it via
:func:`build_task_state`; subclasses read their inputs as top-level keys (the
task's ``args`` are spread in) and the agent records its answer on
``final_output`` / ``view`` / ``error``.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Any, Callable

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from mlflow.entities import SpanType

from genie_agent_sdk.events import Events
from genie_agent_sdk.llm_client import LLMClient
from genie_agent_sdk.mcp_client import MCPClient
from genie_agent_sdk.memory import AgentMemory
from genie_agent_sdk.observable import Observable

load_dotenv()

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


class BaseAgent(Observable):
    """Single composed agent: orchestrates an LLMClient, MCPClient, and AgentMemory.

    Subclasses set ``system_prompt`` and ``tool_names``, then either override
    ``run()`` or call ``answer_with_tool()`` from inside their own ``run()``.

    Inherits :class:`Observable`, so ``run()`` is auto-wrapped in an ``AGENT``
    span for every subclass — no per-agent decoration needed.
    """

    _span_type: str = SpanType.AGENT
    _component_kind: str = "agent"
    _traced_methods: tuple[str, ...] = ("run",)

    system_prompt: str = ""

    # None  → load all permitted MCP tools (default for generic agents).
    # []    → skip MCP connection entirely (hardcoded agents that never call tools).
    # [...] → load only the named tools.
    tool_names: list[str] | None = None

    def __init__(self) -> None:
        """Compose the LLM/MCP/memory clients; load MCP tools if MCP_SERVER_URL is set.

        Passes ``self`` as the observer to both clients so their events flow
        through this agent's :meth:`log` / :meth:`log_event` (inherited from
        :class:`Observable`) and onto the active span.
        """
        self.llm_client = LLMClient(make_chat_model(), observer=self)
        self.mcp_client = MCPClient(observer=self)
        self.memory = AgentMemory()
        self.tools: list[BaseTool] = []
        if os.getenv("MCP_SERVER_URL"):
            self._load_mcp_from_env()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _increment(self, state: AgentState) -> AgentState:
        """Bump the per-run iteration counter."""
        return patch(state, iteration_count=state.get("iteration_count", 0) + 1)

    def append_response(self, state: AgentState, text: str) -> AgentState:
        """Append an assistant message to the state without marking it complete."""
        return patch(state, messages=[AIMessage(content=text)])

    def set_final_output(self, state: AgentState, text: str) -> AgentState:
        """Record a plain-text answer and mark the run complete."""
        self.log_event(Events.FINAL_OUTPUT_SET, length=len(text) if text else 0)
        return patch(
            state,
            final_output=text,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_final_view(self, state: AgentState, text: str, view: dict) -> AgentState:
        """Record an answer plus a structured ``view`` and mark the run complete."""
        self.log_event(
            Events.FINAL_OUTPUT_SET,
            length=len(text) if text else 0,
            view_type=view.get("type") if isinstance(view, dict) else None,
        )
        return patch(
            state,
            final_output=text,
            view=view,
            is_complete=True,
            messages=[AIMessage(content=text)],
        )

    def set_error(self, state: AgentState, msg: str) -> AgentState:
        """Record an error and mark the run complete (no final output)."""
        self.log("error", Events.AGENT_ERROR_SET, agent=type(self).__name__, error=msg)
        return patch(state, error=msg, is_complete=True)

    def _append_trace(self, state: AgentState, **kwargs) -> AgentState:
        """Append a human-readable trace line to short-term memory for debugging."""
        cls_name = type(self).__name__
        self.log_event(f"{cls_name}.trace", **{k: str(v) for k, v in kwargs.items()})
        entry = f"[{cls_name}] " + ", ".join(f"{k}={v}" for k, v in kwargs.items())
        existing = list(state.get("short_term_memory") or [])
        return patch(state, short_term_memory=existing + [entry])

    # ------------------------------------------------------------------
    # LLM / message helpers (used by subclasses with custom run())
    # ------------------------------------------------------------------
    def format_messages(self, state: AgentState) -> list[BaseMessage]:
        """Trim the message window and prepend the system prompt + known facts."""
        raw: list[BaseMessage] = state.get("messages") or []
        trimmed = self.memory.trim(raw)
        self.log_event(
            Events.FORMAT_MESSAGES,
            input_message_count=len(raw),
            trimmed_message_count=len(trimmed),
        )
        facts = state.get("long_term_memory_keys") or []
        return LLMClient.build_messages(
            self.system_prompt, trimmed, self.memory.facts_block(facts)
        )

    def call_llm(self, messages: list[BaseMessage]) -> str:
        """Invoke the LLM and return its text reply."""
        return self.llm_client.call(messages)

    # ------------------------------------------------------------------
    # MCP loading + single-tool invocation
    # ------------------------------------------------------------------
    def _load_mcp_from_env(self) -> None:
        """Connect to the env-configured MCP server and bind the agent's tools."""
        # Empty list means the subclass explicitly opted out — skip the connection.
        if self.tool_names is not None and not self.tool_names:
            return
        config = self.mcp_client.build_config_from_env()
        if not config:
            return
        try:
            self._run_async(self._async_load_mcp_tools(config))
            self.log(
                "info", Events.MCP_TOOLS_LOADED, agent=type(self).__name__, count=len(self.tools)
            )
        except Exception as e:
            self.log(
                "error",
                Events.MCP_LOAD_FAILED,
                agent=type(self).__name__,
                error=str(e),
                exc_info=True,
            )

    async def _async_load_mcp_tools(self, config) -> None:
        """Load the permitted MCP tools and bind them onto the LLM client."""
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
        """Invoke one loaded MCP tool by name and return its unwrapped text result.

        The invocation is wrapped in an MLflow ``TOOL`` span carrying the inputs
        and the (truncated) result. Span operations are guarded so a tracing
        failure never breaks the tool call itself.
        """
        tool = next((t for t in self.tools if t.name == name), None)
        if tool is None:
            raise LookupError(f"MCP tool '{name}' not available")
        with self._tool_span(name, args) as span:
            raw = self._run_async(tool.ainvoke(args))
            report = MCPClient.unwrap_result(raw)
            if span is not None:
                with contextlib.suppress(Exception):
                    span.set_outputs({"result": report})
                    span.set_attribute("mcp.tool", name)
        self.log_event(Events.MCP_TOOL_CALL, tool=name, args=str(args), result=report[:200])
        return report

    @contextlib.contextmanager
    def _tool_span(self, name: str, args: dict):
        """Open a ``TOOL`` span for an MCP call; yield None if the backend is down."""
        try:
            cm = mlflow.start_span(name=f"mcp.{name}", span_type=SpanType.TOOL)
        except Exception:  # pragma: no cover - backend down / no active trace
            yield None
            return
        with cm as span:
            with contextlib.suppress(Exception):
                span.set_inputs({"tool": name, "args": args})
            yield span

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
            self.log(
                "error",
                Events.AGENT_RUN_FAILED,
                agent=type(self).__name__,
                error=str(e),
                exc_info=True,
            )
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
        """Default entry point: one-shot LLM answer, or a tool-calling loop if tools exist.

        Subclasses typically override this (often calling ``answer_with_tool``)
        when they have a fixed single-tool flow.
        """
        state = self._increment(state)
        messages = self.format_messages(state)

        if not self.tools:
            return self.set_final_output(state, self.call_llm(messages))

        return self._run_tool_loop(state, messages)

    def _run_tool_loop(self, state: AgentState, messages: list[BaseMessage]) -> AgentState:
        """Iterate LLM <-> tool calls until a tool-free reply, or max_iterations is hit."""
        max_iters = state.get("max_iterations") or 10
        total_tool_calls = 0

        for iteration in range(max_iters):
            response = self.llm_client.invoke(messages)

            if not response.tool_calls:
                self._log_loop_end(iteration + 1, total_tool_calls, len(messages) + 1)
                return self.set_final_output(state, response.content)

            messages, state = self._step_tools(response, messages, state, iteration + 1)
            total_tool_calls += len(response.tool_calls)

        self._log_loop_end(max_iters, total_tool_calls, exceeded=True)
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
        """Execute the LLM's requested tool calls and append their results to messages."""
        self._log_tool_calls(iteration, response.tool_calls)
        messages.append(response)
        tool_messages = asyncio.run(self.llm_client.execute_tool_calls(response.tool_calls))
        messages.extend(tool_messages)
        self._log_tool_results(iteration, response.tool_calls, tool_messages)
        state = self._append_trace(state, tool_calls=len(response.tool_calls))
        return messages, state

    # ------------------------------------------------------------------
    # Loop logging
    # ------------------------------------------------------------------
    def _log_tool_calls(self, iteration: int, tool_calls: list[dict]) -> None:
        """Emit the LLM's requested tool calls for this iteration."""
        self.log_event(
            Events.LLM_TOOL_CALLS,
            iteration=iteration,
            count=len(tool_calls),
            calls=str([{"name": tc["name"], "args": tc["args"]} for tc in tool_calls]),
        )

    def _log_tool_results(
        self,
        iteration: int,
        tool_calls: list[dict],
        tool_messages: list[ToolMessage],
    ) -> None:
        """Emit the (truncated) results of this iteration's tool calls."""
        self.log_event(
            Events.LLM_TOOL_RESULTS,
            iteration=iteration,
            results=str([
                {"name": tc["name"], "result": str(tm.content)[:200]}
                for tc, tm in zip(tool_calls, tool_messages)
            ]),
        )

    def _log_loop_end(
        self,
        iterations: int,
        total_tool_calls: int,
        final_message_count: int | None = None,
        exceeded: bool = False,
    ) -> None:
        """Emit a summary of how the tool loop terminated."""
        attrs: dict = {"iterations": iterations, "total_tool_calls": total_tool_calls}
        if exceeded:
            attrs["exceeded_max_iters"] = True
        if final_message_count is not None:
            attrs["final_message_count"] = final_message_count
        self.log_event(Events.AGENT_SCRATCHPAD, **attrs)

    # ------------------------------------------------------------------
    # Agent-to-agent (A2A) — discover a peer via the Registry, message it
    # ------------------------------------------------------------------
    def call_peer(
        self,
        agent_id: str,
        args: dict,
        context: dict | None = None,
        *,
        sla_ms: int = 10000,
    ) -> str:
        """Delegate to a peer agent over A2A, discovered through the Registry.

        Returns the peer's text reply. Lets an agent fan work out to another agent
        mid-run (the "agents talk to each other" half of A2A Hybrid) without the
        two ever importing each other — discovery stays centralized in the
        Registry, transport is JSON-RPC ``message/send``.
        """
        from genie_agent_sdk.a2a import get_text
        from genie_agent_sdk.a2a_client import call_agent

        self.log_event(Events.PEER_CALL, peer=agent_id, args=str(args))
        try:
            reply = self._run_async(call_agent(agent_id, args, context or {}, sla_ms=sla_ms))
        except Exception as e:
            self.log("error", Events.PEER_CALL_FAILED, peer=agent_id, error=str(e), exc_info=True)
            raise
        return get_text(reply)
