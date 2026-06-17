"""ChatOpenAI wrapper + tool execution.

Trimmed from the framework's LLMClient: the observability hooks are replaced
with a thin stdlib-logging observer so the SDK does not depend on mlflow or any
``observability`` package.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

_log = logging.getLogger("genie_agent_sdk.llm")


class Observer(Protocol):
    def log(self, level: str, event: str, **attrs) -> None: ...
    def log_event(self, name: str, **attrs) -> None: ...


class _StdlibObserver:
    """Default no-op-ish observer: routes events to stdlib logging."""

    def log(self, level: str, event: str, **attrs) -> None:
        _log.log(getattr(logging, level.upper(), logging.INFO), "%s %s", event, attrs)

    def log_event(self, name: str, **attrs) -> None:
        _log.debug("%s %s", name, attrs)


class LLMClient:
    """Owns the ChatOpenAI handle, message construction, and tool execution."""

    def __init__(self, llm: ChatOpenAI, observer: Observer | None = None) -> None:
        self.llm = llm
        self.tools: list[BaseTool] = []
        self._observer = observer or _StdlibObserver()

    def bind_tools(self, tools: list[BaseTool]) -> None:
        self.tools = tools
        if tools:
            self.llm = self.llm.bind_tools(tools)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        try:
            return self.llm.invoke(messages)
        except Exception as e:
            self._observer.log("error", "llm.invoke_failed", error=str(e))
            raise

    def call(self, messages: list[BaseMessage]) -> str:
        return self.invoke(messages).content

    async def execute_tool_calls(self, tool_calls: list[dict]) -> list[ToolMessage]:
        tool_map = {t.name: t for t in self.tools}

        async def _call_one(tc: dict) -> ToolMessage:
            tool = tool_map.get(tc["name"])
            if tool is None:
                return ToolMessage(
                    content=f"Tool '{tc['name']}' not found.",
                    tool_call_id=tc["id"],
                )
            try:
                result = await tool.ainvoke(tc["args"])
                return ToolMessage(content=str(result), tool_call_id=tc["id"])
            except Exception as e:
                self._observer.log("error", "tool.invoke_failed", tool=tc["name"], error=str(e))
                return ToolMessage(
                    content=f"Error calling '{tc['name']}': {e}",
                    tool_call_id=tc["id"],
                )

        return list(await asyncio.gather(*(_call_one(tc) for tc in tool_calls)))

    @staticmethod
    def build_messages(
        system_prompt: str,
        trimmed: list[BaseMessage],
        facts_block: str,
    ) -> list[BaseMessage]:
        prompt = system_prompt
        if facts_block:
            prompt = f"{prompt}\n\n## Known context about this user:\n{facts_block}"

        lc_messages: list[BaseMessage] = []
        if prompt:
            lc_messages.append(SystemMessage(content=prompt))
        for msg in trimmed:
            if isinstance(msg, (HumanMessage, AIMessage)):
                lc_messages.append(msg)
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                lc_messages.append(
                    HumanMessage(content=content) if role == "user" else AIMessage(content=content)
                )
        return lc_messages
