"""LangGraph checkpointer helpers for HITL and conversation persistence."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver


def create_checkpointer() -> MemorySaver:
    """Create and return a new MemorySaver checkpointer instance."""
    return MemorySaver()


def get_thread_config(conversation_id: str) -> dict:
    """Return a LangGraph thread config dict for the given conversation_id."""
    return {"configurable": {"thread_id": conversation_id}}
