"""In-prompt memory shaping: message-window trimming + fact rendering.

Persistence/retrieval of long-term facts is intentionally left to the host app;
this module only shapes what goes into the prompt.
"""
from langchain_core.messages import BaseMessage


class AgentMemory:
    """Short-term message-window trimming + long-term fact rendering.

    This class only handles the in-prompt shaping of memory: trimming the
    message window and formatting known facts. Fact persistence/retrieval is the
    host application's responsibility (kept out of the SDK on purpose).
    """

    def __init__(self, max_window: int = 15) -> None:
        """Set the max number of recent messages kept in the prompt window."""
        self.max_window = max_window

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        """Return at most the last ``max_window`` messages."""
        if len(messages) <= self.max_window:
            return messages
        return messages[-self.max_window:]

    @staticmethod
    def facts_block(facts: list[str]) -> str:
        """Render known facts as a markdown bullet list for the system prompt."""
        return "\n".join(f"- {f}" for f in facts)
