from langchain_core.messages import BaseMessage


class AgentMemory:
    """Short-term message-window trimming + long-term fact rendering.

    This class only handles the in-prompt shaping of memory: trimming the
    message window and formatting known facts. Fact persistence/retrieval is the
    host application's responsibility (kept out of the SDK on purpose).
    """

    def __init__(self, max_window: int = 15) -> None:
        self.max_window = max_window

    def trim(self, messages: list[BaseMessage]) -> list[BaseMessage]:
        if len(messages) <= self.max_window:
            return messages
        return messages[-self.max_window:]

    @staticmethod
    def facts_block(facts: list[str]) -> str:
        return "\n".join(f"- {f}" for f in facts)
