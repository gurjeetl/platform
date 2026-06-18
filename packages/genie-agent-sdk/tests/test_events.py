"""Events catalog sanity checks — constants exist and are unique strings."""
from genie_agent_sdk.events import Events

_EXPECTED = [
    "FINAL_OUTPUT_SET",
    "AGENT_ERROR_SET",
    "AGENT_RUN_FAILED",
    "AGENT_SCRATCHPAD",
    "FORMAT_MESSAGES",
    "MCP_TOOLS_LOADED",
    "MCP_LOAD_FAILED",
    "MCP_TOOL_CALL",
    "LLM_TOOL_CALLS",
    "LLM_TOOL_RESULTS",
    "LLM_INVOKE_FAILED",
    "TOOL_INVOKE_FAILED",
    "PEER_CALL",
    "PEER_CALL_FAILED",
]


def test_all_expected_events_present_and_strings():
    for name in _EXPECTED:
        value = getattr(Events, name)
        assert isinstance(value, str) and value


def test_event_values_are_unique():
    values = [getattr(Events, n) for n in _EXPECTED]
    assert len(values) == len(set(values))
