"""A2A type round-trip tests — pydantic only, no network or langchain."""
from genie_agent_sdk.a2a import (
    METHOD_MESSAGE_SEND,
    DataPart,
    JsonRpcRequest,
    JsonRpcResponse,
    Message,
    TextPart,
    data_part,
    get_data,
    get_text,
    text_part,
)
from genie_agent_sdk.agent_meta import AgentMeta


def _platform_request() -> dict:
    """A request shaped exactly like the platform's A2A client emits."""
    msg = Message(
        role="user",
        messageId="m1",
        taskId="t1",
        contextId="th1",
        parts=[data_part({"args": {"location": "Paris"}})],
        metadata={
            "agent_id": "weather",
            "task_id": "t1",
            "run_id": "r1",
            "thread_id": "th1",
            "blackboard": {"prior": {"x": 1}},
            "sla_ms": 5000,
        },
    )
    return JsonRpcRequest(
        id="t1",
        method=METHOD_MESSAGE_SEND,
        params={"message": msg.model_dump(mode="json")},
    ).model_dump(mode="json")


def test_request_envelope_round_trips():
    raw = _platform_request()
    assert raw["jsonrpc"] == "2.0"
    assert raw["method"] == "message/send"

    in_msg = Message.model_validate(raw["params"]["message"])
    assert in_msg.role == "user"
    assert in_msg.metadata["run_id"] == "r1"
    assert in_msg.metadata["blackboard"] == {"prior": {"x": 1}}

    args = get_data(in_msg).get("args")
    assert args == {"location": "Paris"}


def test_data_part_discriminator():
    raw = _platform_request()
    part = Message.model_validate(raw["params"]["message"]).parts[0]
    assert isinstance(part, DataPart)
    assert part.kind == "data"


def test_response_message_shape():
    out = Message(
        role="agent",
        messageId="m2",
        taskId="t1",
        contextId="th1",
        parts=[text_part("It is sunny."), data_part({"view": {"type": "card"}})],
        metadata={"agent_id": "weather"},
    )
    resp = JsonRpcResponse(id="t1", result=out.model_dump(mode="json")).model_dump(mode="json")

    rebuilt = JsonRpcResponse.model_validate(resp)
    assert rebuilt.error is None
    result_msg = Message.model_validate(rebuilt.result)
    assert result_msg.role == "agent"
    assert get_text(result_msg) == "It is sunny."
    assert get_data(result_msg)["view"] == {"type": "card"}
    # First part is text, second is the data/view part.
    assert isinstance(result_msg.parts[0], TextPart)
    assert isinstance(result_msg.parts[1], DataPart)


def test_error_response():
    resp = JsonRpcResponse.model_validate(
        {"jsonrpc": "2.0", "id": "t1", "error": {"code": -32601, "message": "nope"}}
    )
    assert resp.result is None
    assert resp.error.code == -32601


def test_agent_meta_card_consistency():
    meta = AgentMeta(agent_id="weather", capability_tags=["weather"], endpoint="http://h:9/")
    # Skills auto-derived; card derivation needs no langchain.
    from genie_agent_sdk.a2a import to_agent_card

    card = to_agent_card(meta)
    assert card.name == "weather"
    assert card.url == "http://h:9/a2a"
    assert card.skills[0].id == "weather"
