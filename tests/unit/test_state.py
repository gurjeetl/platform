"""GraphState carries the DAG-pipeline fields with safe defaults."""

from genie.application.state import GraphState


def test_defaults():
    s = GraphState()
    assert s.route is None
    assert s.plan is None
    assert s.blackboard == {}
    assert s.replan_count == 0
    assert s.partial is False
    assert s.is_complete is False
    assert s.run_id  # auto-generated


def test_roundtrip_dump_load():
    s = GraphState(
        conversation_id="c",
        route="fast",
        waves=[["t1"]],
        blackboard={"t1": {"agent_id": "weather", "text": "ok"}},
    )
    restored = GraphState(**s.model_dump())
    assert restored.route == "fast"
    assert restored.waves == [["t1"]]
    assert restored.blackboard["t1"]["text"] == "ok"
