"""Planner/router parsing + menu helpers."""

from genie.agents.base import AgentInfo, CapabilitySpec
from genie.application.parsing import extract_json, normalize_agent_id, render_capability_menu


def test_extract_json_tolerates_fences_and_trailing():
    raw = '```json\n{"route": "fast", "agent_id": "weather"}\n```\nthanks!'
    parsed = extract_json(raw)
    assert parsed == {"route": "fast", "agent_id": "weather"}


def test_extract_json_none_when_absent():
    assert extract_json("no json here") is None


def test_normalize_agent_id_strips_version_and_case():
    known = {"weather", "outage"}
    assert normalize_agent_id('"Weather v1.0.0"', known) == "weather"
    assert normalize_agent_id("outage", known) == "outage"
    assert normalize_agent_id("ghost", known) is None


def test_render_capability_menu_marks_required_inputs():
    info = AgentInfo(
        agent_id="weather",
        name="weather",
        description="city weather",
        version="1.0.0",
        enabled=True,
        capability_specs=[CapabilitySpec(id="weather")],
        input_schema={"location": {"type": "string", "required": True}},
        tags=["weather"],
    )
    menu = render_capability_menu([info])
    assert 'agent_id: "weather"' in menu
    assert "location*:string" in menu


def test_render_capability_menu_empty():
    assert render_capability_menu([]) == "(no agents registered)"
