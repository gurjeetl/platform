"""Model tests for the registry — no DB or network required (pydantic only)."""
from registry_service.agent_meta import AgentMeta, FieldSpec, Skill


def test_validate_args_passes_when_required_present():
    meta = AgentMeta(
        agent_id="weather",
        input_schema={
            "location": FieldSpec(type="string", required=True),
            "units": FieldSpec(type="string", required=False),
        },
    )
    ok, err = meta.validate_args({"location": "Paris"})
    assert ok is True
    assert err == ""


def test_validate_args_fails_on_missing_required():
    meta = AgentMeta(
        agent_id="weather",
        input_schema={"location": FieldSpec(type="string", required=True)},
    )
    ok, err = meta.validate_args({})
    assert ok is False
    assert "location" in err

    ok, err = meta.validate_args({"location": ""})
    assert ok is False
    assert "location" in err


def test_skill_auto_derived_when_absent():
    meta = AgentMeta(
        agent_id="weather",
        description="Reports the weather",
        capability_tags=["weather", "forecast"],
        input_schema={
            "location": FieldSpec(type="string", required=True),
            "units": FieldSpec(type="string", required=False),
        },
    )
    assert len(meta.skills) == 1
    skill = meta.skills[0]
    assert skill.id == "weather"
    assert skill.tags == ["weather", "forecast"]
    assert "requires location" in skill.description
    assert "optional units" in skill.description


def test_explicit_skills_preserved():
    explicit = Skill(id="s1", name="Custom", description="hand-written", tags=["x"])
    meta = AgentMeta(agent_id="a", skills=[explicit])
    assert meta.skills == [explicit]


def test_no_inputs_skill_note():
    meta = AgentMeta(agent_id="ping", description="Health check")
    assert "no inputs" in meta.skills[0].description
