"""Agent metadata models.

Duplicated intentionally from the registry service so the SDK is independently
installable (it does NOT import from the registry deployable or from genie). The
shape must stay identical to ``registry_service.agent_meta`` so an agent's
``AgentMeta`` deserializes cleanly on the registry side.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class FieldSpec(BaseModel):
    """One field in an agent's input or output schema."""

    type: Literal["string", "integer", "number", "boolean", "object", "array"] = "string"
    required: bool = False
    description: str = ""
    persist: bool = False  # Synthesizer commits this field downstream when True


class Skill(BaseModel):
    """One capability an agent advertises, aligned with the A2A ``AgentSkill``."""

    id: str
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    examples: list[str] | None = None


class AgentMeta(BaseModel):
    """Registry record for one agent."""

    agent_id: str
    version: str = "1.0.0"
    capability_tags: list[str] = Field(default_factory=list)
    description: str = ""
    input_schema: dict[str, FieldSpec] = Field(default_factory=dict)
    output_schema: dict[str, FieldSpec] = Field(default_factory=dict)
    sla_ms: int = 10000
    transport: Literal["json-rpc", "kafka", "both"] = "json-rpc"
    status: Literal["active", "deprecated"] = "active"
    changelog_url: str | None = None

    skills: list[Skill] = Field(default_factory=list)

    endpoint: str | None = None
    instance_id: str | None = None
    last_heartbeat: datetime | None = None
    registered_at: datetime | None = None

    @model_validator(mode="after")
    def _ensure_skills(self) -> "AgentMeta":
        """Guarantee at least one skill (derived from tags + I/O) so the Agent Card
        and registry record always advertise a skill."""
        if not self.skills:
            self.skills = [self._derived_skill()]
        return self

    def _derived_skill(self) -> Skill:
        """A single skill summarizing this agent: tags + an input-shape note."""
        required = [name for name, spec in self.input_schema.items() if spec.required]
        optional = [name for name, spec in self.input_schema.items() if not spec.required]
        parts = []
        if required:
            parts.append("requires " + ", ".join(required))
        if optional:
            parts.append("optional " + ", ".join(optional))
        io_note = "; ".join(parts) if parts else "no inputs"
        return Skill(
            id=self.agent_id,
            name=self.agent_id,
            description=f"{self.description or '(no description)'} ({io_note})".strip(),
            tags=list(self.capability_tags),
        )

    def validate_args(self, args: dict) -> tuple[bool, str]:
        """Lightweight required-field check. Type coercion is intentionally lenient."""
        for name, spec in self.input_schema.items():
            if spec.required and (name not in args or args[name] in (None, "")):
                return False, f"missing required input '{name}'"
        return True, ""
