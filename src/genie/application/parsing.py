"""Parsing/menu helpers shared by the Planner and the Router.

Both turn the registry's live ``AgentInfo`` list into a prompt menu and both must
tolerantly parse an LLM's JSON and resolve the agent id it picked. Keeping these in
one place makes the Router a *fast mirror* of the Planner's routing logic rather than
a second implementation that can drift.

Ported from BaseAgentFramework ``planner/parsing.py`` and adapted to operate on
Genie's ``genie.agents.base.AgentInfo`` (which carries ``input_schema`` /
``output_schema`` / ``tags`` / ``sla_ms`` discovered from each remote agent).
"""

from __future__ import annotations

import json
import re

from genie.agents.base import AgentInfo


def render_capability_menu(agents: list[AgentInfo]) -> str:
    """Format live agents into the menu both the Planner and Router prompt with."""
    lines = []
    for info in agents:
        inputs = (
            ", ".join(
                f"{name}{'*' if (isinstance(spec, dict) and spec.get('required')) else ''}:"
                f"{spec.get('type', 'any') if isinstance(spec, dict) else 'any'}"
                for name, spec in info.input_schema.items()
            )
            or "(none)"
        )
        outputs = (
            ", ".join(
                f"{name}:{spec.get('type', 'any') if isinstance(spec, dict) else 'any'}"
                + (
                    f" ({spec.get('description')})"
                    if isinstance(spec, dict) and spec.get("description")
                    else ""
                )
                for name, spec in info.output_schema.items()
            )
            or "(none)"
        )
        tags = ", ".join(info.tags or info.capabilities) or "(none)"
        lines.append(
            f'- agent_id: "{info.agent_id}"   (use this exact string; the version below is INFO ONLY, do NOT include it)\n'
            f"    version: {info.version}\n"
            f"    capability: {info.description or '(no description)'}\n"
            f"    tags: {tags}\n"
            f"    inputs: {inputs}\n"
            f"    outputs: {outputs}\n"
            f"    sla_ms: {info.sla_ms}"
        )
    return "\n".join(lines) if lines else "(no agents registered)"


def extract_json(raw: str) -> dict | None:
    """Find the first balanced JSON object in ``raw`` and parse it.

    Tolerant of LLM tics like trailing junk, an extra closing brace, or a markdown
    code fence — we walk the string tracking brace depth and string state so we
    stop exactly at the matching closer of the first object.
    """
    if not raw:
        return None
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = raw[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    try:
        return json.loads(raw[start:])
    except json.JSONDecodeError:
        return None


def normalize_agent_id(raw_id: str | None, known_ids: set[str]) -> str | None:
    """Resolve common LLM stumbles to a real discovered agent id.

    Handles: trailing version (`` v1.0.0``), accidental quotes/whitespace, case
    differences. Returns the canonical agent_id if a match is found, else None.
    """
    if not raw_id or not isinstance(raw_id, str):
        return None
    cleaned = raw_id.strip().strip('"').strip("'").strip()
    cleaned = re.sub(r"[\s@]+v?\d+(?:\.\d+){0,3}\s*$", "", cleaned).strip()
    if cleaned in known_ids:
        return cleaned
    lower_map = {k.lower(): k for k in known_ids}
    return lower_map.get(cleaned.lower())
