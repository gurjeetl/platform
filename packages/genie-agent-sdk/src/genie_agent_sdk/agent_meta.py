"""Agent metadata models — re-exported from the shared ``genie_agent_contracts``.

Previously an intentional copy of the registry shape so the SDK was independently
installable. It now re-exports the single shared definition (a tiny, dependency-free
contracts package) so an agent's ``AgentMeta`` and the registry's stay identical by
construction rather than by hand. The module path is kept so existing imports
(``from genie_agent_sdk.agent_meta import AgentMeta``) continue to work.
"""
from __future__ import annotations

from genie_agent_contracts import AgentMeta, FieldSpec, Skill

__all__ = ["AgentMeta", "FieldSpec", "Skill"]
