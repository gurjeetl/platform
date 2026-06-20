"""Agent metadata models — re-exported from the shared ``genie_agent_contracts``.

These were a hand-synced copy of the registry service / SDK shapes. They now
re-export the single shared definition so the three deployables cannot drift.
The module path is kept so existing imports (``from genie.discovery.agent_meta
import AgentMeta``) continue to work.
"""
from __future__ import annotations

from genie_agent_contracts import AgentMeta, FieldSpec, Skill

__all__ = ["AgentMeta", "FieldSpec", "Skill"]
