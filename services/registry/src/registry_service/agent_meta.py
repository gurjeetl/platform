"""Agent metadata models — re-exported from the shared ``genie_agent_contracts``.

``AgentMeta`` is the payload agents register and consumers discover. It used to be
hand-synced with the SDK and platform copies; it now re-exports the single shared
definition so the shapes cannot drift. The module path is kept so existing imports
(``from registry_service.agent_meta import AgentMeta``) continue to work.
"""
from __future__ import annotations

from genie_agent_contracts import AgentMeta, FieldSpec, Skill

__all__ = ["AgentMeta", "FieldSpec", "Skill"]
