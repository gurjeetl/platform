"""Service discovery: the platform's async client for the registry service.

Resolves live distributed agents (their ``AgentMeta`` + endpoint) so the bootstrap
can surface each as a ``RemoteAgent`` in the in-process registry.
Ported from BaseAgentFramework ``registry/``.
"""

from genie.discovery.agent_meta import AgentMeta, FieldSpec, Skill
from genie.discovery.client import DiscoveryClient, RegistryUnavailable

__all__ = ["AgentMeta", "FieldSpec", "Skill", "DiscoveryClient", "RegistryUnavailable"]
