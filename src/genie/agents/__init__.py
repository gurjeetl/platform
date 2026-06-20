"""Agent framework — public API: protocols, types, and registry.

Concrete agent implementations (MeterDataAgent, RulesEngineAgent) are NOT
exported here. The composition root (genie.interface.bootstrap) imports them
directly to keep the application layer free of implementation coupling.
"""

from genie.agents.base import (
    AgentCapability,
    AgentInfo,
    AgentProtocol,
    AgentResult,
    AgentTask,
    CapabilitySpec,
)
from genie.agents.registry import AgentRegistry

__all__ = [
    "AgentCapability",
    "AgentTask",
    "AgentResult",
    "AgentInfo",
    "AgentProtocol",
    "AgentRegistry",
    "CapabilitySpec",
]
