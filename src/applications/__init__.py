"""Application agents — in-process plugins injected into the platform.

Each agent implements the platform agent SDK surface (``genie.agents.base``) and is
wired in via ``applications.providers.AGENT_PROVIDERS`` (see ``src/app.py``). These
coexist with distributed agents discovered from the registry (``agent_mode`` =
``local`` uses only these; ``hybrid`` uses both).
"""
