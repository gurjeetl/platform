# This information is not to be used, disseminated, distributed,
#   or otherwise transferred without the expressed written permission
#          of Open Access Technology International, Inc.
#
#          2025 Open Access Technology International, Inc.
#                       All rights reserved.

"""Centralized system-prompt templates for the standalone SDK agents.

Each agent's persona is composed from the shared ``SYSTEM_PROMPT`` and
``SYSTEM_CONTEXT`` blocks via :class:`string.Template`, with task instructions
wrapped in ``[INST] … [/INST]`` markers. The rendered ``*_SYSTEM_PROMPT`` strings
are assigned to each agent's ``system_prompt`` class attribute; ``RAG_USER_PROMPT``
stays a live template because it is filled per request with ``$query``/``$context``.
"""

from string import Template

# ── Shared base pieces ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a helpful, respectful, and accurate assistant developed to support "
    "user requests within an agentic-workflow platform."
)

SYSTEM_CONTEXT = (
    "You operate as a single specialized agent invoked by the platform's planner. "
    "Stay within your specialty and return a concise, useful result."
)


def _persona(instructions: str) -> str:
    """Render the shared persona + a task ``[INST]`` block into a system prompt."""
    return Template(
        "$system_prompt\n\n$system_context\n\n[INST]\n$instructions\n[/INST]"
    ).safe_substitute(
        system_prompt=SYSTEM_PROMPT,
        system_context=SYSTEM_CONTEXT,
        instructions=instructions,
    )


# ── Per-agent system prompts ──────────────────────────────────────────────────
WEATHER_SYSTEM_PROMPT = _persona(
    "Act as a weather reporter for a travel assistant. Report a named city's "
    "current weather clearly and concisely."
)

OUTAGE_SYSTEM_PROMPT = _persona(
    "Act as a grid-outage analyst. Summarize electricity outage reports — either "
    "the top-N outage list or the details of one outage — accurately and concisely."
)

RAG_SYSTEM_PROMPT = _persona(
    "Act as a documentation assistant for an agentic-workflow platform. Answer the "
    "user's question USING ONLY the provided context chunks. Cite the chunks you "
    "use inline as [n]. If the context does not contain the answer, say so plainly "
    "instead of guessing. Be concise and concrete."
)

# Per-request task template for the RAG agent (filled with $query and $context).
RAG_USER_PROMPT = Template(
    """QUESTION:
$query

CONTEXT:
$context

Answer using only the context above, citing sources as [n]."""
)
