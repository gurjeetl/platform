"""In-process RAG agent (Genie application plugin).

Answers questions about the platform from a small bundled documentation set using a
dependency-free keyword-overlap ranker, returning a grounded answer with a cited
``view``. A production version would call the extracted RAG service or an MCP
``search_docs`` tool; this keeps the reference agent self-contained.
"""

from __future__ import annotations

from typing import Any

from genie.agents.base import AgentInfo, AgentResult, AgentTask, CapabilitySpec

# Small bundled doc set describing the platform.
_DOCS = [
    (
        "docs/a2a",
        "A2A is the JSON-RPC message/send protocol that agents are invoked over. "
        "The platform resolves an agent's endpoint via the registry and POSTs to /a2a.",
    ),
    (
        "docs/registry",
        "The registry is a discovery service: agents self-register and heartbeat; "
        "the platform discovers them and surfaces each as a RemoteAgent.",
    ),
    (
        "docs/planner",
        "The planner turns the user prompt into a DAG of subtasks over the discovered "
        "agent capability menu; the orchestrator groups it into dependency waves.",
    ),
    (
        "docs/blackboard",
        "The executor runs each wave concurrently and writes results to a shared "
        "blackboard; the completion gate re-plans on missing or errored tasks.",
    ),
    (
        "docs/synthesizer",
        "The synthesizer merges the blackboard into one answer; the input and "
        "output guards bracket the pipeline with content scanning.",
    ),
]


def _rank(query: str, k: int = 3) -> list[dict]:
    q = set(query.lower().split())
    scored = []
    for source, text in _DOCS:
        overlap = len(q & set(text.lower().split()))
        if overlap:
            scored.append({"source": source, "text": text, "score": overlap})
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:k]


class RagAgent:
    agent_id = "rag"
    name = "rag"
    description = "Answers questions about the platform from its documentation (RAG)."
    capabilities = ["rag"]
    version = "1.0.0"
    enabled = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    async def health_check(self) -> str:
        return "healthy"

    def get_info(self) -> AgentInfo:
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            version=self.version,
            enabled=self.enabled,
            capability_specs=[
                CapabilitySpec(
                    id="rag",
                    display_name="Docs Q&A",
                    description=self.description,
                    routing_keywords=[
                        "docs",
                        "documentation",
                        "explain",
                        "what",
                        "how",
                        "why",
                        "a2a",
                        "architecture",
                        "registry",
                        "planner",
                    ],
                )
            ],
            input_schema={
                "query": {"type": "string", "required": True, "description": "The question"}
            },
            output_schema={"text": {"type": "string", "persist": True}, "view": {"type": "object"}},
            tags=["docs", "documentation", "rag", "knowledge"],
            sla_ms=8000,
        )

    async def execute(self, task: AgentTask, context: dict[str, Any]) -> AgentResult:
        query = str((task.context or {}).get("args", {}).get("query", "")).strip()
        chunks = _rank(query)
        if not chunks:
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                output="I couldn't find anything about that in the platform docs.",
            )
        cited = " ".join(f"{c['text']} [{i + 1}]" for i, c in enumerate(chunks))
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            output=f"From the platform docs: {cited}",
            data={
                "view": {
                    "type": "rag",
                    "query": query,
                    "sources": [
                        {"n": i + 1, "source": c["source"], "score": c["score"]}
                        for i, c in enumerate(chunks)
                    ],
                }
            },
        )
