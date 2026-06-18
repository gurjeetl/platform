"""Standalone MCP tool server for the genie-platform example agents.

Exposes static travel/outage data and a self-contained documentation search
behind the FastMCP tool surface. Ported from the reference framework's
``weather_server.py`` — the only behavioral change is that ``search_docs`` is
backed by a small bundled in-memory doc corpus (see ``DOCS`` below) with a tiny
dependency-free BM25 ranker, so this server is fully self-contained.

Run:      python mcp_weather_server.py
Endpoint: http://127.0.0.1:2002/mcp   (transport: streamable_http)

The SDK's MCP client connects via MCP_SERVER_URL / MCP_TRANSPORT. Point the
agents at MCP_SERVER_URL=http://127.0.0.1:2002/mcp with MCP_TRANSPORT=streamable_http.
"""
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

WEATHER_DATA: dict[str, str] = {
    "london": "Cloudy, 14°C, light rain expected",
    "paris": "Sunny, 22°C, clear skies",
    "new york": "Partly cloudy, 18°C, mild winds",
    "tokyo": "Humid, 28°C, chance of thunderstorm",
    "dubai": "Hot and sunny, 41°C, no cloud cover",
    "minneapolis": "warm and humid, 28-30°C, small chance of showers or thunderstorms",
    "bloomington": "warm and mostly cloudy, 28-30°C, scattered thunderstorms or showers possible",
}

# OUTAGE_DATA_PATH points at the Data.Json copied into this directory so the
# outage tools are self-contained. Override with the OUTAGE_DATA_PATH env var.
OUTAGE_DATA_PATH = Path(
    os.getenv("OUTAGE_DATA_PATH") or (Path(__file__).resolve().parent / "Data.Json")
)
_OUTAGE_CACHE: dict[str, Any] | None = None
_OUTAGE_INDEX: dict[int, dict[str, Any]] = {}


def _load_outage_data() -> dict[str, Any]:
    """Load + cache the outage report JSON, indexing each outage by id (lazily, once)."""
    global _OUTAGE_CACHE
    if _OUTAGE_CACHE is None:
        with OUTAGE_DATA_PATH.open(encoding="utf-8") as f:
            _OUTAGE_CACHE = json.load(f)
        for item in _OUTAGE_CACHE.get("outagewise_analysis", []):
            _OUTAGE_INDEX[int(item["id"])] = item
    return _OUTAGE_CACHE


# --- Self-contained documentation corpus backing search_docs ----------------
# A handful of short markdown-ish strings describing the platform so the RAG
# agent returns something meaningful without depending on an external docs tree.
DOCS: list[dict[str, str]] = [
    {
        "source": "docs/router.md#Router",
        "text": (
            "The Router is the front door of the platform. It inspects each incoming "
            "user request and decides whether to fast-path it to a single specialist "
            "agent (for example the RAG agent for 'what is X' questions) or to hand it "
            "to the Planner for multi-step decomposition. Routing uses capability tags "
            "and keyword overlap against the agents discovered in the Registry."
        ),
    },
    {
        "source": "docs/planner.md#Planner",
        "text": (
            "The Planner decomposes a complex user goal into a directed plan of agent "
            "tasks. Each task names a capability and its arguments; tasks can reference "
            "the outputs of earlier tasks by path (e.g. ${<id>.view.items.0.id}) so the "
            "Orchestrator can chain them. The Planner only plans against capabilities "
            "that the Registry reports as live."
        ),
    },
    {
        "source": "docs/orchestrator.md#Orchestrator",
        "text": (
            "The Orchestrator executes the Planner's plan. It resolves task arguments "
            "(including cross-task references), invokes each agent over the A2A protocol, "
            "writes every result onto the shared Blackboard, and respects per-task SLAs. "
            "Independent tasks run concurrently; dependent tasks wait for their inputs."
        ),
    },
    {
        "source": "docs/blackboard.md#Blackboard",
        "text": (
            "The Blackboard is the shared run state for a single request. Agents read "
            "upstream results from it and write their own structured outputs (text and "
            "view) back to it. It is how the Orchestrator passes data between agents and "
            "how the Synthesizer assembles the final answer."
        ),
    },
    {
        "source": "docs/synthesizer.md#Synthesizer",
        "text": (
            "The Synthesizer reads the completed Blackboard and composes the final "
            "user-facing answer, merging the text and structured views produced by the "
            "individual agents into one coherent response with citations where available."
        ),
    },
    {
        "source": "docs/gate.md#Gate",
        "text": (
            "The Gate is the safety and policy checkpoint. It guards inputs and outputs "
            "against disallowed content and enforces permissions before a result is "
            "returned to the user, acting as a guardrail around the agent workflow."
        ),
    },
    {
        "source": "docs/a2a.md#A2A",
        "text": (
            "A2A (Agent-to-Agent) is the wire protocol the platform uses to invoke "
            "remote agents. It is JSON-RPC 2.0 over HTTP: the caller POSTs message/send "
            "to an agent's /a2a endpoint with arguments carried in a DataPart and run "
            "context (task_id, run_id, thread_id, blackboard, sla_ms) in the message "
            "metadata. The agent replies with an agent-role Message containing a TextPart "
            "and an optional DataPart with a structured view."
        ),
    },
    {
        "source": "docs/registry.md#Registry",
        "text": (
            "The Registry (discovery service) tracks which agents are live. Agents built "
            "on the genie-agent-sdk self-register their AgentMeta on startup, heartbeat on "
            "an interval so the Registry keeps them under a TTL, and deregister on "
            "shutdown. The Router and Planner query the Registry to discover available "
            "capabilities and the endpoints to reach them."
        ),
    },
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_K1, _B = 1.5, 0.75


def _tokenize(text: str) -> list[str]:
    """Lowercase + split text into alphanumeric tokens for BM25 indexing."""
    return _TOKEN_RE.findall((text or "").lower())


class _DocIndex:
    """Tiny in-memory BM25 index over the bundled DOCS corpus."""

    def __init__(self, docs: list[dict[str, str]]) -> None:
        """Precompute per-doc tokens/term-frequencies/lengths and corpus stats (N,
        avgdl, document frequency) used by the BM25 scoring in ``search``."""
        self.docs = docs
        self._tokens = [_tokenize(d["text"]) for d in docs]
        self._tf = [Counter(t) for t in self._tokens]
        self._len = [len(t) for t in self._tokens]
        self.N = len(docs)
        self._avgdl = (sum(self._len) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for toks in self._tokens:
            for term in set(toks):
                df[term] += 1
        self._df = df

    def _idf(self, term: str) -> float:
        """BM25 inverse document frequency for a term (0 if it appears nowhere)."""
        df = self._df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 4) -> list[dict]:
        """BM25-score every doc against the query; return the top-k with source + score."""
        q_terms = _tokenize(query)
        if not q_terms or self.N == 0:
            return []
        scored: list[tuple[float, int]] = []
        for i in range(self.N):
            tf, dl = self._tf[i], self._len[i] or 1
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if not f:
                    continue
                denom = f + _K1 * (1 - _B + _B * dl / (self._avgdl or 1))
                score += self._idf(term) * (f * (_K1 + 1)) / denom
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: s[0], reverse=True)
        return [
            {
                "source": self.docs[i]["source"],
                "text": self.docs[i]["text"],
                "score": round(score, 4),
            }
            for score, i in scored[:k]
        ]


_DOC_INDEX = _DocIndex(DOCS)


mcp = FastMCP("weather-server", host="127.0.0.1", port=2002)


@mcp.tool()
def get_weather(city: str) -> str:
    """Return the current weather report for the given city.

    Supported cities (case-insensitive): london, paris, new york, tokyo, dubai.
    Returns a short human-readable weather summary, or a not-found message.
    """
    key = (city or "").strip().lower()
    report = WEATHER_DATA.get(key)
    if report is None:
        return f"No weather data available for '{city}'."
    return report


@mcp.tool()
def get_outage_report_summary() -> dict[str, Any]:
    """Return top-level summary of the outage analysis report.

    Includes report id/name, time period, status, and aggregate counts
    (total outages, significant outages, report inconsistencies, keywords).
    """
    data = _load_outage_data()
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "status": data.get("status"),
        "created_dt": data.get("created_dt"),
        "time_period": data.get("time_period"),
        "total_outages": data.get("total_outages"),
        "total_significant_outages": data.get("total_significant_outages"),
        "total_report_inconsistencies": data.get("total_report_inconsistencies"),
        "total_keywords_detected": data.get("total_keywords_detected"),
        "linked_outages_count": len(data.get("linked_outages_detected", [])),
    }


TOP_OUTAGES_LIMIT = 5


@mcp.tool()
def list_outage_ids() -> dict[str, Any]:
    """Return the top 5 outages from the report with short descriptions."""
    data = _load_outage_data()
    items = data.get("outagewise_analysis", [])
    top = [
        {
            "id": item.get("id"),
            "short_description": item.get("metadata", {}).get("short_description"),
            "outage_type": item.get("metadata", {}).get("outage_type"),
            "participant": item.get("metadata", {}).get("participant"),
            "status": item.get("metadata", {}).get("status"),
            "is_significant": item.get("analysis", {}).get("is_significant"),
        }
        for item in items[:TOP_OUTAGES_LIMIT]
    ]
    return {"total": len(items), "returned": len(top), "items": top}


@mcp.tool()
def get_outage_metadata(outage_id: int) -> dict[str, Any]:
    """Return the metadata block for a specific outage id."""
    _load_outage_data()
    item = _OUTAGE_INDEX.get(int(outage_id))
    if item is None:
        return {"error": f"No outage found with id {outage_id}."}
    return item.get("metadata", {})


@mcp.tool()
def get_outage_analysis_summary(outage_id: int) -> dict[str, Any]:
    """Return the analysis summary for a specific outage id.

    Includes total inconsistencies, the markdown summary, significance flag,
    and the list of attributewise inconsistencies (without the full
    long-form analysis blob).
    """
    _load_outage_data()
    item = _OUTAGE_INDEX.get(int(outage_id))
    if item is None:
        return {"error": f"No outage found with id {outage_id}."}
    analysis = item.get("analysis", {})
    return {
        "id": item.get("id"),
        "total_outage_inconsistencies": analysis.get("total_outage_inconsistencies"),
        "is_significant": analysis.get("is_significant"),
        "summary": analysis.get("summary"),
        "critical_criteria": analysis.get("critical_criteria"),
        "attributewise_inconsistencies": [
            {
                "attribute": a.get("attribute"),
                "is_inconsistent": a.get("is_inconsistent"),
            }
            for a in analysis.get("attributewise_analysis", [])
        ],
    }


@mcp.tool()
def get_outage_attribute_analysis(outage_id: int, attribute: str) -> dict[str, Any]:
    """Return the full analysis text for one attribute of a specific outage.

    Use list_outage_ids and get_outage_analysis_summary first to discover
    which attributes have inconsistencies.
    """
    _load_outage_data()
    item = _OUTAGE_INDEX.get(int(outage_id))
    if item is None:
        return {"error": f"No outage found with id {outage_id}."}
    target = (attribute or "").strip().lower()
    for a in item.get("analysis", {}).get("attributewise_analysis", []):
        if (a.get("attribute") or "").lower() == target:
            return a
    return {"error": f"No attribute '{attribute}' found for outage {outage_id}."}


@mcp.tool()
def get_linked_outages() -> list[dict[str, Any]]:
    """Return the list of linked-outage detections from the report."""
    data = _load_outage_data()
    return data.get("linked_outages_detected", [])


@mcp.tool()
def search_docs(query: str, k: int = 4) -> dict[str, Any]:
    """Retrieve the top-k most relevant documentation chunks for a query.

    Backs the RAG agent: searches a small bundled corpus describing this
    platform (router, planner, orchestrator, blackboard, synthesizer, gate,
    A2A, registry) with a compact BM25 ranking and returns the matching chunks
    with their source path and relevance score. Use to answer 'what is', 'how
    does', 'explain', 'why' questions about the system.
    """
    chunks = _DOC_INDEX.search(query or "", k=max(1, min(int(k or 4), 10)))
    return {"query": query, "returned": len(chunks), "chunks": chunks}


if __name__ == "__main__":
    # streamable_http matches the SDK's MCP client (MCP_TRANSPORT=streamable_http,
    # MCP_SERVER_URL=http://127.0.0.1:2002/mcp). FastMCP also supports "sse" if needed.
    mcp.run(transport="streamable_http")
