"""Standalone outage agent service (genie-agent-sdk).

Lists grid outages or describes one by id, sourcing data from the MCP outage
tools. Returns a structured ``view`` so the planner can chain a detail lookup off
a listed id. Runs as an independent A2A service that self-registers with the
Registry (see ``serve_agent`` at the bottom).
"""

import json

from genie_agent_sdk import AgentMeta, BaseAgent, FieldSpec, serve_agent
from prompts import OUTAGE_SYSTEM_PROMPT


class OutageAgent(BaseAgent):
    """SDK agent that lists/describes grid outages via the MCP outage tools."""

    system_prompt = OUTAGE_SYSTEM_PROMPT
    tool_names: list[str] = [
        "list_outage_ids",
        "get_outage_metadata",
        "get_outage_analysis_summary",
    ]

    @staticmethod
    def _parse_json(s: str) -> dict:
        """Parse an MCP tool's JSON string result; return ``{}`` on bad/empty input."""
        try:
            return json.loads(s)
        except (json.JSONDecodeError, TypeError):
            return {}

    def _list_view(self) -> tuple[str, dict] | str:
        """Build the top-N outage-list (text, view), or a message if empty."""
        data = self._parse_json(self.call_mcp_tool("list_outage_ids", {}))
        items = data.get("items", [])
        total = data.get("total")
        if not items:
            return "No outages found in the current report."
        text = f"Top {len(items)} outages (of {total} total)."
        view = {"type": "outage_list", "total": total, "items": items}
        return text, view

    def _detail_view(self, outage_id: int) -> tuple[str, dict] | str:
        """Build the (text, view) detail for one outage, or an error message."""
        metadata = self._parse_json(
            self.call_mcp_tool("get_outage_metadata", {"outage_id": outage_id})
        )
        if metadata.get("error"):
            return f"Could not find outage {outage_id}: {metadata['error']}"

        analysis = self._parse_json(
            self.call_mcp_tool("get_outage_analysis_summary", {"outage_id": outage_id})
        )
        if analysis.get("error"):
            return f"Could not load analysis for outage {outage_id}: {analysis['error']}"

        text = f"Outage {outage_id}: {metadata.get('short_description') or '(no description)'}"
        view = {
            "type": "outage_detail",
            "outage_id": outage_id,
            "metadata": metadata,
            "analysis": analysis,
        }
        return text, view

    def run(self, state: dict) -> dict:
        """Detail path when ``state.outage_id`` is set, else the top-N list path."""
        outage_id = state.get("outage_id")
        if outage_id is not None:
            oid = int(outage_id)
            return self.answer_with(
                state, lambda: self._detail_view(oid),
                source="mcp:outage_detail", outage_id=oid,
            )
        return self.answer_with(state, self._list_view, source="mcp:outage_list")


META = AgentMeta(
    agent_id="outage",
    version="1.0.0",
    capability_tags=[
        "outage", "outages", "grid", "power", "report",
        "list", "top", "summary", "outage_detail", "outage_list",
    ],
    description=(
        "Lists or describes grid outages. Call with no args to get the top-N outage "
        "list (covers 'show me outages', 'top 5 outages', 'recent outages'). "
        "Call with outage_id to get a structured detail view for one specific outage."
    ),
    input_schema={
        "outage_id": FieldSpec(
            type="integer",
            required=False,
            description="Specific outage ID. Omit to get the top-N list.",
        ),
    },
    output_schema={
        "text": FieldSpec(type="string", description="Short headline for the result.", persist=True),
        "view": FieldSpec(
            type="object",
            persist=True,
            description=(
                "outage_list = {total, items:[{id, short_description, outage_type, participant, status, is_significant}]}; "
                "outage_detail = {outage_id, metadata, analysis}. "
                "To chain, reference a field by path, e.g. ${<id>.view.items.0.id} = first listed outage's id."
            ),
        ),
    },
    sla_ms=6000,
)


if __name__ == "__main__":
    # Run this agent as an independent service that self-registers with the
    # Registry Service and exposes the A2A endpoint POST /a2a.
    # Default advisory port: AGENT_PORT=2011 (host/port/registry come from env).
    serve_agent(OutageAgent(), agent_meta=META)
