"""In-process outage agent (Genie application plugin).

Lists grid outages (honoring a ``limit``/top-N), or describes one by ``outage_id``.
Returns a structured ``view`` so the planner can chain a detail lookup off a listed
id (e.g. ``${t1.view.items.0.id}``). Self-contained reference data; a production
version would query an operational DB via MCP / the tool gateway.
"""

from __future__ import annotations

from genie.agents.base import AgentInfo, AgentResult, AgentTask, CapabilitySpec

# Reference dataset (ordered by significance/recency). A production agent would
# pull these from the operational DB; this keeps the example self-contained.
_OUTAGES = [
    {
        "id": 18645677,
        "short_description": "Transmission line fault on 345 kV corridor",
        "outage_type": "transmission",
        "participant": "MISO",
        "status": "resolved",
        "is_significant": True,
    },
    {
        "id": 18553223,
        "short_description": "Substation breaker trip",
        "outage_type": "substation",
        "participant": "PJM",
        "status": "investigating",
        "is_significant": False,
    },
    {
        "id": 17299126,
        "short_description": "Distribution feeder lockout after storm",
        "outage_type": "distribution",
        "participant": "SPP",
        "status": "restored",
        "is_significant": True,
    },
    {
        "id": 18701044,
        "short_description": "Generator forced outage — boiler tube leak",
        "outage_type": "generation",
        "participant": "ERCOT",
        "status": "ongoing",
        "is_significant": True,
    },
    {
        "id": 18699210,
        "short_description": "Capacitor bank failure on 138 kV bus",
        "outage_type": "substation",
        "participant": "MISO",
        "status": "resolved",
        "is_significant": False,
    },
    {
        "id": 18688500,
        "short_description": "Wildfire de-energization (PSPS) on 230 kV line",
        "outage_type": "transmission",
        "participant": "CAISO",
        "status": "restored",
        "is_significant": True,
    },
    {
        "id": 18650913,
        "short_description": "Transformer overheating — load curtailed",
        "outage_type": "substation",
        "participant": "PJM",
        "status": "investigating",
        "is_significant": True,
    },
    {
        "id": 18610777,
        "short_description": "Ice loading conductor galloping — line tripped",
        "outage_type": "transmission",
        "participant": "SPP",
        "status": "resolved",
        "is_significant": False,
    },
]
_BY_ID = {o["id"]: o for o in _OUTAGES}
_DEFAULT_LIMIT = 5


class OutageAgent:
    """In-process outage agent: lists outages (top-N) or describes one by id."""

    agent_id = "outage"
    name = "outage"
    description = (
        "Lists grid outages, or describes one by outage_id. Call with no args (or a "
        "limit) for the top-N outage list — pass limit=N for 'top N'; call with "
        "outage_id for a structured detail view of one outage."
    )
    capabilities = ["outage"]
    version = "1.0.0"
    enabled = True

    def enable(self) -> None:
        """Mark this agent as available for routing."""
        self.enabled = True

    def disable(self) -> None:
        """Mark this agent as unavailable for routing."""
        self.enabled = False

    async def health_check(self) -> str:
        """Report liveness; this static agent is always ``healthy``."""
        return "healthy"

    def get_info(self) -> AgentInfo:
        """Return the agent's capability/schema descriptor for discovery + routing."""
        return AgentInfo(
            agent_id=self.agent_id,
            name=self.name,
            description=self.description,
            version=self.version,
            enabled=self.enabled,
            capability_specs=[
                CapabilitySpec(
                    id="outage",
                    display_name="Grid outages",
                    description=self.description,
                    routing_keywords=[
                        "outage",
                        "outages",
                        "grid",
                        "power",
                        "report",
                        "list",
                        "top",
                    ],
                )
            ],
            input_schema={
                "outage_id": {
                    "type": "integer",
                    "required": False,
                    "description": "Specific outage id; omit for the top list",
                },
                "limit": {
                    "type": "integer",
                    "required": False,
                    "description": "How many to list for 'top N' (default 5)",
                },
            },
            output_schema={
                "text": {"type": "string", "persist": True},
                "view": {"type": "object", "persist": True},
            },
            tags=["outage", "outages", "grid", "power"],
            sla_ms=6000,
        )

    async def execute(self, task: AgentTask) -> AgentResult:
        """Detail path if ``args.outage_id`` is given, else the top-N list path."""
        args = (task.context or {}).get("args", {})
        oid = args.get("outage_id")
        if oid is not None:
            rec = _BY_ID.get(int(oid))
            if rec is None:
                return AgentResult(
                    task_id=task.task_id,
                    agent_id=self.agent_id,
                    success=True,
                    output=f"No outage found with id {oid}.",
                )
            return AgentResult(
                task_id=task.task_id,
                agent_id=self.agent_id,
                success=True,
                output=f"Outage {rec['id']}: {rec['short_description']} (status: {rec['status']}).",
                data={"view": {"type": "outage_detail", "outage_id": rec["id"], "metadata": rec}},
            )

        # List path — honor a requested top-N (clamped to what we have).
        try:
            limit = int(args.get("limit") or _DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = _DEFAULT_LIMIT
        limit = max(1, min(limit, len(_OUTAGES)))
        items = _OUTAGES[:limit]
        return AgentResult(
            task_id=task.task_id,
            agent_id=self.agent_id,
            success=True,
            output=f"Top {len(items)} outages (of {len(_OUTAGES)} total).",
            data={"view": {"type": "outage_list", "total": len(_OUTAGES), "items": items}},
        )
