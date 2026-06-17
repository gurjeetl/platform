"""Application entry point.

This file assembles the in-process application agents into the platform via
``applications.providers.AGENT_PROVIDERS`` (the weather / outage / rag agents under
``src/applications/``). It is the only file that imports application code; the
platform kernel stays agent-agnostic.

Agents can run two ways and coexist:
  * **In-process** — the providers wired here (used in ``agent_mode`` ``local`` /
    ``hybrid``). Add one by creating ``applications/<name>/agent.py`` + a provider.
  * **Distributed** — separate services on ``packages/genie-agent-sdk`` that
    self-register with the registry and are discovered as ``RemoteAgent``s
    (``agent_mode`` ``distributed`` / ``hybrid``).

Start command:
    uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
"""

from __future__ import annotations

from typing import Any

from applications.providers import AGENT_PROVIDERS
from genie.interface.bootstrap import create_app as _create_platform_app
from genie.platform.config import Settings

# ── RAG seed data ─────────────────────────────────────────────────────────────
# Loaded once at startup; survives hot-reloads without a manual seed step.
# In production (RemoteRAGAdapter) this is a no-op — the remote service
# already holds persistent documents.

_RAG_SEED_DOCS = [
    {
        "id": "doc-acsr-guide",
        "title": "ACSR Conductor Types and Specifications",
        "source": "internal/technical",
        "content": (
            "ACSR (Aluminum Conductor Steel Reinforced) Conductor Reference Guide\n\n"
            "ACSR conductors are the most widely used overhead transmission line conductors "
            "in North America. They combine high-strength steel core strands with aluminum "
            "conductor strands to balance mechanical strength and electrical conductivity.\n\n"
            "Common ACSR types and ratings:\n"
            "Drake (795 kcmil, 26/7 strands): rated ampacity 900 A at 75 degrees C, "
            "resistance 0.1172 ohm/mile. Widely used for 115 kV to 345 kV transmission.\n"
            "Cardinal (954 kcmil, 54/7 strands): rated ampacity 1010 A at 75 degrees C, "
            "resistance 0.0969 ohm/mile. Used for high-capacity lines.\n"
            "Pheasant (1272 kcmil, 54/7 strands): rated ampacity 1200 A. "
            "Used for EHV transmission.\n"
            "Hawk (477 kcmil, 26/7 strands): rated ampacity 660 A. "
            "Common for 69 kV subtransmission.\n"
            "Wren (336.4 kcmil, 26/7 strands): rated ampacity 530 A. "
            "Used for 34.5 kV distribution.\n\n"
            "Ampacity ratings are based on: 25 degrees C ambient, 2 ft/s wind, "
            "emissivity 0.5, absorptivity 0.5, per IEEE 738 standard."
        ),
    },
    {
        "id": "doc-line-ratings",
        "title": "Transmission Line Rating Methods: Static vs Dynamic",
        "source": "internal/operations",
        "content": (
            "Transmission Line Rating Methods\n\n"
            "Static Thermal Ratings (STR): Conservative fixed values based on worst-case "
            "ambient conditions — 40 degrees C ambient, zero wind, full solar radiation.\n\n"
            "Dynamic Line Ratings (DLR): Real-time ampacity based on actual weather "
            "conditions. DLR can increase available capacity by 10 to 40 percent. "
            "FERC Order 881 requires utilities to implement ambient-adjusted ratings (AAR) "
            "by July 2025. NERC FAC-008 requires transmission owners to document their "
            "facility ratings methodology.\n\n"
            "Rating update frequency: AAR must update at least every 30 minutes. "
            "Full DLR systems typically update every 5 to 15 minutes."
        ),
    },
    {
        "id": "doc-nerc-tpl",
        "title": "NERC Reliability Standards for Transmission Planning",
        "source": "internal/compliance",
        "content": (
            "NERC Reliability Standards — Transmission Planning Overview\n\n"
            "TPL-001-5 (Transmission System Planning Performance Requirements): "
            "Establishes requirements for transmission planning to ensure BES reliability.\n\n"
            "Planning Events:\n"
            "P0 (No contingency): System must operate within normal ratings, no load shed.\n"
            "P1 (Single contingency): Loss of one element. No cascading outages.\n"
            "P2 (Single contingency, delayed clearing): Loss of element with delayed fault clearing.\n"
            "P3 (Multiple contingency, common structure): Two or more elements on common tower.\n"
            "P4 (Extreme events): Beyond design basis; controlled islanding permitted.\n\n"
            "IROL (Interconnection Reliability Operating Limits): MW transfer limits that, "
            "if violated, could cause cascading failures. Planners must identify IROLs and "
            "ensure operating procedures keep the system within these limits."
        ),
    },
    {
        "id": "doc-eim-operations",
        "title": "Energy Imbalance Market (EIM) Operations Guide",
        "source": "internal/market-operations",
        "content": (
            "Energy Imbalance Market (EIM) Operations Overview\n\n"
            "The Western Energy Imbalance Market (WEIM) operated by CAISO provides "
            "real-time energy balancing across multiple balancing authority areas (BAAs).\n\n"
            "Resource Sufficiency Evaluation (RSE): Each EIM entity must demonstrate "
            "resource sufficiency before each 15-minute dispatch interval. Failure to pass "
            "RSE results in penalty charges.\n\n"
            "15-Minute Market (FMM) and 5-Minute Market (RTD): FMM runs every 15 minutes "
            "to minimize imbalance cost. RTD runs every 5 minutes for fine-tuning dispatch.\n\n"
            "Congestion Management: EIM resolves transmission congestion using locational "
            "marginal prices (LMPs) including energy, congestion, and loss components. "
            "Shadow prices on constrained paths indicate the marginal cost of congestion.\n\n"
            "Settlement: Calculated on a 5-minute basis and netted at end of each operating day."
        ),
    },
    {
        "id": "doc-alarm-management",
        "title": "SCADA and EMS Alarm Management Best Practices",
        "source": "internal/operations",
        "content": (
            "SCADA/EMS Alarm Management Best Practices\n\n"
            "Every alarm must have a defined consequence, required operator action, "
            "response time window, and priority: 1=Critical, 2=High, 3=Medium, 4=Low.\n\n"
            "Key Performance Indicators:\n"
            "Average alarm rate: should not exceed 1 alarm per 10 minutes during normal operations.\n"
            "Flood threshold: more than 10 alarms in 10 minutes constitutes an alarm flood.\n"
            "Stale alarms: active for more than 24 hours without acknowledgment.\n"
            "Chattering alarms: activate/deactivate more than 3 times in 10 minutes.\n\n"
            "ICCP alarms from neighboring control centers must be integrated into the "
            "unified alarm display with the same priority schema. Operator response "
            "procedures must be reviewed annually per NERC PER-005 training standards."
        ),
    },
]


async def _seed_rag(deps: dict[str, Any]) -> None:
    """Ingest sample domain documents into the RAG adapter at startup."""
    adapter = deps.get("rag_adapter")
    if adapter is None:
        return
    if getattr(adapter, "_index", None):
        return
    from genie.observability.logging import get_logger

    log = get_logger(__name__)
    for doc in _RAG_SEED_DOCS:
        await adapter.ingest(
            doc["content"],
            {"document_id": doc["id"], "title": doc["title"], "source": doc["source"]},
        )
    log.info("rag_seeded", document_count=len(_RAG_SEED_DOCS))


# ── Factory ───────────────────────────────────────────────────────────────────


def create_app(settings: Settings | None = None):
    """Application-level FastAPI factory used by uvicorn --factory.

    Injects the in-process application agents (weather / outage / rag). In
    distributed/hybrid mode the bootstrap additionally discovers remote agents.
    """
    return _create_platform_app(
        settings=settings,
        agent_providers=AGENT_PROVIDERS,
        startup_hooks=[_seed_rag],
    )
