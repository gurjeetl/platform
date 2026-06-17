"""Seed the RAG index with sample power-grid domain documents.

Run once after starting the platform:
    python scripts/seed_rag.py

Or point at a different host:
    python scripts/seed_rag.py --base-url http://staging-host:8081

The LocalRAGAdapter is in-memory — run this script after every restart.
In production (RemoteRAGAdapter) documents persist in the RAG service and
only need to be ingested once.
"""
from __future__ import annotations

import argparse
import sys

import httpx

# ── Sample documents (power-grid / utility domain) ────────────────────────────

SAMPLE_DOCUMENTS = [
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
            "- Drake (795 kcmil, 26/7 strands): rated ampacity ~900 A at 75°C, "
            "resistance 0.1172 ohm/mile. Widely used for 115 kV to 345 kV transmission.\n"
            "- Cardinal (954 kcmil, 54/7 strands): rated ampacity ~1010 A at 75°C, "
            "resistance 0.0969 ohm/mile. Used for high-capacity lines.\n"
            "- Pheasant (1272 kcmil, 54/7 strands): rated ampacity ~1200 A. "
            "Used for EHV transmission.\n"
            "- Hawk (477 kcmil, 26/7 strands): rated ampacity ~660 A. "
            "Common for 69 kV subtransmission.\n"
            "- Wren (336.4 kcmil, 26/7 strands): rated ampacity ~530 A. "
            "Used for 34.5 kV distribution.\n\n"
            "Sag and tension calculations must account for thermal expansion, ice loading "
            "(NESC heavy loading), and wind loading. Drake conductor has a coefficient of "
            "thermal expansion of 12.8 × 10⁻⁶/°C.\n\n"
            "Ampacity ratings are based on: 25°C ambient, 2 ft/s wind, emissivity 0.5, "
            "absorptivity 0.5, per IEEE 738 standard."
        ),
    },
    {
        "id": "doc-line-ratings",
        "title": "Transmission Line Rating Methods: Static vs Dynamic",
        "source": "internal/operations",
        "content": (
            "Transmission Line Rating Methods\n\n"
            "Static Thermal Ratings (STR):\n"
            "Static ratings are conservative fixed values assigned to transmission lines "
            "based on worst-case ambient conditions. Typical assumptions: 40°C ambient, "
            "zero wind, full solar radiation. STRs are the default operating limit used "
            "by most utilities and ISOs/RTOs.\n\n"
            "Dynamic Line Ratings (DLR):\n"
            "Dynamic ratings calculate real-time ampacity based on actual weather "
            "conditions measured by sensors on or near the transmission line. DLR systems "
            "use weather sensors (wind speed, direction, ambient temperature, solar "
            "radiation), PMUs for real-time conductor temperature estimation, and SCADA "
            "integration for automatic rating updates.\n\n"
            "Benefits of DLR: Can increase available capacity by 10–40% on average, "
            "reduces renewable energy curtailment, improves grid utilization without "
            "capital investment. FERC Order 881 requires utilities to implement "
            "ambient-adjusted ratings (AAR) as a minimum starting point by July 2025.\n\n"
            "NERC FAC-008 requires transmission owners to document their facility ratings "
            "methodology and ensure ratings are based on the most limiting element.\n\n"
            "Rating update frequency: AAR must update at least every 30 minutes. "
            "Full DLR systems typically update every 5–15 minutes."
        ),
    },
    {
        "id": "doc-nerc-tpl",
        "title": "NERC Reliability Standards for Transmission Planning",
        "source": "internal/compliance",
        "content": (
            "NERC Reliability Standards — Transmission Planning Overview\n\n"
            "TPL-001-5 (Transmission System Planning Performance Requirements):\n"
            "This standard establishes requirements for transmission planning to ensure "
            "reliability of the bulk electric system (BES).\n\n"
            "Planning Events:\n"
            "P0 (No contingency): System must operate within normal ratings, no load shed.\n"
            "P1 (Single contingency): Loss of one element. No cascading outages.\n"
            "P2 (Single contingency, delayed clearing): Loss of element with delayed "
            "fault clearing.\n"
            "P3 (Multiple contingency, common structure): Two or more elements on common "
            "tower.\n"
            "P4 (Extreme events): Beyond design basis; controlled islanding permitted.\n\n"
            "IROL (Interconnection Reliability Operating Limits):\n"
            "IROLs are MW transfer limits that, if violated, could cause cascading failures "
            "and large-scale blackouts. Planners must identify IROLs and ensure operating "
            "procedures keep the system within these limits.\n\n"
            "FAC-002-3 (Facility Connection Requirements): Establishes requirements for "
            "generator and transmission facility interconnection studies.\n\n"
            "TPL-007-4 (GMD Events): Requires utilities to assess geomagnetic disturbance "
            "vulnerability and develop transformer protection mitigation plans."
        ),
    },
    {
        "id": "doc-eim-operations",
        "title": "Energy Imbalance Market (EIM) Operations Guide",
        "source": "internal/market-operations",
        "content": (
            "Energy Imbalance Market (EIM) Operations Overview\n\n"
            "The Western Energy Imbalance Market (WEIM) operated by CAISO provides a "
            "real-time energy balancing mechanism across multiple balancing authority "
            "areas (BAAs) in the Western Interconnection.\n\n"
            "Resource Sufficiency Evaluation (RSE):\n"
            "Each EIM entity must demonstrate resource sufficiency before each 15-minute "
            "dispatch interval. The RSE checks that an entity has enough resources to meet "
            "its load plus a reserve margin. Failure to pass RSE results in penalty charges.\n\n"
            "15-Minute Market (FMM) and 5-Minute Market (RTD):\n"
            "FMM runs every 15 minutes to minimize imbalance cost. RTD runs every 5 minutes "
            "for fine-tuning dispatch.\n\n"
            "Congestion Management: EIM automatically identifies and resolves transmission "
            "congestion across BAA boundaries using locational marginal prices (LMPs). "
            "LMPs include an energy component, congestion component, and loss component. "
            "Shadow prices on constrained transmission paths indicate the marginal cost "
            "of congestion.\n\n"
            "Settlement: EIM settlements are calculated on a 5-minute basis and netted "
            "at the end of each operating day."
        ),
    },
    {
        "id": "doc-alarm-management",
        "title": "SCADA and EMS Alarm Management Best Practices",
        "source": "internal/operations",
        "content": (
            "SCADA/EMS Alarm Management Best Practices\n\n"
            "Alarm Philosophy: Effective alarm management is critical for grid reliability. "
            "NERC PRC-004 and EEMUA Publication 191 define industry best practices.\n\n"
            "Alarm Rationalization: Every alarm must have a defined consequence if not "
            "responded to, a required operator action, a time window for response, and "
            "a priority classification: 1=Critical, 2=High, 3=Medium, 4=Low.\n\n"
            "Key Performance Indicators:\n"
            "- Average alarm rate: should not exceed 1 alarm per 10 minutes during normal "
            "operations.\n"
            "- Flood threshold: more than 10 alarms in 10 minutes constitutes an alarm "
            "flood.\n"
            "- Stale alarms: alarms active for more than 24 hours without acknowledgment.\n"
            "- Chattering alarms: activate/deactivate more than 3 times in 10 minutes.\n\n"
            "Console configuration: limit unacknowledged alarms displayed (max 20 "
            "recommended). Color coding: Red=Critical, Orange=High, Yellow=Medium, "
            "Gray=Acknowledged.\n\n"
            "ICCP alarms from neighboring control centers must be integrated into the "
            "unified alarm display with the same priority schema.\n\n"
            "Operator response procedures must be reviewed annually. Simulator training "
            "on alarm response is required per NERC PER-005 training standards."
        ),
    },
]


# ── HTTP client ───────────────────────────────────────────────────────────────

def seed(base_url: str) -> None:
    ingest_url = f"{base_url}/api/v1/rag/ingest"
    stats_url = f"{base_url}/api/v1/rag/stats"

    print(f"Seeding RAG index at {base_url} ...\n")

    with httpx.Client(timeout=30) as client:
        # Show current state
        try:
            stats = client.get(stats_url).json()
            print(f"Before: {stats['indexed_chunks']} chunks in {stats['adapter']}\n")
        except Exception as exc:
            print(f"Could not reach {stats_url}: {exc}")
            sys.exit(1)

        # Ingest each document
        for doc in SAMPLE_DOCUMENTS:
            payload = {
                "title": doc["title"],
                "source": doc["source"],
                "content": doc["content"],
                "metadata": {"document_id": doc["id"]},
            }
            resp = client.post(ingest_url, json=payload)
            resp.raise_for_status()
            result = resp.json()
            print(f"  ✓  [{result['document_id']}] {result['title']} — {result['chunks']} chunks")

        # Show updated state
        stats = client.get(stats_url).json()
        print(f"\nAfter:  {stats['indexed_chunks']} chunks indexed in {stats['adapter']}")

    print("\nDone. Try these queries in the chat endpoint:")
    print('  "What are the ampacity ratings for Drake and Cardinal conductors?"')
    print('  "Explain the NERC TPL-001 planning events P0 through P4."')
    print('  "What is the EIM Resource Sufficiency Evaluation?"')
    print('  "What are best practices for SCADA alarm management?"')


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the Genie Platform RAG index.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8081",
        help="Base URL of the running Genie Platform (default: http://127.0.0.1:8081)",
    )
    args = parser.parse_args()
    seed(args.base_url)
