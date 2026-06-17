# ADR 0008 — Rules Engine Selection: Zen Engine (GoRules) over Microsoft Rules Engine

**Status:** Accepted  
**Date:** 2026-06-07

---

## Context

The Genie platform needs a rules engine to execute business logic that is:
- Defined and maintained by domain experts (not software engineers)
- Auditable and explainable — grid operators must understand why a decision was made
- Performant enough to execute in real-time during AI agent pipeline runs
- Embeddable into the existing Python/Rust service stack without external process dependencies
- Integrated with the AI layer so that rule outcomes can feed into LLM reasoning

Two options were evaluated: **Microsoft Rules Engine** (C# / .NET) and **Zen Engine** by GoRules (Rust core, Python binding).

---

## Comparison: Zen Engine (GoRules) vs Microsoft Rules Engine

| Criteria | Zen Engine (GoRules) | Microsoft Rules Engine | Winner |
|----------|----------------------|------------------------|--------|
| **Primary language** | Rust core; Python, Node.js, Go, Java bindings | C# / .NET only | Zen Engine |
| **Rule format** | JSON Decision Tables and Decision Trees (visual) | Code-defined `Rule` objects in C# | Zen Engine |
| **Visual modeling** | GoRules Business Rules Editor (open-source web UI); rules are JSON files checked into Git | No visual editor; rules are written as code | Zen Engine |
| **Non-engineer authoring** | Yes — domain experts edit rules in the visual editor without writing code | No — requires a C# developer to modify rules | Zen Engine |
| **Architecture** | Embedded library (no external service); rules loaded from JSON at runtime | Embedded .NET library; rules defined in code or XML | Zen Engine |
| **Performance** | Sub-millisecond execution; Rust core with no GC pause; optimised for streaming evaluation | Reasonable performance for C# LINQ evaluation; GC pauses on large rule sets | Zen Engine |
| **Rule format portability** | JSON Decision Tables are human-readable, version-controlled, diff-friendly | C# rule objects are not easily diffable or auditable outside the IDE | Zen Engine |
| **AI / LLM integration** | Decision Table outputs are structured JSON — directly usable as LLM context; agent can explain rule outcome from JSON trace | No structured output format for LLM consumption | Zen Engine |
| **Platform fit** | Python binding (`pip install zen-engine`) integrates directly into the existing FastAPI service | Requires .NET runtime; incompatible with Python/Rust stack | Zen Engine |
| **Audit trail** | Execution trace returns the matched row, input values, and output values — full explainability | No built-in execution trace | Zen Engine |
| **OATI stack compatibility** | Native Python; same `pyproject.toml` as the rest of the platform | Would require a separate .NET service or process bridge | Zen Engine |
| **Open-source** | Yes — MIT license; active development | Yes — Apache 2.0; maintained by Microsoft | Tie |
| **Hosting model** | Embedded in the Python process; no external service | Embedded in .NET process | Zen Engine |
| **Community** | Growing; GoRules provides commercial support | Mature; large .NET community | Microsoft |
| **Rule versioning** | Rules are JSON files; Git history provides complete version audit trail | Rule versions managed in code; harder to track outside source control | Zen Engine |
| **Complex rule support** | Decision Trees, Scorecards, Expression functions, custom operators | Forward/backward chaining; complex rule inference | Microsoft |

**Decision: Zen Engine (GoRules) wins on 13 of 15 criteria.**

---

## Decision

Use **Zen Engine (GoRules)** as the platform rules engine.

### Primary Rationale

**1. Platform language compatibility**

The Genie platform is Python. Microsoft Rules Engine runs on .NET only. Embedding a .NET service into a Python ASGI application requires either a subprocess bridge (unreliable, high latency) or running a separate .NET microservice (operational overhead, network hop). Zen Engine has a native Python binding with no external process dependency.

**2. Non-engineer rule authoring**

Grid operations rules (e.g. alarm escalation thresholds, outage notification triggers, equipment rating overrides) must be maintainable by domain experts who understand the power system but do not write C# code. GoRules' visual Decision Table editor allows these experts to update rules via a web UI; changes produce a JSON diff that is reviewed and merged via GitLab like any other code change.

**3. AI integration**

When the `RulesEngineAgent` executes a Decision Table, the result includes the full execution trace: which rows were evaluated, which conditions matched, what output values were produced. This structured JSON trace can be passed directly to the synthesiser LLM as context, enabling responses like *"The alarm was escalated to Critical because the voltage deviation exceeded the 5% threshold defined in the substation alarm policy"*. Microsoft Rules Engine has no equivalent structured output for LLM consumption.

**4. Auditability**

Rules as JSON files in Git means:
- Every rule change is a commit with author, timestamp, and diff
- Rule versions can be tagged at release time
- Rollback is a `git revert`
- Code review is the approval mechanism for rule changes

---

## Integration Pattern

```python
# src/agents/rules_engine/agent.py

from zen import ZenEngine

class RulesEngineAgent:
    def __init__(self, rules_directory: str):
        self._engine = ZenEngine()
        self._rules_directory = rules_directory

    async def execute(self, task: AgentTask, context: dict) -> AgentResult:
        # Load the appropriate decision table from the rules directory
        rule_name = self._extract_rule_name(task.instruction)
        decision = self._engine.get_decision(f"{self._rules_directory}/{rule_name}.json")

        # Evaluate with structured input; result includes trace
        result = decision.evaluate(context.get("inputs", {}))

        # result.performance: execution time
        # result.result: output values
        # result.trace: row-by-row evaluation detail (for LLM context)
        return AgentResult(
            task_id=task.task_id,
            agent_id=self._agent_id,
            success=True,
            output=self._format_result(result),
            data={"result": result.result, "trace": result.trace},
        )
```

Decision Tables are JSON files stored in `config/rules/`:

```
config/
└── rules/
    ├── alarm_escalation.json        ← alarm priority escalation policy
    ├── outage_notification.json     ← notification routing by severity
    └── equipment_rating_override.json
```

Each JSON file is edited visually in the GoRules editor and committed to Git. Application agents reference rules by filename; adding a new rule requires no code change.

---

## Consequences

**Positive**
- Domain experts can create and update rules using the visual editor without developer involvement.
- Sub-millisecond rule evaluation adds negligible latency to the AI pipeline.
- Full execution trace enables explainable AI responses from the synthesiser.
- Rules are version-controlled JSON; change history and rollback are free from Git.
- No new runtime dependency; `pip install zen-engine` is the only addition.

**Negative**
- GoRules is a smaller vendor than Microsoft; commercial support availability should be verified.
- Complex forward/backward chaining (deep inference) is not a native Zen Engine capability; highly complex rule graphs may require a purpose-built inference engine.
- The GoRules visual editor must be deployed for domain experts; this is an additional internal tool to operate.

---

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| Microsoft Rules Engine | .NET only; incompatible with Python stack; no visual authoring; no structured output for LLM integration |
| Drools (Java) | JVM dependency; subprocess bridge required; overkill for the current rule complexity |
| Plain Python conditionals | Not maintainable by domain experts; no visual editor; no audit trail independent of Python releases |
| Rete.js / other JS engines | Requires Node.js runtime; does not fit Python FastAPI deployment |
