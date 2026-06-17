# ADR 0007 — Technology Stack Selection

**Status:** Accepted  
**Date:** 2026-06-07

---

## Context

The Genie platform must support production AI workloads for a utility company (OATI) that operates its own on-premises data centres. Technology choices must satisfy:

- **Data sovereignty** — customer grid data cannot leave OATI-controlled infrastructure.
- **Cost predictability** — per-token SaaS LLM fees are not acceptable at production query volumes.
- **Regulatory fit** — utility operations require auditability, access control, and compliance tooling.
- **Developer velocity** — application teams should ship new agents without DevOps involvement.

This ADR records the decisions and rationale for each major technology component.

---

## Selected Technologies and Rationale

### 1. Kubernetes + GPU Nodes (Compute Infrastructure)

**Decision:** Deploy on OATI's existing Kubernetes clusters with GPU-enabled node pools for model inference.

| Benefit | Detail |
|---------|--------|
| **Hardware control** | GPU nodes (NVIDIA A100/H100) run vLLM model servers; no dependency on cloud GPU availability or pricing |
| **Unified operations** | Same K8s tooling used for all OATI services — Helm charts, Prometheus/Grafana, GitLab CI/CD pipelines already exist |
| **Horizontal scaling** | LangGraph control-plane replicas scale independently of inference nodes |
| **Isolation** | Separate namespaces for platform, agent services, RAG service, and MLflow |
| **Cost** | Amortised hardware cost vs. continuous per-token SaaS billing at scale |

**Trade-off:** Higher initial infrastructure investment; OATI already owns the hardware so the incremental cost is low.

---

### 2. vLLM (LLM Inference Engine)

**Decision:** Use vLLM as the self-hosted LLM inference server, accessed via its OpenAI-compatible REST API.

| Benefit | Detail |
|---------|--------|
| **OpenAI API compatibility** | The platform's `OpenAICompatibleLLMProvider` works with vLLM with zero code changes; switching models means updating `config/default.yaml` only |
| **Throughput optimisation** | PagedAttention and continuous batching give 2–24× higher throughput vs. naive HuggingFace inference |
| **Multi-model serving** | Single vLLM instance can serve multiple models (e.g. Llama-3-70B for reasoning, Llama-3-8B for classification) |
| **Quantisation** | GPTQ/AWQ quantisation reduces GPU memory requirements, enabling larger models on existing hardware |
| **Data privacy** | All tokens stay within OATI network; no data sent to external APIs |

**Trade-off:** OATI team is responsible for model updates, GPU driver maintenance, and scaling. This is acceptable given data sovereignty requirements.

---

### 3. LangChain + LangGraph (Orchestration Framework)

**Decision:** Use LangChain as the LLM integration toolkit and LangGraph as the pipeline orchestration engine. See [ADR 0002](0002-langgraph-pipeline.md) for the detailed agent framework comparison (LangGraph vs. Google ADK).

| Benefit | Detail |
|---------|--------|
| **Unified LLM abstraction** | `BaseChatModel` works with vLLM, OpenAI, Anthropic, and any OpenAI-compatible endpoint |
| **Tool integration** | `@tool` decorator; MCP adapter; native function calling schema generation |
| **Graph-based pipeline** | LangGraph `StateGraph` provides typed state, conditional edges, retry, HITL |
| **Observability** | `mlflow.langchain.autolog()` captures full node-by-node trace waterfall with zero extra code |
| **Community** | Largest AI orchestration community; extensive documentation and examples |

---

### 4. Agent-to-Agent Protocol (A2A)

**Decision:** Adopt A2A (Google's open agent communication protocol) for inter-agent communication.

| Benefit | Detail |
|---------|--------|
| **Standardised interface** | Agents expose a well-defined HTTP API (Task, Artifact, Message schema) regardless of implementation language |
| **Language agnostic** | A Python LangGraph agent can call a Java-based grid calculation agent without wrapper code |
| **Discoverability** | Agents publish an `agent-card.json` manifest; the registry can discover capabilities dynamically |
| **Future-proofing** | As the agent ecosystem grows, A2A enables a marketplace model — agents from different teams or vendors can interoperate |
| **Decoupling** | Sub-agents can be deployed and scaled independently; the orchestrator only knows the A2A endpoint |

**Current status:** A2A endpoints are stubbed in `src/agents/`; full integration is on the roadmap for Phase 2.

---

### 5. Model Context Protocol (MCP)

**Decision:** Expose all tools (data APIs, calculation services, external systems) as MCP servers.

| Benefit | Detail |
|---------|--------|
| **Universal tool standard** | Any MCP-compatible AI system (GPT-4, LangGraph agents, etc.) can use platform tools without integration code |
| **Security boundary** | MCP servers validate inputs and enforce access control before calling backend systems |
| **Tool discoverability** | `tools/list` call returns available tools with schema; agents can query what tools exist at runtime |
| **OATI data exposure** | Grid data, outage records, market data can be exposed as read-only MCP resources; write operations require explicit tool calls |

**Current implementation:** `MCPClient` + `MCPToolAdapter` in `src/genie/mcp/` connect to configured MCP servers at startup. Tool registrations are loaded automatically and made available to all agents via `ToolGateway`.

---

### 6. Milvus (Vector Database for RAG)

**Decision:** Use Milvus as the vector database backing the production RAG service.

| Benefit | Detail |
|---------|--------|
| **Self-hosted** | Milvus runs on OATI's Kubernetes cluster; no external SaaS dependency |
| **Scale** | Handles billions of vectors with millisecond ANN search; appropriate for large OATI document corpus |
| **Metadata filtering** | Supports structured field filters alongside vector similarity (e.g. filter by document source, date range) |
| **Kubernetes-native** | Milvus Operator deploys via Helm; integrates with OATI's existing K8s monitoring stack |
| **GPU acceleration** | Index build can use GPU for faster embedding computation on large ingestion batches |

**Current status:** `LocalRAGAdapter` (keyword matching) is used in development and CI. `RemoteRAGAdapter` connects to the production Milvus service via the RAG microservice HTTP API.

---

### 7. ReactJS / webVision (Frontend Visualisation)

**Decision:** Use OATI's existing webVision platform (ReactJS-based) as the user interface for Genie.

| Benefit | Detail |
|---------|--------|
| **Existing investment** | webVision is already deployed across OATI customer sites; no new UI infrastructure required |
| **Domain context** | Grid visualisation, single-line diagrams, and real-time SCADA data are already rendered in webVision; AI responses can be shown alongside live grid state |
| **Single SSO** | Users authenticate once to webVision; Genie inherits the same session and RBAC roles |
| **Chat widget integration** | Genie chat panel is embedded as a webVision widget; responses can reference and highlight elements on the grid diagram |

---

### 8. MLflow (Experiment Tracking and Observability)

**Decision:** Use MLflow for experiment tracking, model metadata, and LLM request tracing. See [ADR 0005](0005-mlflow-tracking.md) for implementation details.

| Benefit | Detail |
|---------|--------|
| **Unified observability** | Single UI for experiment runs (params/metrics) and LangChain trace waterfall |
| **Self-hosted** | MLflow server with SQLite (dev) or PostgreSQL (prod) backend; no SaaS dependency |
| **LangChain integration** | `mlflow.langchain.autolog(log_traces=True)` captures node-by-node latency with zero code |
| **Model registry** | When fine-tuned models are developed, MLflow Model Registry tracks versions, stages, and deployment history |
| **Audit trail** | Every chat request is logged as an MLflow run; request parameters, agent selections, and response metrics are permanently recorded |

---

### 9. webVision RBAC + Conversation Logging

**Decision:** Inherit role-based access control from webVision and log all AI conversations to OATI's existing audit database.

| Benefit | Detail |
|---------|--------|
| **Regulatory compliance** | NERC CIP and utility regulatory frameworks require access audit logs; reusing webVision's existing logging satisfies this requirement |
| **No new auth system** | Genie does not introduce a separate identity system; users, roles, and permissions are managed in webVision's existing RBAC |
| **Conversation history** | All Genie conversations are stored with user ID, timestamp, and full message history; available for compliance review |
| **Role-based feature gating** | Administrative actions (bulk agent configuration, model selection) can be restricted to specific webVision roles |

---

### 10. GitLab (Source Control, CI/CD, Container Registry)

**Decision:** Use OATI's existing GitLab instance for source control, CI/CD pipelines, and the Docker/Helm container registry.

| Benefit | Detail |
|---------|--------|
| **Existing investment** | GitLab is already deployed and operated by OATI; zero new infrastructure |
| **Integrated pipelines** | `.gitlab-ci.yml` runs `lint-imports`, `pytest`, `docker build`, and Helm deployment in a single pipeline |
| **Air-gapped compatibility** | GitLab works in OATI's controlled network environment without external internet access during CI |
| **Container registry** | All platform and agent Docker images are pushed to and pulled from OATI's internal registry |

---

## Summary Table

| Component | Technology | Key Reason |
|-----------|------------|------------|
| Compute | Kubernetes + GPU nodes | OATI-owned hardware; no cloud GPU dependency |
| LLM inference | vLLM | OpenAI-compatible API; PagedAttention throughput; data privacy |
| Orchestration | LangChain + LangGraph | Best-in-class HITL; open-source; no vendor lock-in |
| Inter-agent comms | A2A protocol | Language-agnostic; discoverable; future marketplace |
| Tool exposure | MCP server | Universal standard; security boundary; discoverability |
| Vector DB | Milvus | Self-hosted; billion-scale; K8s native |
| UI | ReactJS / webVision | Existing OATI platform; grid context integration |
| Observability | MLflow | Unified runs + traces; self-hosted; LangChain native |
| Access control | webVision RBAC | Regulatory compliance; no new auth system |
| DevOps | GitLab | Existing OATI infrastructure; air-gap compatible |

---

## Consequences

**Positive**
- All components are self-hosted or open-source; no SaaS vendor can unilaterally change pricing or deprecate an API.
- Every component is already used in OATI's infrastructure except vLLM and Milvus, minimising new operational surface.
- The OpenAI-compatible API surface means LLM providers can be swapped by changing `config/default.yaml`.

**Negative**
- Self-hosted infrastructure requires OATI DevOps ownership of vLLM upgrades and Milvus operations.
- The A2A integration is not yet complete; inter-agent communication currently uses in-process calls.
- GPU node procurement lead times must be factored into capacity planning.
