"""Build and compile the LangGraph workflow for the Genie platform.

Pipeline (ported from BaseAgentFramework, on Genie's StateGraph)::

    router ─┬─ chitchat ─────────────────────────────► synthesizer ─► END
            ├─ fast ──────────────────────────────────► executor
            └─ plan ─► planner ─► orchestrator ─┬─(HITL)─ human_approval ─► executor
                                                └────────────────────────► executor
    executor ─► completion_gate ─┬─ replan ─► planner
                                 └─ synthesize ─► synthesizer ─► END

The optional input/output content-guard nodes are added by the bootstrap when
``enable_guards`` is on (Phase 2); they bookend this graph.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from genie.application.checkpointer import create_checkpointer
from genie.application.nodes.completion_gate import CompletionGateNode
from genie.application.nodes.executor import ExecutorNode
from genie.application.nodes.guards import InputGuardNode, OutputGuardNode
from genie.application.nodes.orchestrator import OrchestratorNode
from genie.application.nodes.planner import PlannerNode
from genie.application.nodes.router import RouterNode
from genie.application.nodes.synthesizer import SynthesizerNode
from genie.application.state import GraphState
from genie.observability.logging import get_logger

logger = get_logger(__name__)

# ── Node names ────────────────────────────────────────────────────────────────
NODE_INPUT_GUARD = "input_guard"
NODE_OUTPUT_GUARD = "output_guard"
NODE_ROUTER = "router"
NODE_PLANNER = "planner"
NODE_ORCHESTRATOR = "orchestrator"
NODE_HUMAN_APPROVAL = "human_approval"
NODE_EXECUTOR = "executor"
NODE_COMPLETION_GATE = "completion_gate"
NODE_SYNTHESIZER = "synthesizer"


def _human_approval_node(state: GraphState) -> dict[str, Any]:
    """HITL pause point. LangGraph interrupts BEFORE this node when configured."""
    return {}


def _after_input_guard(state: GraphState) -> str:
    """A blocking input guard short-circuits straight to END with a refusal."""
    return END if state.guard_block else NODE_ROUTER


def _after_router(state: GraphState) -> str:
    route = (state.route or "plan").lower()
    if route == "chitchat":
        return NODE_SYNTHESIZER
    if route == "fast":
        return NODE_EXECUTOR
    return NODE_PLANNER


def _after_orchestrator(state: GraphState) -> str:
    return NODE_HUMAN_APPROVAL if state.requires_approval else NODE_EXECUTOR


def _after_completion_gate(state: GraphState) -> str:
    if state.metadata.get("gate_action") == "replan":
        return NODE_PLANNER
    return NODE_SYNTHESIZER


def build_graph(
    llm_provider: Any,
    agent_registry: Any,
    settings: Any,
    event_bus: Any | None = None,
    memory: Any | None = None,
    redis_store: Any | None = None,
    guard: Any | None = None,
    checkpointer: MemorySaver | None = None,
    # Accepted for backward compatibility; RAG is now served by a discovered agent.
    tool_gateway: Any | None = None,
    retrieval_service: Any | None = None,
    ingestion_service: Any | None = None,
) -> tuple[Any, MemorySaver]:
    """Build and compile the LangGraph workflow. Returns (compiled_graph, checkpointer)."""
    if checkpointer is None:
        checkpointer = create_checkpointer()

    router_node = RouterNode(
        llm_provider=llm_provider, agent_registry=agent_registry, settings=settings
    )
    planner_node = PlannerNode(
        agent_registry=agent_registry, settings=settings, llm_provider=llm_provider, memory=memory
    )
    orchestrator_node = OrchestratorNode(settings=settings)
    executor_node = ExecutorNode(
        agent_registry=agent_registry,
        settings=settings,
        event_bus=event_bus,
        redis_store=redis_store,
    )
    completion_gate_node = CompletionGateNode(settings=settings)
    synthesizer_node = SynthesizerNode(llm_provider=llm_provider, settings=settings, memory=memory)

    graph = StateGraph(GraphState)
    graph.add_node(NODE_ROUTER, router_node)
    graph.add_node(NODE_PLANNER, planner_node)
    graph.add_node(NODE_ORCHESTRATOR, orchestrator_node)
    graph.add_node(NODE_HUMAN_APPROVAL, _human_approval_node)
    graph.add_node(NODE_EXECUTOR, executor_node)
    graph.add_node(NODE_COMPLETION_GATE, completion_gate_node)
    graph.add_node(NODE_SYNTHESIZER, synthesizer_node)

    # ── Optional content guards bracket the pipeline (Phase 2) ────────────────
    if guard is not None:
        graph.add_node(NODE_INPUT_GUARD, InputGuardNode(guard))
        graph.add_node(NODE_OUTPUT_GUARD, OutputGuardNode(guard))
        graph.set_entry_point(NODE_INPUT_GUARD)
        graph.add_conditional_edges(
            NODE_INPUT_GUARD,
            _after_input_guard,
            {END: END, NODE_ROUTER: NODE_ROUTER},
        )
    else:
        graph.set_entry_point(NODE_ROUTER)

    # Router → planner | executor (fast) | synthesizer (chitchat)
    graph.add_conditional_edges(
        NODE_ROUTER,
        _after_router,
        {
            NODE_PLANNER: NODE_PLANNER,
            NODE_EXECUTOR: NODE_EXECUTOR,
            NODE_SYNTHESIZER: NODE_SYNTHESIZER,
        },
    )
    graph.add_edge(NODE_PLANNER, NODE_ORCHESTRATOR)
    graph.add_conditional_edges(
        NODE_ORCHESTRATOR,
        _after_orchestrator,
        {NODE_HUMAN_APPROVAL: NODE_HUMAN_APPROVAL, NODE_EXECUTOR: NODE_EXECUTOR},
    )
    graph.add_edge(NODE_HUMAN_APPROVAL, NODE_EXECUTOR)
    graph.add_edge(NODE_EXECUTOR, NODE_COMPLETION_GATE)
    graph.add_conditional_edges(
        NODE_COMPLETION_GATE,
        _after_completion_gate,
        {NODE_PLANNER: NODE_PLANNER, NODE_SYNTHESIZER: NODE_SYNTHESIZER},
    )
    if guard is not None:
        graph.add_edge(NODE_SYNTHESIZER, NODE_OUTPUT_GUARD)
        graph.add_edge(NODE_OUTPUT_GUARD, END)
    else:
        graph.add_edge(NODE_SYNTHESIZER, END)

    hitl_enabled = getattr(settings, "enable_hitl", False)
    hitl_auto_approve = getattr(settings, "hitl_auto_approve", True)
    interrupt_before_nodes: list[str] | None = None
    if hitl_enabled and not hitl_auto_approve:
        interrupt_before_nodes = [NODE_HUMAN_APPROVAL]

    compiled = graph.compile(checkpointer=checkpointer, interrupt_before=interrupt_before_nodes)
    logger.info("graph_built", hitl_enabled=hitl_enabled, interrupt_before=interrupt_before_nodes)
    return compiled, checkpointer
