"""Application bootstrap — creates and configures the FastAPI app.

The platform deliberately imports NO application-specific agents or tools.
Agents are injected by the caller (see src/app.py) via the ``agent_providers``
parameter so that adding a new agent never requires touching this file.

Each provider is a callable with signature:
    (*, tool_gateway: ConcreteToolGateway, settings: Settings) -> BaseAgent

The provider is responsible for registering any tools its agent needs and
returning the fully constructed agent instance.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from genie.agents import AgentRegistry
from genie.agents.remote import RemoteAgent
from genie.application.graph import build_graph
from genie.discovery.client import DiscoveryClient
from genie.llm import LLMRegistry, MockLLMProvider, OpenAICompatibleLLMProvider
from genie.mcp.client import MCPClient
from genie.mcp.tool_adapter import MCPToolAdapter
from genie.memory import InMemoryLongTermStore, InMemorySessionStore
from genie.memory.factory import create_memory, create_redis
from genie.observability.correlation import CorrelationMiddleware
from genie.observability.logging import configure_logging, get_logger
from genie.platform.config import Settings
from genie.platform.errors import ErrorCode, GenieError, error_response
from genie.platform.event_bus import EventBus
from genie.rag.factory import create_rag_adapter
from genie.security.auth import ApiKeyMiddleware
from genie.security.guard import create_guard
from genie.tools.gateway import ConcreteToolGateway
from genie.tracking.mlflow_tracker import MLflowTracker

# Type alias for an agent provider callable
AgentProviderFn = Callable[..., Any]

from genie.interface.routers.agents import router as agents_router
from genie.interface.routers.chat import router as chat_router
from genie.interface.routers.health import router as health_router
from genie.interface.routers.rag import router as rag_router
from genie.interface.routers.ui import router as ui_router

logger = get_logger(__name__)


async def _refresh_discovered_agents(
    agent_registry: AgentRegistry, discovery: DiscoveryClient
) -> None:
    """Reconcile the in-process registry with the live distributed-agent set.

    Registers newly-discovered agents as ``RemoteAgent``s and unregisters those
    that have dropped out of discovery. Best-effort — a registry outage is logged
    and the last known set is kept.
    """
    try:
        metas = await discovery.list_active(force_refresh=True)
    except Exception as exc:  # noqa: BLE001 — registry down: keep current set
        logger.warning("discovery_refresh_failed", error=str(exc))
        return

    discovered = {m.agent_id: m for m in metas}
    current = set(agent_registry.list_agents())
    # Add new remote agents.
    for agent_id, meta in discovered.items():
        if agent_id not in current:
            agent_registry.register(RemoteAgent(meta))
            logger.info("remote_agent_registered", agent_id=agent_id, endpoint=meta.endpoint)
    # Remove remote agents that disappeared (only RemoteAgents — leave in-process ones).
    for agent_id in current - set(discovered):
        existing = agent_registry.get(agent_id)
        if isinstance(existing, RemoteAgent):
            with contextlib.suppress(Exception):
                agent_registry.unregister(agent_id)
                logger.info("remote_agent_unregistered", agent_id=agent_id)


async def _discovery_loop(
    agent_registry: AgentRegistry, discovery: DiscoveryClient, interval: float
) -> None:
    while True:
        try:
            await asyncio.sleep(interval)
            await _refresh_discovered_agents(agent_registry, discovery)
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("discovery_loop_error", error=str(exc))


def _build_llm_registry(settings: Settings) -> tuple[LLMRegistry, Any]:
    """Wire LLM registry and return (registry, selected_provider).

    Only providers declared in ``llm_services.models`` (config/default.yaml) are
    registered.  When no models are configured (unit/e2e tests that pass a bare
    Settings object), a mock is silently added so tests remain self-contained.
    """
    registry = LLMRegistry()

    # Register every model declared in the YAML config — nothing extra.
    import os as _os

    for model_name, model_cfg in settings.llm_services.models.items():
        # Hosted providers (OpenAI) read the key from OPENAI_API_KEY when the YAML
        # doesn't carry one — avoids committing secrets to config.
        api_key = model_cfg.api_key
        if api_key in ("", "not-required", None):
            api_key = _os.getenv("OPENAI_API_KEY", api_key)
        provider = OpenAICompatibleLLMProvider(
            base_url=model_cfg.base_url,
            model_name=model_cfg.model_name,
            api_key=api_key,
            provider_name=model_name,
            max_token_limit=model_cfg.max_token_limit,
        )
        registry.register(model_name, provider)
        logger.info("llm_registered", name=model_name, base_url=model_cfg.base_url)

    # Dev/test fallback: if the YAML defines no models, register a mock so
    # tests that pass a bare Settings() still have a working provider.
    if not settings.llm_services.models:
        registry.register("mock", MockLLMProvider())
        logger.info("llm_registered", name="mock", base_url="(in-process)")

    # Select the provider named in settings.llm_provider.
    selected_name = settings.llm_provider
    if selected_name not in registry.list_providers():
        available = registry.list_providers()
        fallback = available[0] if available else "mock"
        logger.warning(
            "llm_provider_not_found",
            requested=selected_name,
            using=fallback,
            available=available,
        )
        selected_name = fallback

    return registry, registry.get(selected_name)


def _build_mcp_adapters(
    settings: Settings, tool_gateway: ConcreteToolGateway
) -> list[MCPToolAdapter]:
    adapters = []
    for service_name, svc_cfg in settings.mcp_services.items():
        client = MCPClient(
            name=service_name,
            url=svc_cfg.url,
            transport=svc_cfg.transport,
            connect_timeout=svc_cfg.connect_timeout,
        )
        adapter = MCPToolAdapter(
            service_name=service_name,
            client=client,
            gateway=tool_gateway,
        )
        adapters.append(adapter)
    return adapters


def _build_dependencies(
    settings: Settings,
    agent_providers: list[AgentProviderFn] | None = None,
) -> dict[str, Any]:
    """Wire up all platform dependencies and return them as a dict.

    ``agent_providers`` is a list of callables supplied by the application layer.
    Each is invoked with keyword arguments ``tool_gateway`` and ``settings`` and
    must return a ``BaseAgent`` instance.  Providers are also responsible for
    registering any tools their agent depends on.
    """
    # LLM
    llm_registry, llm_provider = _build_llm_registry(settings)

    # RAG
    rag_adapter = create_rag_adapter(settings) if settings.enable_rag else None

    # Tools — gateway starts empty; application providers register their own tools
    tool_gateway = ConcreteToolGateway()

    # MCP adapters (connected during lifespan startup)
    mcp_adapters = _build_mcp_adapters(settings, tool_gateway)

    # Agents — registered via injected providers (in-process; used in "local"/"hybrid"
    # modes). Distributed agents are discovered from the registry in the lifespan.
    agent_registry = AgentRegistry()
    for provider in agent_providers or []:
        agent = provider(tool_gateway=tool_gateway, settings=settings)
        agent_registry.register(agent)
        logger.info("agent_registered", agent_id=agent.agent_id)

    # Discovery client — when agents are distributed/hybrid, the lifespan queries it
    # and registers each live agent as a RemoteAgent.
    discovery_client = (
        DiscoveryClient(settings)
        if getattr(settings, "agent_mode", "distributed") in ("distributed", "hybrid")
        else None
    )

    # Memory — in-process session/long-term stores (LangGraph context) plus the
    # optional multi-store backend (Mongo/Milvus/Redis) used by planner recall +
    # synthesizer write-back. ``memory`` is None when memory_backend == "in_memory".
    session_store = InMemorySessionStore()
    long_term_store = InMemoryLongTermStore()
    memory = create_memory(settings, llm=llm_provider)
    # Redis blackboard mirror: from the memory facade when present, else standalone.
    redis_store = memory.redis if memory is not None else create_redis(settings)

    # Event bus
    event_bus = EventBus()

    # Content guards (mandatory when enable_guards; fail-closed on missing extra)
    guard = create_guard(settings)

    # Tracking
    tracker = MLflowTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )

    # Graph
    graph, checkpointer = build_graph(
        llm_provider=llm_provider,
        agent_registry=agent_registry,
        tool_gateway=tool_gateway,
        retrieval_service=rag_adapter,
        ingestion_service=rag_adapter,
        event_bus=event_bus,
        settings=settings,
        guard=guard,
        memory=memory,
        redis_store=redis_store,
    )

    return {
        "settings": settings,
        "llm_registry": llm_registry,
        "llm_provider": llm_provider,
        "rag_adapter": rag_adapter,
        "tool_gateway": tool_gateway,
        "mcp_adapters": mcp_adapters,
        "agent_registry": agent_registry,
        "discovery_client": discovery_client,
        "session_store": session_store,
        "long_term_store": long_term_store,
        "memory": memory,
        "event_bus": event_bus,
        "tracker": tracker,
        "guard": guard,
        "graph": graph,
        "checkpointer": checkpointer,
    }


def create_app(
    settings: Settings | None = None,
    agent_providers: list[AgentProviderFn] | None = None,
    startup_hooks: list[AgentProviderFn] | None = None,
) -> FastAPI:
    """Build and return a configured FastAPI application.

    Args:
        settings: Platform settings. Loaded from config/default.yaml when None.
        agent_providers: Application-supplied callables that construct and
            register agents.  Each is called with keyword arguments
            ``tool_gateway`` and ``settings``.  Pass your agents here — the
            platform never imports application code directly.

    Example (src/app.py)::

        from genie.interface.bootstrap import create_app

        def my_agent_provider(*, tool_gateway, settings):
            return MyAgent(tool_gateway=tool_gateway)

        app = create_app(agent_providers=[my_agent_provider])
    """
    from genie.platform.config import get_settings

    if settings is None:
        settings = get_settings()

    configure_logging(debug=settings.debug)

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        deps = _build_dependencies(settings, agent_providers=agent_providers)
        for key, val in deps.items():
            setattr(app.state, key, val)

        await deps["event_bus"].start()
        await deps["agent_registry"].start()  # Gap 1: begin background health-check loop

        # Warm the content-guard models so the FIRST request doesn't pay the
        # cold-kernel penalty (~1s). Off-thread + best-effort; never blocks startup.
        guard = deps.get("guard")
        if guard is not None and hasattr(guard, "warm"):
            try:
                await asyncio.to_thread(guard.warm)
                logger.info("llm_guard_warmed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("llm_guard_warm_failed", error=str(exc))

        # Initialise the durable memory backend (index creation) — best-effort.
        memory = deps.get("memory")
        if memory is not None and getattr(memory, "mongo", None) is not None:
            try:
                await memory.mongo.ensure_indexes()
            except Exception as exc:  # noqa: BLE001 — degrade to in-memory behavior
                logger.warning("memory_index_init_failed", error=str(exc))

        # Discover distributed agents and surface them as RemoteAgents, then keep
        # the set fresh on a background loop (distributed/hybrid modes only).
        discovery = deps.get("discovery_client")
        discovery_task: asyncio.Task | None = None
        if discovery is not None:
            await _refresh_discovered_agents(deps["agent_registry"], discovery)
            interval = float(getattr(settings, "agent_refresh_seconds", 30.0))
            if interval > 0:
                discovery_task = asyncio.create_task(
                    _discovery_loop(deps["agent_registry"], discovery, interval),
                    name="agent-discovery-loop",
                )

        # Run application startup hooks (e.g. seed RAG data).
        # Each hook receives the full deps dict; failures are isolated.
        for hook in startup_hooks or []:
            try:
                await hook(deps)
            except Exception as exc:
                logger.warning(
                    "startup_hook_failed", hook=getattr(hook, "__name__", "?"), error=str(exc)
                )

        # Connect MCP services — each failure is isolated; the app still starts.
        # Hard outer timeout so a hung service can never stall platform startup.
        for adapter in deps["mcp_adapters"]:
            svc = getattr(adapter, "_service_name", "unknown")
            try:
                await asyncio.wait_for(adapter.register(), timeout=8.0)
            except asyncio.TimeoutError:
                logger.warning("mcp_adapter_startup_timeout", service=svc, timeout=8.0)
            except Exception as exc:
                logger.warning(
                    "mcp_adapter_startup_failed",
                    service=svc,
                    error=type(exc).__name__,
                    detail=str(exc),
                )

        logger.info(
            "genie_platform_started",
            mode=settings.rag_mode,
            llm=settings.llm_provider,
            mcp_services=list(settings.mcp_services.keys()),
        )
        yield

        # Disconnect MCP services
        for adapter in deps["mcp_adapters"]:
            try:
                await adapter.unregister()
            except Exception:
                pass

        if discovery_task is not None:
            discovery_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await discovery_task
        await deps["agent_registry"].stop()  # Gap 1: cancel health-check loop
        await deps["event_bus"].stop()
        rag = deps.get("rag_adapter")
        if rag is not None and hasattr(rag, "aclose"):
            await rag.aclose()
        memory = deps.get("memory")
        if memory is not None:
            for store in (
                getattr(memory, "mongo", None),
                getattr(memory, "vector", None),
                getattr(memory, "redis", None),
            ):
                if store is not None and hasattr(store, "aclose"):
                    try:
                        await store.aclose()
                    except Exception:  # noqa: BLE001
                        pass
        logger.info("genie_platform_stopped")

    # Serve Swagger UI assets from the locally installed swagger-ui-bundle package
    # so the /docs page works on corporate networks that block cdn.jsdelivr.net.
    _swagger_js = "/swagger-static/swagger-ui-bundle.js"
    _swagger_css = "/swagger-static/swagger-ui.css"
    try:
        import swagger_ui_bundle as _sui

        _swagger_static_dir: str | None = str(_sui.swagger_ui_path)
    except ImportError:
        # Package not installed — fall back to CDN (requires internet access).
        _swagger_static_dir = None
        _swagger_js = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"
        _swagger_css = "https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css"
        logger.warning(
            "swagger_ui_bundle_missing",
            detail="run 'uv sync' — Swagger UI requires internet access until installed",
        )

    app = FastAPI(
        title="Genie Platform",
        description="Production-grade AI platform with LangGraph pipeline",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        swagger_js_url=_swagger_js,
        swagger_css_url=_swagger_css,
    )

    # ── Middleware (order matters: outermost is applied last) ─────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationMiddleware)  # type: ignore[arg-type]
    if settings.api_key:
        app.add_middleware(ApiKeyMiddleware, api_key=settings.api_key)  # type: ignore[arg-type]

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(agents_router)
    app.include_router(rag_router)
    app.include_router(ui_router)

    # ── Swagger UI static files ───────────────────────────────────────────────
    if _swagger_static_dir is not None:
        app.mount(
            "/swagger-static",
            StaticFiles(directory=_swagger_static_dir),
            name="swagger-static",
        )

    # ── Exception handlers ────────────────────────────────────────────────────
    @app.exception_handler(GenieError)
    async def genie_error_handler(request: Request, exc: GenieError) -> JSONResponse:
        cid = getattr(request.state, "correlation_id", "")
        status = _error_code_to_http_status(exc.code)
        return JSONResponse(
            status_code=status,
            content=error_response(exc, correlation_id=cid).model_dump(),
        )

    # ── Bundled frontends (chat UI + trace visualizer) ────────────────────────
    # Mounted LAST at "/" so all API routes above take precedence; serves
    # index.html at "/" and trace.html by name.
    from pathlib import Path as _Path

    _static_dir = _Path(__file__).resolve().parent / "static"
    if _static_dir.is_dir():
        app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="frontend")

    return app


def _error_code_to_http_status(code: ErrorCode) -> int:
    return {
        ErrorCode.NOT_FOUND: 404,
        ErrorCode.VALIDATION_ERROR: 422,
        ErrorCode.UNAUTHORIZED: 401,
        ErrorCode.FORBIDDEN: 403,
        ErrorCode.PROMPT_INJECTION: 400,
        ErrorCode.RATE_LIMITED: 429,
        ErrorCode.TIMEOUT: 504,
    }.get(code, 500)
