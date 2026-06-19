"""Platform configuration via pydantic-settings.

Supports flat env-var overrides (GENIE_ prefix) and a YAML config file
that can carry nested structures for llm_services and mcp_services.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Nested config models (loaded from YAML, not env vars) ─────────────────────


class LLMModelConfig(BaseModel):
    """Configuration for one LLM backend (OpenAI-compatible HTTP API)."""

    host: str = "localhost"
    port: int = 8080
    model_name: str = "default"
    prompting_path: str = "v1"
    max_token_limit: int = 4096
    api_key: str = "not-required"
    # Explicit base URL override for hosted endpoints that the host/port/path
    # scheme can't express (e.g. https://api.openai.com/v1). When set, it wins.
    url: str | None = None

    @property
    def base_url(self) -> str:
        """Resolved endpoint: explicit ``url`` wins, else host/port/path scheme."""
        if self.url:
            return self.url
        return f"http://{self.host}:{self.port}/{self.prompting_path}"


class LLMServicesConfig(BaseModel):
    """Holds one entry per named LLM backend."""

    models: dict[str, LLMModelConfig] = {}


class MCPServiceConfig(BaseModel):
    """Connection details for one MCP server."""

    url: str
    transport: str = "streamable_http"
    connect_timeout: float = 30.0


class ApplicationServiceConfig(BaseModel):
    """Connection details for one application REST API backend."""

    url: str
    timeout_seconds: float = 5.0


# ── Top-level Settings ────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Top-level platform settings; flat fields come from env (GENIE_*) or YAML."""

    model_config = SettingsConfigDict(
        env_prefix="GENIE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = "genie-platform"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # LLM — flat settings (select which named model to use)
    llm_provider: str = "mock"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024

    # LLM service definitions (populated from YAML llm_services block)
    llm_services: LLMServicesConfig = Field(default_factory=LLMServicesConfig)

    # MCP service definitions (populated from YAML mcp_services block)
    # key = logical service name (e.g. "meter_service"), value = connection config
    mcp_services: dict[str, MCPServiceConfig] = Field(default_factory=dict)

    # Application REST API backends (populated from YAML application_services block)
    # key = logical service name (e.g. "conductor_service"), value = connection config
    application_services: dict[str, ApplicationServiceConfig] = Field(default_factory=dict)

    # RAG
    rag_mode: Literal["local", "remote"] = "local"
    rag_service_url: str = "http://localhost:8001"
    rag_timeout_seconds: float = 30.0
    rag_max_retries: int = 3
    rag_retry_backoff_factor: float = 0.5

    # API server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str | None = None

    # MLflow / Tracking
    mlflow_tracking_uri: str = (
        "sqlite:///mlruns.db"  # override in config/default.yaml for absolute path
    )
    mlflow_experiment_name: str = "genie-platform"

    # ── Application-provided prompt ───────────────────────────────────────────
    # Each application runs its own platform instance and supplies its persona +
    # domain context here (config YAML / GENIE_* env). Empty = use the platform's
    # generic default (see genie.application.prompts.DEFAULT_SYSTEM_*).
    app_system_prompt: str = ""
    app_system_context: str = ""

    # Human-in-the-loop
    hitl_auto_approve: bool = True
    hitl_approval_timeout_seconds: float = 300.0

    # Feature gates
    enable_rag: bool = True
    enable_hitl: bool = False  # HITL node retained but routed-around by default
    enable_tracking: bool = True

    # ── Content guards (ported from BaseAgentFramework; default ON) ───────────
    enable_guards: bool = True
    # Comma-separated topics the ban-topics scanner blocks (None = built-in list).
    guard_ban_topics: str | None = None
    # PII/secret redaction scanners. Disable to drop the heavy NER pass.
    guard_pii: bool = True
    # On a scanner RUNTIME error: False = fail-closed (block), True = fail-open.
    guard_fail_open: bool = False

    # ── Multi-store memory ────────────────────────────────────────────────────
    # "in_memory" (default, zero-dependency) or "mongo" (durable, multi-store).
    memory_backend: Literal["in_memory", "mongo"] = "in_memory"
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "agent_memory"
    redis_url: str | None = None  # hot blackboard mirror; None = disabled
    milvus_uri: str | None = None  # remote Milvus; None = use milvus_db_path
    milvus_db_path: str = "./milvus_local.db"
    milvus_collection: str = "long_term_memory"
    postgres_dsn: str | None = None  # durable commit store; None = disabled
    openai_embed_model: str = "text-embedding-3-small"

    # ── Distributed agents (A2A + registry discovery) ─────────────────────────
    # "distributed" (agents are remote services, primary), "local" (in-process
    # providers only — tests/dev), or "hybrid" (both).
    agent_mode: Literal["local", "distributed", "hybrid"] = "distributed"
    registry_url: str = "http://127.0.0.1:2005"
    registry_auth_token: str | None = None
    registry_ttl_seconds: int = 90
    registry_heartbeat_seconds: int = 30
    registry_cache_ttl_seconds: float = 5.0
    registry_timeout_seconds: float = 3.0
    registry_serve_stale: bool = True
    # How often the platform refreshes the discovered-agent list (0 = once at startup).
    agent_refresh_seconds: float = 30.0

    # ── Router / planner tuning ───────────────────────────────────────────────
    # Toggle the router step. When false, the pipeline skips fast/chitchat triage
    # and every request goes straight to the full planner (input_guard → planner → …).
    enable_router: bool = True
    router_intent_classifier: bool = True
    router_intent_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    router_intent_threshold: float = 0.30
    router_intent_min_agents: int = 2
    router_min_confidence: float = 0.7
    planner_max_facts: int = 40
    max_replans: int = 3

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file, with env vars taking priority.

        Strategy:
        - Flat settings (llm_provider, rag_mode, …): env var wins if set;
          YAML value used only when the corresponding GENIE_* env var is absent.
        - Nested settings (llm_services, mcp_services): always taken from YAML
          because env vars cannot express nested dicts.
        """
        yaml_data: dict[str, Any] = {}
        try:
            import yaml  # type: ignore[import]

            with open(path, encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh)
                if isinstance(loaded, dict):
                    yaml_data = loaded
        except (ImportError, OSError):
            pass

        # Pull out nested structures before processing flat keys
        llm_raw = yaml_data.pop("llm_services", None)
        mcp_raw = yaml_data.pop("mcp_services", None)
        app_svc_raw = yaml_data.pop("application_services", None)

        # Load base settings from env vars (authoritative for flat fields)
        instance = cls()

        # Collect updates: YAML flat values only when env var is absent
        updates: dict[str, Any] = {
            k: v for k, v in yaml_data.items() if f"GENIE_{k.upper()}" not in os.environ
        }

        # Nested configs always come from YAML (env vars can't represent them)
        if llm_raw and isinstance(llm_raw, dict):
            updates["llm_services"] = LLMServicesConfig.model_validate(llm_raw)
        if mcp_raw and isinstance(mcp_raw, dict):
            updates["mcp_services"] = {
                name: MCPServiceConfig.model_validate(svc) for name, svc in mcp_raw.items()
            }
        if app_svc_raw and isinstance(app_svc_raw, dict):
            updates["application_services"] = {
                name: ApplicationServiceConfig.model_validate(svc)
                for name, svc in app_svc_raw.items()
            }

        return instance.model_copy(update=updates)


# ── Singleton cache ───────────────────────────────────────────────────────────
_settings: Settings | None = None
_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Config resolution order:
    1. ``GENIE_CONFIG_FILE`` env var — explicit path to a YAML file
    2. ``config/default.yaml`` next to the project root (resolved from this
       file's location so it works regardless of the CWD when uvicorn starts)
    3. ``config/default.yaml`` in the current working directory (legacy fallback)
    4. Env vars only (no YAML file)
    """
    global _settings
    if _settings is None:
        with _lock:
            if _settings is None:
                config_file = os.environ.get("GENIE_CONFIG_FILE")
                if config_file is None:
                    # Resolve relative to project root: src/genie/platform/config.py
                    # goes up 4 levels to reach the project root.
                    _project_root = Path(__file__).resolve().parent.parent.parent.parent
                    anchor_path = _project_root / "config" / "default.yaml"
                    if anchor_path.exists():
                        config_file = str(anchor_path)
                    else:
                        # Legacy CWD-relative fallback
                        cwd_path = Path("config/default.yaml")
                        if cwd_path.exists():
                            config_file = str(cwd_path)
                _settings = Settings.from_yaml(config_file) if config_file else Settings()
    return _settings


def override_settings(s: Settings) -> None:
    """Replace the cached singleton — intended for test injection."""
    global _settings
    with _lock:
        _settings = s
