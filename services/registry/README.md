# Registry Service

The agent discovery service for the Genie platform: a standalone FastAPI app that agents self-register with and heartbeat to, and that consumers query to discover live agents. Records are persisted in MongoDB with a TTL liveness window (`REGISTRY_TTL_SECONDS`, default 90s) so stale instances disappear automatically; each agent advertises an `AgentMeta` (capabilities, I/O schema, endpoint, A2A skills). All routes except `/health` are guarded by an optional bearer token (`REGISTRY_AUTH_TOKEN`).

## Run

```bash
uv run python -m registry_service.service   # binds 0.0.0.0:2005
```

## Endpoints

- `POST /register` — register/refresh an instance (`{"meta": <AgentMeta>}`)
- `POST /heartbeat/{instance_id}` — refresh liveness (optional `{"status": ...}` body)
- `POST /deregister` — remove an instance (`{"instance_id": ...}`)
- `GET /agents` — list live agents (optional `?agent_id=` / `?tag=`)
- `GET /agents/{agent_id}` — list live instances of one agent
- `GET /health` — liveness probe (unauthenticated)
