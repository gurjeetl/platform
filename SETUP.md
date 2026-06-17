# Developer Setup Guide

> **Merged distributed-agent platform.** The fastest path is the README
> [Quick start](README.md#quick-start) plus `scripts/run-all.ps1`, which launches the
> full stack: MLflow (2001), registry (2005), MCP tools (2002), the example agents
> (2010–2012), and the platform (2003). Agents are **distributed services** that
> self-register; the platform discovers them. Content guards default ON — install the
> `guards` extra (`uv sync --extra dev --extra guards`) or set
> `GENIE_ENABLE_GUARDS=false`. Multi-store memory is optional (`--extra memory-mongo`
> / `memory-redis` / `memory-vector`); the default is in-memory.
>
> The sections below are general `uv`/Python setup and troubleshooting reference.

This repository contains several independent services (platform, registry, agents,
optional RAG). Each has its own dependencies and is started separately.

---

## Quick Start (uv + Python 3.11+)

If you have **uv** and **Python 3.11+** already installed, this is everything
you need. No virtual environment activation required — `uv run` handles it.

### First time only (run once after cloning)

```bash
# From the project root
uv sync --extra dev

# DLR MCP service — install and configure DB credentials
cd services/dlr/mcp
uv sync
cp .env.example .env
```

Open `services/dlr/mcp/.env` and fill in the PostgreSQL credentials
(`POSTGRES_HOST`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`).

```bash
# RAG service (optional — only needed for remote RAG mode)
cd services/rag_service
uv sync
```

### Every time you start the project (3 terminals minimum, in order)

**Linux / macOS:**
```
Terminal 1 — MLflow (from project root):
    uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1

Terminal 2 — DLR MCP (from services/dlr/mcp/):
    cd services/dlr/mcp
    uv run python server.py

Terminal 3 — Main platform (from project root, start last):
    uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

**Windows:**
```
Terminal 1 — MLflow (from project root):
    uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1

Terminal 2 — DLR MCP (from services\dlr\mcp\):
    cd services\dlr\mcp
    uv run python server.py

Terminal 3 — Main platform (from project root, start last):
    uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

Start Terminal 1 first and wait for `Listening at: http://127.0.0.1:2001` before
starting Terminal 3. Terminal 2 can start in any order.

**Service URLs:**

| Service | URL | Notes |
|---------|-----|-------|
| Genie Platform | http://127.0.0.1:2003/docs | Swagger UI |
| DLR MCP | http://127.0.0.1:2002/mcp | MCP endpoint |
| MLflow UI | http://localhost:2001 | Traces appear here |
| RAG Service | http://127.0.0.1:2004 | Optional (remote RAG mode only) |

The platform finds `config/default.yaml` automatically — no `GENIE_CONFIG_FILE`
env var needed. MLflow URI and MCP URLs are already set in the config file.

---

## Prerequisites

### Python version

All services require **Python 3.11 or higher** (3.11, 3.12, and 3.13 all work —
the codebase sets `requires-python = ">=3.11"`).

Check what you have:

```bash
python3 --version        # may show 3.9, 3.10 — too old
python3.11 --version     # try explicit versions
python3.12 --version
python3.13 --version
```

If your system Python is older than 3.11, install a newer one:

| OS | How |
|----|-----|
| **macOS** | `brew install python@3.13` |
| **Ubuntu/Debian** | `sudo apt install python3.13` or use [deadsnakes PPA](https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa) |
| **Windows** | Download from [python.org](https://python.org/downloads/) |
| **Any OS** | Install `uv` (below) — it downloads Python automatically |

---

## Package manager: choose one

Both options produce identical results. Choose based on your environment.

---

### Option A — pip (works everywhere, no extra install)

pip ships with every Python installation. No downloads, no scripts to run,
no execution policy changes needed. **Use this if you are on a corporate
machine or cannot install new tools.**

Verify pip is available:
```bash
python3 -m pip --version     # Linux / macOS
py -m pip --version          # Windows
```

---

### Option B — uv (faster installs, handles Python version automatically)

`uv` is a fast package manager that resolves and locks dependencies in
seconds and handles Python version selection automatically. The project's
`Makefile` (`make install`, `make run`, `make test`) is written for `uv`.

**Install uv — pick the method your IT policy allows:**

```bash
# macOS / Linux — downloads and runs an install script
curl -LsSf https://astral.sh/uv/install.sh | sh
# restart your terminal after, or: source ~/.bashrc / source ~/.zshrc
```

```bash
# Windows — via pip (safest on corporate machines, no policy bypass)
pip install uv
```

```bash
# Windows — via winget (built into Windows 10/11, no policy change needed)
winget install astral-sh.uv
```

> **Corporate / restricted Windows machines:**
> Do not use the `powershell -ExecutionPolicy ByPass -c "irm ... | iex"` form
> that appears in uv's official docs. That command has two problems:
> `-ExecutionPolicy ByPass` explicitly overrides your IT-set policy, and
> `irm | iex` downloads and runs a script without inspection — both are
> common corporate security violations. Use `pip install uv` or
> `winget install astral-sh.uv` instead.

Verify: `uv --version`

---

## Service 1 — MLflow Tracking Server

**Port:** 2001
**Directory:** project root
**Required:** yes, if `enable_tracking: true` in config (default)

Start this **first** — the platform connects to MLflow at startup and will
skip trace recording if it cannot reach the server.

### Start MLflow server

```bash
# From the project root — uv
uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1

# pip (venv activated) — Linux / macOS / Windows
mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1
```

MLflow UI is then at `http://localhost:2001`.

Run this command from the **project root** — MLflow creates `mlflow.db` in
whatever directory you are in when you start it. Always start from the same
directory so every run uses the same database file.

### Using a different port

If you need a different port, edit `config/default.yaml` to match:

```yaml
mlflow_tracking_uri: "http://localhost:5000"   # ← change to your port
```

---

## Service 2 — DLR MCP Service (operational database)

**Port:** 2002
**Directory:** `services/dlr/mcp/`

This service is **completely independent** — it has its own environment and
does not import any genie platform code. Keep its environment separate.

### Setup — uv

```bash
cd services/dlr/mcp

uv sync                    # installs: mcp[cli], asyncpg, uvicorn, python-dotenv
```

### Setup — pip

```bash
cd services/dlr/mcp

python3.13 -m venv .venv           # or whichever ≥3.11 interpreter you have
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

### Configure database

```bash
cp .env.example .env
# Open .env and fill in:
#   POSTGRES_HOST=your-db-host
#   POSTGRES_USER=your-user
#   POSTGRES_PASSWORD=your-password
#   POSTGRES_DB=dlr_operational
#   HOST=0.0.0.0
#   PORT=2002
```

### Start the service

```bash
# uv
uv run python server.py

# pip (venv activated)
python server.py
```

Service is ready when you see:
```
DLR MCP service starting → http://0.0.0.0:2002/mcp
INFO:     Application startup complete.
```

### Run as a system service (Linux)

```bash
sudo cp dlr-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now dlr-mcp
journalctl -u dlr-mcp -f          # tail logs
```

---

## Service 3 — Genie Platform (main app)

**Port:** 2003
**Directory:** project root
**Start this last** — it connects to MLflow and the DLR MCP service at startup.

### Setup — uv

```bash
cd genie-platform          # project root

uv sync --extra dev        # installs all dependencies including dev tools
                           # uv auto-picks a compatible Python (3.11+)
```

### Setup — pip

```bash
cd genie-platform

# Create an isolated virtual environment using your ≥3.11 interpreter
python3.13 -m venv .venv           # or python3.11, python3.12, etc.

# Activate it
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# Install the local contracts package first (uv handles this automatically,
# pip needs it done manually)
pip install -e packages/genie-rag-contracts

# Install the platform with all dev dependencies
pip install -e ".[dev]"
```

### Start the platform

```bash
# Using uv (no activation needed)
uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload

# Using pip (virtual environment must be activated)
uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

Platform is ready when you see:
```
INFO:     Uvicorn running on http://0.0.0.0:2003 (Press CTRL+C to quit)
```

Swagger UI: http://127.0.0.1:2003/docs

### Chat interface (CLI)

```bash
uv run python -m genie.interface.cli      # uv
python -m genie.interface.cli             # pip (venv activated)
```

---

## Service 4 — RAG Service (optional)

**Port:** 2004
**Directory:** `services/rag_service/`

Only needed when `GENIE_RAG_MODE=remote` (default mode is `local`, which
runs RAG in-process inside the platform — no separate service required).

### Setup — uv

```bash
cd services/rag_service
uv sync
```

### Setup — pip

```bash
cd services/rag_service

python3.13 -m venv .venv
source .venv/bin/activate
pip install -e ../../packages/genie-rag-contracts   # local dependency
pip install -e .
```

### Start the service

```bash
# uv
uv run uvicorn rag_service.main:create_app --factory --host 0.0.0.0 --port 2004 --reload

# pip (venv activated)
uvicorn rag_service.main:create_app --factory --host 0.0.0.0 --port 2004 --reload
```

### Enable remote RAG mode

Start the platform with the mode override:

```bash
# Linux / macOS
GENIE_RAG_MODE=remote uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload

# Windows CMD
set GENIE_RAG_MODE=remote
uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload

# Windows PowerShell
$env:GENIE_RAG_MODE="remote"
uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

---

## Running all services together

Open three terminals from the **project root**. Start them in this order —
MLflow must be running before the platform starts.

### Linux / macOS (uv)

```
Terminal 1 (MLflow):   uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1
Terminal 2 (DLR MCP):  cd services/dlr/mcp && uv run python server.py
Terminal 3 (platform): uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

Optional (only for remote RAG mode):
```
Terminal 4 (RAG):      cd services/rag_service && uv run uvicorn rag_service.main:create_app --factory --host 0.0.0.0 --port 2004 --reload
```

### Windows (uv)

```
Terminal 1 (MLflow)    — from project root:
    uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1

Terminal 2 (DLR MCP)   — from services\dlr\mcp\:
    uv run python server.py

Terminal 3 (platform)  — from project root, start last:
    uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

### Windows (pip, venv activated per terminal)

```
Terminal 1 (MLflow)    — from project root, venv activated:
    mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1

Terminal 2 (DLR MCP)   — from services\dlr\mcp\, its own venv activated:
    python server.py

Terminal 3 (platform)  — from project root, venv activated, start last:
    uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload
```

**Port reference:**

| Terminal | Service | URL | Notes |
|----------|---------|-----|-------|
| 1 | MLflow UI | http://localhost:2001 | Traces appear here |
| 2 | DLR MCP | http://127.0.0.1:2002/mcp | MCP endpoint |
| 3 | Genie Platform | http://127.0.0.1:2003/docs | Swagger UI |
| 4 | RAG Service | http://127.0.0.1:2004 | Optional (remote RAG mode only) |

---

## Running tests

All commands from the project root, virtual environment activated or using `uv run`:

```bash
uv run pytest tests/              # all tests
uv run pytest tests/unit/         # unit only (fast, no external deps)
uv run pytest tests/integration/  # integration tests
uv run pytest tests/e2e/          # end-to-end (starts the full app)
```

Or via Make (requires uv):
```bash
make test        # all tests
make test-unit   # unit only
make lint        # ruff + mypy + import-linter
make check       # lint + typecheck + all tests
```

---

## Troubleshooting

### MLflow UI shows no traces after submitting a chat request

**Most common cause: platform and MLflow server are using different backends,
or the platform cannot reach the MLflow server.**

Check these in order:

1. **Is the MLflow server running?**
   Open `http://localhost:2001` in a browser.
   If it doesn't load, start it first (from the project root):
   ```bash
   uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db --workers 1
   ```

2. **Is the platform pointing at the right port?**
   Check `config/default.yaml` — it should say `http://localhost:2001`.
   The platform reads this on startup. If you changed the port, update the yaml
   to match, then restart the platform.

3. **Are both services using the same `--backend-store-uri`?**
   Always start ONE MLflow server from the project root so both use `mlflow.db`
   in the same directory.

4. **Check the platform startup log for mlflow warnings.**
   Look for `mlflow_init_failed` or `mlflow_autolog_failed` in the
   platform terminal.

### DLR pylon/span/rating queries return "service isn't available"

The DLR MCP service is not running or the platform could not connect to it at startup.

1. Start the MCP service first: `cd services/dlr/mcp && uv run python server.py`
2. Then restart the platform — MCP tools are registered at startup only.
3. Check `services/dlr/mcp/.env` has real PostgreSQL credentials (not `.env.example`).
4. On Windows: check `config/default.yaml` uses `127.0.0.1:2002` not `localhost:2002`
   (Windows may resolve `localhost` to IPv6 `::1`).

### `No module named 'structlog'` (or any other missing module)

You are running the main platform without having installed its dependencies.
Run from the **project root** (not a subdirectory):

```bash
uv sync --extra dev            # uv
# or
pip install -e ".[dev]"        # pip, with venv activated
```

### `uv: command not found`

uv is not installed. Either install it (see top of this guide) or use the
pip commands instead.

### `python3.11: command not found`

Your system does not have Python 3.11 installed under that name. Find what
you have:
```bash
python3 --version
python3.12 --version
python3.13 --version
```
Use whichever ≥3.11 interpreter is available. Or install uv — it handles
Python version management automatically (`uv python install 3.13`).

### `python3: command not found` (Windows)

Use `py` instead:
```bash
py --version          # shows installed version
py -3.13 -m venv .venv
```

### Port already in use

```bash
# Find what's using the port
lsof -i :2003              # macOS / Linux (substitute whichever port)
netstat -ano | findstr :2003   # Windows

# Option 1: kill the process using the port
# Option 2: use a different port and update config/default.yaml to match
uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 1005 --reload
```

On Windows, VS Code Remote SSH occupies several ports. Check with
`netstat -ano | findstr :<port>` and pick a free one if needed.

### `ModuleNotFoundError: No module named 'applications'` or `genie`

The `PYTHONPATH` is not set. When using uv, this is handled automatically
via `pyproject.toml`'s `pythonpath = ["src"]`. When using pip directly:

```bash
PYTHONPATH=src uvicorn app:create_app --factory --port 2003
```

Or activate the venv after `pip install -e ".[dev]"` — editable install adds
`src/` to the path.

### DLR MCP service connects but returns empty results

The PostgreSQL database is not reachable or the query matched no rows.

1. Confirm `services/dlr/mcp/.env` has valid credentials.
2. On Windows, if PostgreSQL is on a remote Linux host, you cannot connect directly —
   use an SSH tunnel: `ssh -L 5433:localhost:5432 user@dbhost -N`
   then set `POSTGRES_HOST=127.0.0.1` and `POSTGRES_PORT=5433` in `.env`.
3. Check that the pylon/span name you queried actually exists in the database
   (e.g. `P-108` must be an exact match — case-sensitive).
