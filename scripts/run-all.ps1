# Launch the full Genie Platform stack, each service in its own PowerShell window.
#
#   powershell -ExecutionPolicy Bypass -File scripts/run-all.ps1
#
# Order: MLflow (2001) -> Registry (2005) -> MCP tools (2002) -> agents (2010-2012)
#        -> Platform (2003). Agents self-register with the registry; the platform
# discovers them. Close a window to stop that service.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

function Start-Svc($title, $workdir, $command) {
    Write-Host "Starting $title ..."
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$Host.UI.RawUI.WindowTitle='$title'; Set-Location '$workdir'; $command"
    )
    Start-Sleep -Seconds 2
}

# 1. MLflow tracking server
Start-Svc "mlflow:2001" $root "uv run mlflow server --host 127.0.0.1 --port 2001 --backend-store-uri sqlite:///mlflow.db"

# 2. Registry / discovery service
Start-Svc "registry:2005" "$root\services\registry" "uv run python -m registry_service.service"

# 3. MCP tool server (weather + outage + docs)
Start-Svc "mcp:2002" "$root\services\agents" "uv run python mcp_weather_server.py"

# 4-6. Example distributed agents (self-register with the registry)
Start-Svc "weather-agent:2010" "$root\services\agents" "`$env:AGENT_PORT='2010'; uv run python weather_agent.py"
Start-Svc "outage-agent:2011"  "$root\services\agents" "`$env:AGENT_PORT='2011'; uv run python outage_agent.py"
Start-Svc "rag-agent:2012"     "$root\services\agents" "`$env:AGENT_PORT='2012'; uv run python rag_agent.py"

# 7. Genie Platform control plane
Start-Svc "genie-platform:2003" $root "uv run uvicorn app:create_app --factory --host 0.0.0.0 --port 2003 --reload"

Write-Host ""
Write-Host "All services launching. Open http://localhost:2003/ (chat) and /trace.html (trace viewer)."
Write-Host "Tip: set GENIE_ENABLE_GUARDS=false to run without the heavy guards extra."
