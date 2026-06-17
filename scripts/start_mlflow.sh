#!/usr/bin/env bash
# Start the MLflow tracking UI.
# Run from any directory — uses absolute path to the database.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_PATH="${REPO_DIR}/mlruns.db"

echo "Starting MLflow UI  →  http://127.0.0.1:5000"
echo "Database: ${DB_PATH}"

mlflow ui \
  --backend-store-uri "sqlite:////${DB_PATH}" \
  --host 127.0.0.1 \
  --port 5000
