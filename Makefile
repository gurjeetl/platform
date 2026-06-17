.PHONY: install run chat rag-service run-split test test-unit test-integration test-e2e lint format typecheck check import-check mlflow-ui clean

install:
	uv sync --all-extras --dev

run:
	uv run uvicorn genie.interface.app:create_app --factory --host 0.0.0.0 --port 8000 --reload

chat:
	uv run python -m genie.interface.cli

rag-service:
	cd services/rag_service && uv run uvicorn rag_service.app:create_app --factory --host 0.0.0.0 --port 8001 --reload

run-split:
	@echo "Split-service mode:"
	@echo "  Terminal 1 — start the RAG service:"
	@echo "    make rag-service"
	@echo ""
	@echo "  Terminal 2 — start the platform with remote RAG:"
	@echo "    GENIE_RAG_MODE=remote make run"

test:
	uv run pytest tests/ -v

test-unit:
	uv run pytest tests/unit/ -v

test-integration:
	uv run pytest tests/integration/ -v

test-e2e:
	uv run pytest tests/e2e/ -v

lint:
	uv run ruff check src/ tests/ services/

format:
	uv run ruff format src/ tests/ services/

typecheck:
	uv run mypy src/genie/

import-check:
	uv run lint-imports

check: lint typecheck import-check test

mlflow-ui:
	uv run mlflow ui --backend-store-uri ./mlruns

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
