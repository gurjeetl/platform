"""Contract tests — verify import-linter boundaries are enforced."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_linter_contracts_pass() -> None:
    """Run import-linter and assert all contracts pass (portable: repo-root cwd)."""
    result = subprocess.run(
        [sys.executable, "-m", "importlinter.cli", "lint-imports"],
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )
    if result.returncode != 0:
        pytest.fail(
            "import-linter contract violations:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def test_platform_does_not_import_extracted_services() -> None:
    """The platform (src/genie) must never import extracted-service internals."""
    forbidden = ("rag_service", "registry_service", "genie_agent_sdk")
    genie_root = _REPO_ROOT / "src" / "genie"
    violations: list[str] = []

    for dirpath, _dirs, filenames in os.walk(genie_root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            filepath = os.path.join(dirpath, filename)
            with open(filepath, encoding="utf-8") as fh:
                try:
                    tree = ast.parse(fh.read(), filename=filepath)
                except SyntaxError:
                    continue
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                elif isinstance(node, ast.Import):
                    names.extend(a.name for a in node.names)
                for name in names:
                    if any(name == f or name.startswith(f + ".") for f in forbidden):
                        violations.append(f"{filepath}: imports {name}")

    assert not violations, "Platform imports extracted service internals:\n" + "\n".join(violations)
