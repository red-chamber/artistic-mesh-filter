"""Shared test helpers.

Tests must not depend on the repository's own fixture paths being writable, so
they build a throwaway source tree under `tmp_path` and copy the small OBJ
fixtures into it. The config file always lives in a `configs/` directory so
`load_project_config` resolves the project root to `tmp_path`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"

VALID_MESH = (FIXTURE_ROOT / "positive" / "cube.obj").read_text(encoding="utf-8")
SECOND_MESH = (FIXTURE_ROOT / "positive" / "tetrahedron.obj").read_text(encoding="utf-8")
INVALID_MESH = "this file is not a mesh\n"


@pytest.fixture
def valid_mesh() -> str:
    return VALID_MESH


@pytest.fixture
def second_mesh() -> str:
    return SECOND_MESH


@pytest.fixture
def invalid_mesh() -> str:
    return INVALID_MESH


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., Path]:
    def _make(
        sources: dict[str, Any],
        outputs: dict[str, Any],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        config_path = tmp_path / "configs" / "test.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"sources": sources, "outputs": outputs}
        payload.update(extra or {})
        config_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
        return config_path

    return _make


@pytest.fixture
def source_tree(tmp_path: Path) -> Callable[[str, dict[str, str]], Path]:
    def _make(name: str, files: dict[str, str]) -> Path:
        root = tmp_path / "sources" / name
        for relative, content in files.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return root

    return _make
