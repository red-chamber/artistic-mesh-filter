from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a project configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    path: Path
    project_root: Path
    values: dict[str, Any]

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def output_path(self, key: str) -> Path:
        outputs = self.values.get("outputs", {})
        if key not in outputs:
            raise ConfigError(f"Missing outputs.{key} in {self.path}")
        return self.resolve(outputs[key])

    def source_root(self, source_name: str) -> Path:
        sources = self.values.get("sources", {})
        if source_name not in sources:
            raise ConfigError(f"Unknown source_dataset {source_name!r}")
        root = sources[source_name].get("root")
        if not root:
            raise ConfigError(f"Missing sources.{source_name}.root")
        return self.resolve(root)


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ConfigError(f"Expected a YAML mapping in {config_path}")
    return values


def load_project_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).expanduser().resolve()
    values = load_yaml(config_path)
    config_directory = next(
        (parent for parent in config_path.parents if parent.name == "configs"),
        None,
    )
    project_root = config_directory.parent if config_directory else Path.cwd().resolve()
    return ProjectConfig(path=config_path, project_root=project_root, values=values)
