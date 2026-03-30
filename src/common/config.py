from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a config file cannot be loaded or validated."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON-compatible YAML config without forcing external dependencies."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as import_error:
            raise ConfigError(
                f"{config_path} is not valid JSON-compatible YAML and PyYAML is unavailable."
            ) from import_error

        try:
            data = yaml.safe_load(text)
        except Exception as yaml_error:  # pragma: no cover - depends on optional YAML parser
            raise ConfigError(f"Failed to parse YAML config at {config_path}.") from yaml_error

    if not isinstance(data, dict):
        raise ConfigError(f"Config at {config_path} must load to a mapping.")

    return data
