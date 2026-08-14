from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a flat project YAML config.

    PyYAML is used when installed. A tiny fallback parser supports the project's
    simple scalar config files so validation can still run in minimal environments.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_simple_yaml(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return loaded


def _load_simple_yaml(text: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            cfg[key] = value[1:-1]
        elif value.lower() in {"true", "false"}:
            cfg[key] = value.lower() == "true"
        else:
            try:
                cfg[key] = int(value)
            except ValueError:
                try:
                    cfg[key] = float(value)
                except ValueError:
                    cfg[key] = value
    return cfg
