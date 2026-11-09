"""Load and lightly normalize config/moe_rail.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repo root = parent of src/."""
    return Path(__file__).resolve().parent.parent


def default_config_path() -> Path:
    return project_root() / "config" / "moe_rail.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """
    Read YAML into a plain dict.

    Why a separate loader? Keeps file I/O out of MoE/topology math so those
    modules stay easy to unit-test with hand-built dicts.
    """
    config_path = path or default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    data["_config_path"] = str(config_path.resolve())
    return data


def resolve_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    """Turn relative path knobs into absolute dirs under the repo root."""
    root = project_root()
    paths = config.get("paths", {})
    return {
        "workload": root / paths.get("workload_dir", "generated/workload"),
        "topology": root / paths.get("topology_dir", "generated/topology"),
        "system": root / paths.get("system_dir", "generated/system"),
    }
