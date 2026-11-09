"""Emit analytical remote-memory config (no expansion)."""

from __future__ import annotations

from pathlib import Path

from ..io_util import write_json_replace


def generate_remote_memory(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "remote_memory.json"
    write_json_replace(path, {"memory-type": "NO_MEMORY_EXPANSION"})
    return path
