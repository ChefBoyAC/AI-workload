"""Safe file writes for Docker Desktop bind mounts (avoid macOS EDEADLK)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_text_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_json_replace(path: Path, payload: dict[str, Any]) -> None:
    write_text_replace(path, json.dumps(payload, indent=2) + "\n")
