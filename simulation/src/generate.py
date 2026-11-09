"""
Generate workload + Rail topology (+ baseline system / remote memory / text trace).

Usage (from repo root):
  python -m src.generate --config config/moe_rail.yaml
  python -m src.generate --config config/llm_v1_rail.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config_loader import default_config_path, load_config, resolve_output_dirs
from .io_util import write_json_replace
from .topology import (
    generate_analytical_network,
    generate_baseline_system,
    generate_remote_memory,
    rail_from_config,
)
from .workload import (
    generate_llm_workload,
    generate_moe_workload,
    llm_params_from_config,
    moe_params_from_config,
    write_llm_text_workload,
    write_moe_text_workload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MoE or LLM-v1 workload + Rail topology for Astra-SIM."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to YAML config (default: {default_config_path()})",
    )
    return parser.parse_args()


def detect_workload_type(config: dict[str, Any]) -> str:
    explicit = str(config.get("workload_type", "")).strip().lower()
    if explicit in {"moe", "llm_v1", "llm"}:
        return "llm_v1" if explicit == "llm" else explicit
    if "llm" in config:
        return "llm_v1"
    if "moe" in config:
        return "moe"
    raise ValueError("Config must set workload_type or include a moe:/llm: block")


def generate_all(config_path: Path | None = None) -> dict[str, str]:
    """Load config → validate → write artifacts (overwrites generated/)."""
    config = load_config(config_path)
    workload_type = detect_workload_type(config)
    rail = rail_from_config(config)
    dirs = resolve_output_dirs(config)

    if workload_type == "moe":
        params = moe_params_from_config(config)
        workload_path = generate_moe_workload(params, dirs["workload"])
        text_path = write_moe_text_workload(params, dirs["workload"])
        run_name = params.run_name
        world_size = params.gpus
    elif workload_type == "llm_v1":
        params = llm_params_from_config(config)
        workload_path = generate_llm_workload(params, dirs["workload"])
        text_path = write_llm_text_workload(params, dirs["workload"])
        run_name = params.run_name
        world_size = params.gpus
    else:
        raise ValueError(f"Unsupported workload_type: {workload_type}")

    network_path = generate_analytical_network(rail, dirs["topology"])
    system_path = generate_baseline_system(config, rail, dirs["system"])
    remote_memory_path = generate_remote_memory(dirs["system"])

    manifest = {
        "run_name": run_name,
        "workload_type": workload_type,
        "config": config.get("_config_path", ""),
        "workload": str(workload_path),
        "workload_text": str(text_path),
        "network": str(network_path),
        "system": str(system_path),
        "remote_memory": str(remote_memory_path),
        "world_size": world_size,
        "logical_dims": list(rail.logical_dims),
        "generate_mode_note": (
            "Docker entrypoint: GENERATE=1 regenerates/overwrites generated/; "
            "GENERATE=0 reuses them for scheduler comparisons."
        ),
    }

    manifest_dir = dirs["workload"].parent
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.json"
    write_json_replace(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate_all(args.config)
    print("Generation complete:")
    print(f"  workload_type: {manifest['workload_type']}")
    for key in ("workload", "workload_text", "network", "system", "remote_memory", "manifest"):
        print(f"  {key}: {manifest[key]}")


if __name__ == "__main__":
    main()
