"""
Emit Astra-SIM analytical network.yml + baseline system.json.

Current analytical schema (astra-sim examples/network/analytical style):

  topology: [ FullyConnected, Switch ]
  npus_count: [ 8, 2 ]
  bandwidth: [ 300.0, 50.0 ]   # GB/s
  latency: [ 100.0, 500.0 ]    # ns
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io_util import write_json_replace, write_text_replace
from .rail import RailTopology


def build_network_yaml_text(rail: RailTopology) -> str:
    """Serialize analytical network config as YAML text (no extra dependency quirks)."""
    topologies = [rail.intra_node_topology, rail.inter_node_topology]
    lines = [
        "# Auto-generated Rail analytical network for Astra-SIM.",
        "# dim0 = intra-node, dim1 = inter-node rail.",
        f"topology: [{_csv_strings(topologies)}]",
        f"npus_count: [{_csv_ints(rail.logical_dims)}]",
        f"bandwidth: [{_csv_floats(rail.bandwidth_gbs)}]  # GB/s",
        f"latency: [{_csv_floats(rail.latency_ns)}]  # ns",
        "",
    ]
    return "\n".join(lines)


def build_baseline_system(config: dict[str, Any], rail: RailTopology) -> dict[str, Any]:
    """
    Default collective scheduler knobs.

    This is Astra-SIM's built-in scheduler config — not TACCL/ByteScheduler/etc.
    Dimensional collective lists MUST match len(logical_dims).
    """
    sys_cfg = config["system"]
    dims = len(rail.logical_dims)

    def _dim_list(key: str) -> list[str]:
        values = list(sys_cfg[key])
        if len(values) != dims:
            raise ValueError(f"system.{key} must have {dims} entries (one per logical dim)")
        return [str(v) for v in values]

    return {
        "scheduling-policy": sys_cfg["scheduling_policy"],
        "intra-dimension-scheduling": sys_cfg["intra_dimension_scheduling"],
        "inter-dimension-scheduling": sys_cfg["inter_dimension_scheduling"],
        "endpoint-delay": int(sys_cfg["endpoint_delay"]),
        "active-chunks-per-dimension": int(sys_cfg["active_chunks_per_dimension"]),
        "preferred-dataset-splits": int(sys_cfg["preferred_dataset_splits"]),
        "all-reduce-implementation": _dim_list("all_reduce_implementation"),
        "all-gather-implementation": _dim_list("all_gather_implementation"),
        "reduce-scatter-implementation": _dim_list("reduce_scatter_implementation"),
        "all-to-all-implementation": _dim_list("all_to_all_implementation"),
        "collective-optimization": sys_cfg["collective_optimization"],
        "local-mem-bw": 50,
        "boost-mode": 0,
        "_comment": (
            "Baseline Astra-SIM system scheduler. Replace this file in Phase D to "
            "approximate NCCL/TACCL/ByteScheduler/SYNDICATE/CadentFlow behavior."
        ),
    }


def generate_analytical_network(rail: RailTopology, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    network_path = output_dir / "network.yml"
    write_text_replace(network_path, build_network_yaml_text(rail))

    meta_path = output_dir / "rail_metadata.json"
    _write_json(
        meta_path,
        {
            "format": "moe_rail_topology_v1",
            "total_gpus": rail.total_gpus,
            "layout": rail.rank_layout(),
            "analytical_network_file": network_path.name,
            "developer_note": (
                "Expect Astra-SIM analytical backend to treat physical==logical dims. "
                "npus_count product must equal workload world_size (16)."
            ),
        },
    )
    return network_path


def generate_baseline_system(
    config: dict[str, Any],
    rail: RailTopology,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    system_doc = build_baseline_system(config, rail)
    # Strip helper comment key if we want pure Astra input; keep a sidecar instead.
    comment = system_doc.pop("_comment", None)
    system_path = output_dir / "system.json"
    _write_json(system_path, system_doc)

    if comment:
        _write_json(
            output_dir / "system_notes.json",
            {"baseline_scheduler_note": comment, "system_file": system_path.name},
        )
    return system_path


def _csv_strings(values: list[str]) -> str:
    return ", ".join(values)


def _csv_ints(values: tuple[int, ...]) -> str:
    return ", ".join(str(v) for v in values)


def _csv_floats(values: tuple[float, ...]) -> str:
    return ", ".join(str(float(v)) for v in values)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_replace(path, payload)
