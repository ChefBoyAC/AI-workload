"""
ns-3 network backend helpers for the Progress Report 2 matrix.

Astra-SIM's ns-3 frontend needs:
  - a physical topology .txt (nodes / switches / links)
  - a congestion-control config.txt that points at that topology
  - a logical-dims JSON (must match system.json dim count)
  - the AstraSimNetwork binary under extern/network_backend/ns-3/build/scratch/

All matrix cells (baseline + NCCL + future schedulers) share this fabric.
Only system.json differs between scheduler columns.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def ns3_root(astra_home: Path) -> Path:
    return astra_home / "extern" / "network_backend" / "ns-3"


def ns3_binary(astra_home: Path) -> Path:
    """Locate the built AstraSimNetwork binary (version suffix may vary)."""
    override = __import__("os").environ.get("ASTRA_SIM_NS3_BIN")
    if override:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"ASTRA_SIM_NS3_BIN not found: {path}")
        return path

    scratch = ns3_root(astra_home) / "build" / "scratch"
    if scratch.exists():
        matches = sorted(scratch.glob("ns3*-AstraSimNetwork*"))
        # Prefer the default (non-debug) binary when several exist.
        preferred = [m for m in matches if "default" in m.name and m.is_file()]
        if preferred:
            return preferred[0]
        executables = [m for m in matches if m.is_file() and m.stat().st_mode & 0o111]
        if executables:
            return executables[0]

    raise FileNotFoundError(
        f"ns-3 AstraSimNetwork binary not found under {scratch}. "
        "Build with ./build/astra_ns3/build.sh or set ASTRA_SIM_NS3_BIN."
    )


def prepare_ns3_inputs(
    *,
    root: Path,
    astra_home: Path,
    topology: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Stage physical topology + config into the ns-3 scratch tree and copy
    logical.json into generated/ for the runner CLI.

    Returns (network_config_path, logical_topology_path).
    """
    src_dir = root / "backends" / "ns3" / topology
    physical_src = src_dir / "physical_topology.txt"
    config_src = src_dir / "config.txt"
    logical_src = src_dir / "logical.json"

    for path in (physical_src, config_src, logical_src):
        if not path.exists():
            raise FileNotFoundError(f"Missing ns-3 asset: {path}")

    ns3 = ns3_root(astra_home)
    topo_dir = ns3 / "scratch" / "topology"
    cfg_dir = ns3 / "scratch" / "config"
    out_ns3 = ns3 / "scratch" / "output"
    for d in (topo_dir, cfg_dir, out_ns3):
        d.mkdir(parents=True, exist_ok=True)

    physical_name = f"matrix_{topology}_16.txt"
    physical_dst = topo_dir / physical_name
    shutil.copyfile(physical_src, physical_dst)

    # Keep a workspace-local copy too (visible on the bind mount).
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(physical_src, out_dir / "physical_topology.txt")

    config_dst = cfg_dir / f"matrix_{topology}_16_config.txt"
    shutil.copyfile(config_src, config_dst)
    shutil.copyfile(config_src, out_dir / "config.txt")

    logical_dst = out_dir / "logical.json"
    shutil.copyfile(logical_src, logical_dst)

    # ns-3 SetupNetwork() opens FLOW_FILE and TRACE_FILE for *read* at boot.
    # Astra-SIM drives traffic itself, so these are stubs: a leading 0 count.
    # Paths in config.txt are relative to cwd = ns-3/build/scratch.
    _ensure_ns3_boot_stubs(config_src=config_src, scratch_cwd=ns3 / "build" / "scratch")

    return config_dst, logical_dst


def _ensure_ns3_boot_stubs(*, config_src: Path, scratch_cwd: Path) -> None:
    """Create empty FLOW_FILE / TRACE_FILE inputs ns-3 requires at startup."""
    keys = {"FLOW_FILE", "TRACE_FILE"}
    resolved: dict[str, Path] = {}
    for line in config_src.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, raw = parts[0].strip(), parts[1].strip()
        if key not in keys:
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = (scratch_cwd / path).resolve()
        resolved[key] = path

    for key in keys:
        if key not in resolved:
            raise FileNotFoundError(f"{config_src} missing required {key}")
        path = resolved[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Official Astra-SIM test inputs use a single "0" (zero entries).
        path.write_text("0\n", encoding="utf-8")


def run_ns3_cmd(
    *,
    binary: Path,
    workload_prefix: Path,
    system: Path,
    network_config: Path,
    remote_memory: Path,
    logical_topology: Path,
    comm_group: str = "empty",
) -> list[str]:
    """CLI matching examples/run_scripts/ns3/*.sh."""
    return [
        str(binary),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system}",
        f"--network-configuration={network_config}",
        f"--remote-memory-configuration={remote_memory}",
        f"--logical-topology-configuration={logical_topology}",
        f"--comm-group-configuration={comm_group}",
    ]
