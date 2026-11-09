"""
Docker/matrix runner: LLM-v1 | MoE-v1  ×  Ring | Rail | Torus  ×  scheduler.

Environment:
  WORKLOAD=llm-v1|moe-v1          (required for matrix mode; default llm-v1)
  TOPOLOGY=ring|rail|torus        (default rail)
  SCHEDULER=nccl|baseline         (default nccl)
  GENERATE=1|0                    regenerate text workload (default 1)
  ASTRA_SIM_HOME                  Astra-SIM checkout (default /opt/astra-sim)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config_loader import load_config, project_root
from .io_util import write_json_replace, write_text_replace
from .results_parser import build_results_document, write_results
from .run_sim import astra_binary, astra_home, convert_text_to_et, env_flag
from .workload.llm_params import llm_params_from_config
from .workload.llm_workload import build_llm_workload_document
from .workload.moe_params import moe_params_from_config
from .workload.moe_workload import build_workload_document
from .workload.text_trace import write_llm_text_workload, write_moe_text_workload


VALID_WORKLOADS = {"llm-v1", "moe-v1"}
VALID_TOPOLOGIES = {"ring", "rail", "torus"}
VALID_SCHEDULERS = {"nccl", "baseline"}

TOPOLOGY_NPUS = {
    "ring": 16,
    "rail": 16,
    "torus": 64,
}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip().lower()


def resolve_matrix_choice() -> tuple[str, str, str]:
    workload = _env("WORKLOAD", "llm-v1")
    topology = _env("TOPOLOGY", "rail")
    scheduler = _env("SCHEDULER", "nccl")  # NCCL is the default scheduler

    if workload not in VALID_WORKLOADS:
        raise ValueError(f"WORKLOAD must be one of {sorted(VALID_WORKLOADS)}, got {workload}")
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(f"TOPOLOGY must be one of {sorted(VALID_TOPOLOGIES)}, got {topology}")
    if scheduler not in VALID_SCHEDULERS:
        raise ValueError(f"SCHEDULER must be one of {sorted(VALID_SCHEDULERS)}, got {scheduler}")
    return workload, topology, scheduler


def experiment_dir(root: Path, workload: str, topology: str) -> Path:
    return root / "experiments" / f"{workload}-{topology}"


def copy_inputs(
    root: Path,
    *,
    topology: str,
    scheduler: str,
    out_topology: Path,
    out_system: Path,
) -> tuple[Path, Path]:
    network_src = root / "topologies" / topology / "network.yml"
    system_src = root / "schedulers" / scheduler / f"{topology}.json"
    remote_src = root / "shared" / "remote_memory.json"

    if not network_src.exists():
        raise FileNotFoundError(f"Missing topology network: {network_src}")
    if not system_src.exists():
        raise FileNotFoundError(f"Missing scheduler system config: {system_src}")
    if not remote_src.exists():
        raise FileNotFoundError(f"Missing remote memory config: {remote_src}")

    out_topology.mkdir(parents=True, exist_ok=True)
    out_system.mkdir(parents=True, exist_ok=True)

    network_dst = out_topology / "network.yml"
    system_dst = out_system / "system.json"
    remote_dst = out_system / "remote_memory.json"

    shutil.copyfile(network_src, network_dst)
    # Strip helper comments Astra-SIM may not accept in JSON.
    system_doc = json.loads(system_src.read_text(encoding="utf-8"))
    system_doc.pop("_comment", None)
    write_json_replace(system_dst, system_doc)
    shutil.copyfile(remote_src, remote_dst)
    return network_dst, system_dst


def write_workload_text(root: Path, workload: str, out_dir: Path) -> tuple[Path, dict]:
    """
    Build Chakra text workload from the Rail YAML knobs (same payloads as matrix params).
    NPU count is applied later at conversion time (16 vs 64).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if workload == "llm-v1":
        config = load_config(root / "config" / "llm_v1_rail.yaml")
        params = llm_params_from_config(config)
        text_path = write_llm_text_workload(params, out_dir)
        doc = build_llm_workload_document(params)
        write_json_replace(out_dir / "llm_v1_workload.json", doc)
        return text_path, doc

    config = load_config(root / "config" / "moe_rail.yaml")
    params = moe_params_from_config(config)
    text_path = write_moe_text_workload(params, out_dir)
    doc = build_workload_document(params)
    write_json_replace(out_dir / "moe_workload.json", doc)
    return text_path, doc


def run_astra(
    *,
    binary: Path,
    workload_prefix: Path,
    system: Path,
    network: Path,
    remote_memory: Path,
    log_path: Path,
) -> str:
    cmd = [
        str(binary),
        f"--workload-configuration={workload_prefix}",
        f"--system-configuration={system}",
        f"--network-configuration={network}",
        f"--remote-memory-configuration={remote_memory}",
    ]
    print("Running Astra-SIM:")
    print(" ", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    write_text_replace(log_path, combined)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Astra-SIM exited with code {completed.returncode}. See {log_path}"
        )
    return combined


def main() -> int:
    root = project_root()
    workload, topology, scheduler = resolve_matrix_choice()
    npus = TOPOLOGY_NPUS[topology]
    run_name = f"{workload}_{topology}_{scheduler}"

    print("=== ASTRA-sim matrix run ===")
    print(f"WORKLOAD={workload}")
    print(f"TOPOLOGY={topology} (npus={npus})")
    print(f"SCHEDULER={scheduler}  (default is nccl)")
    print(f"GENERATE={env_flag('GENERATE', '1')}")

    generated = root / "generated"
    wl_dir = generated / "workload"
    topo_dir = generated / "topology"
    sys_dir = generated / "system"
    for d in (wl_dir, topo_dir, sys_dir):
        d.mkdir(parents=True, exist_ok=True)

    network, system = copy_inputs(
        root,
        topology=topology,
        scheduler=scheduler,
        out_topology=topo_dir,
        out_system=sys_dir,
    )
    remote_memory = sys_dir / "remote_memory.json"

    if env_flag("GENERATE", "1"):
        text_path, workload_doc = write_workload_text(root, workload, wl_dir)
    else:
        candidate = (
            wl_dir / "llm_v1_workload.txt"
            if workload == "llm-v1"
            else wl_dir / "moe_workload.txt"
        )
        if not candidate.exists():
            raise FileNotFoundError(f"GENERATE=0 but missing {candidate}")
        text_path = candidate
        json_candidate = (
            wl_dir / "llm_v1_workload.json"
            if workload == "llm-v1"
            else wl_dir / "moe_workload.json"
        )
        workload_doc = json.loads(json_candidate.read_text(encoding="utf-8"))

    et_prefix = wl_dir / "workload_et"
    convert_text_to_et(text_path, et_prefix, npus)

    home = astra_home()
    binary = astra_binary(home)
    log_path = generated / "astra_sim.log"
    log_text = run_astra(
        binary=binary,
        workload_prefix=et_prefix,
        system=system,
        network=network,
        remote_memory=remote_memory,
        log_path=log_path,
    )

    results = build_results_document(
        run_name=run_name,
        log_text=log_text,
        workload_doc=workload_doc,
        extra={
            "workload": workload,
            "topology": topology,
            "scheduler": scheduler,
            "npus": npus,
            "experiment": str(experiment_dir(root, workload, topology)),
            "system_config": str(system),
            "network_config": str(network),
            "astra_sim_bin": str(binary),
            "log_file": str(log_path),
        },
    )
    results_path = write_results(results, generated / "results.json")

    print("\n=== Parsed results (§7) ===")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")
    print(f"  JCT proxy (us): {results['jct_proxy_us']}")
    print(f"Wrote {results_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
