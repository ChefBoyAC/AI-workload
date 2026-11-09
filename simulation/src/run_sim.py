"""
Phase C orchestrator: generate (optional) → Chakra convert → Astra-SIM → results.json.

Environment:
  GENERATE=1|0          regenerate artifacts (default 1)
  ASTRA_SIM_HOME        path to astra-sim checkout
  ASTRA_SIM_BIN         analytical binary (optional override)
  CONFIG_PATH           path to moe_rail.yaml or llm_v1_rail.yaml
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .config_loader import default_config_path, project_root
from .generate import generate_all
from .io_util import write_text_replace
from .results_parser import build_results_document, write_results


def env_flag(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip() not in {"0", "false", "False", "no", "NO"}


def astra_home() -> Path:
    override = os.environ.get("ASTRA_SIM_HOME")
    if override:
        return Path(override)
    return Path("/opt/astra-sim")


def astra_binary(home: Path) -> Path:
    override = os.environ.get("ASTRA_SIM_BIN")
    if override:
        return Path(override)
    unaware = home / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"
    aware = home / "build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
    if unaware.exists():
        return unaware
    if aware.exists():
        return aware
    raise FileNotFoundError(
        f"Astra-SIM analytical binary not found under {home}. "
        "Build with ./build/astra_analytical/build.sh or set ASTRA_SIM_BIN."
    )


def convert_text_to_et(text_path: Path, et_prefix: Path, num_npus: int) -> Path:
    """
    Prefer `chakra_converter Text` (pip-installed Chakra). Fall back to python -m.
    Output files: {et_prefix}.0.et … {et_prefix}.{N-1}.et
    """
    et_prefix.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale ETs for this prefix.
    for stale in et_prefix.parent.glob(f"{et_prefix.name}.*.et"):
        stale.unlink()

    cmd_primary = [
        "chakra_converter",
        "Text",
        "--input",
        str(text_path),
        "--output",
        str(et_prefix),
        "--num-npus",
        str(num_npus),
        "--num-passes",
        "1",
    ]
    cmd_fallback = [
        sys.executable,
        "-m",
        "chakra.et_converter.et_converter",
        "--input_type",
        "Text",
        "--input_filename",
        str(text_path),
        "--output_filename",
        str(et_prefix),
        "--num_npus",
        str(num_npus),
        "--num_passes",
        "1",
    ]

    if shutil.which("chakra_converter"):
        _run(cmd_primary, label="chakra_converter Text")
    else:
        _run(cmd_fallback, label="python -m chakra.et_converter")

    expected = et_prefix.parent / f"{et_prefix.name}.0.et"
    if not expected.exists():
        raise FileNotFoundError(f"Chakra conversion produced no ET at {expected}")
    return et_prefix


def run_astra_sim(
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running Astra-SIM:")
    print(" ", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    combined = (completed.stdout or "") + "\n" + (completed.stderr or "")
    # Write via temp+replace to avoid Docker Desktop volume deadlocks on macOS.
    write_text_replace(log_path, combined)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Astra-SIM exited with code {completed.returncode}. See {log_path}"
        )
    return combined


def main() -> int:
    root = project_root()
    config_path = Path(os.environ.get("CONFIG_PATH", default_config_path()))
    generated = root / "generated"

    if env_flag("GENERATE", "1"):
        print("GENERATE=1 → regenerating workload + topology + system")
        manifest = generate_all(config_path)
    else:
        print("GENERATE=0 → reusing existing generated/ artifacts")
        manifest_path = generated / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"{manifest_path} missing; run with GENERATE=1 once first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    workload_json = Path(manifest["workload"])
    text_path = Path(manifest.get("workload_text", generated / "workload" / "moe_workload.txt"))
    network = Path(manifest["network"])
    system = Path(manifest["system"])
    remote_memory = Path(
        manifest.get("remote_memory", generated / "system" / "remote_memory.json")
    )
    world_size = int(manifest["world_size"])
    run_name = str(manifest.get("run_name", "moe_rail_baseline"))
    workload_type = str(manifest.get("workload_type", "moe"))

    if not text_path.exists():
        raise FileNotFoundError(f"Missing text workload: {text_path}")

    # Generic ET prefix; generated/ is overwritten per run to save storage.
    et_prefix = generated / "workload" / "workload_et"
    print(f"workload_type={workload_type} run_name={run_name}")
    convert_text_to_et(text_path, et_prefix, world_size)

    home = astra_home()
    binary = astra_binary(home)
    log_path = generated / "astra_sim.log"
    log_text = run_astra_sim(
        binary=binary,
        workload_prefix=et_prefix,
        system=system,
        network=network,
        remote_memory=remote_memory,
        log_path=log_path,
    )

    workload_doc = json.loads(workload_json.read_text(encoding="utf-8"))
    results = build_results_document(
        run_name=run_name,
        log_text=log_text,
        workload_doc=workload_doc,
        extra={
            "workload_type": workload_type,
            "astra_sim_home": str(home),
            "astra_sim_bin": str(binary),
            "generate": env_flag("GENERATE", "1"),
            "log_file": str(log_path),
            "workload_et_prefix": str(et_prefix),
        },
    )
    results_path = write_results(results, generated / "results.json")

    print("\n=== Parsed results (§7) ===")
    for key, value in results["metrics"].items():
        print(f"  {key}: {value}")
    print(f"  JCT proxy (us): {results['jct_proxy_us']}")
    print(f"Wrote {results_path}")
    return 0


def _run(cmd: list[str], label: str) -> None:
    print(f"Running {label}:")
    print(" ", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        sys.stderr.write(completed.stdout or "")
        sys.stderr.write(completed.stderr or "")
        raise RuntimeError(f"{label} failed with code {completed.returncode}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — surface clear failure in Docker logs
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
