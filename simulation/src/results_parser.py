"""
Parse Astra-SIM analytical stdout into the §7 metrics contract.

Typical log line:
  [workload] [info] sys[0] finished, 64440 cycles, exposed communication 64440 cycles.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FINISHED_RE = re.compile(
    r"sys\[(?P<sys_id>\d+)\]\s+finished,\s+(?P<total>\d+)\s+cycles,"
    r"\s+exposed communication\s+(?P<exposed>\d+)\s+cycles",
    re.IGNORECASE,
)

# Default Astra-SIM clock: 1 cycle = 1 ns → 1000 cycles = 1 µs
NS_PER_CYCLE = 1.0


def parse_astra_log(log_text: str) -> dict[str, Any]:
    """
    Extract per-NPU finish lines and aggregate §7-style metrics.

    Total Time uses the max finish cycles across NPUs (job completion).
    Exposed communication uses the max exposed cycles (conservative critical-path view).
    """
    rows: list[dict[str, int]] = []
    for match in FINISHED_RE.finditer(log_text):
        rows.append(
            {
                "sys_id": int(match.group("sys_id")),
                "total_cycles": int(match.group("total")),
                "exposed_comm_cycles": int(match.group("exposed")),
            }
        )

    if not rows:
        raise ValueError(
            "No Astra-SIM finish lines found in log. "
            "Expected: 'sys[N] finished, X cycles, exposed communication Y cycles.'"
        )

    total_cycles = max(r["total_cycles"] for r in rows)
    exposed_cycles = max(r["exposed_comm_cycles"] for r in rows)
    # Compute time is not always printed separately; approximate when possible.
    compute_cycles = max(total_cycles - exposed_cycles, 0)

    total_us = cycles_to_us(total_cycles)
    compute_us = cycles_to_us(compute_cycles)
    exposed_us = cycles_to_us(exposed_cycles)

    return {
        "npu_rows": rows,
        "total_cycles": total_cycles,
        "compute_cycles_approx": compute_cycles,
        "exposed_comm_cycles": exposed_cycles,
        "total_time_us": total_us,
        "compute_time_us": compute_us,
        "exposed_communication_time_us": exposed_us,
        "comm_fraction_of_jct": round(exposed_us / total_us, 6) if total_us else None,
    }


def cycles_to_us(cycles: int, ns_per_cycle: float = NS_PER_CYCLE) -> float:
    return (cycles * ns_per_cycle) / 1000.0


def expected_message_size_mb(workload_doc: dict[str, Any]) -> dict[str, Any]:
    """
    Sum expected collective payloads from generator metadata.

    Prefers `expected_collectives_per_step` (works for MoE and LLM-v1).
    Falls back to payload fields for older MoE docs.
    """
    collectives = workload_doc.get("expected_collectives_per_step")
    if isinstance(collectives, list) and collectives:
        per_step_mib = sum(float(item.get("size_mib", 0)) for item in collectives)
    else:
        payloads = workload_doc.get("expected_payloads_mib", {})
        per_step_mib = (
            float(payloads.get("activation", 0))
            + 2.0 * float(payloads.get("moe_routed_aggregate", 0))
            + float(payloads.get("dp_gradient_bucket", 0))
        )
    steps = int(workload_doc.get("training_steps", 1))
    total_mib = per_step_mib * steps
    return {
        "total_message_size_mib": total_mib,
        "total_message_size_mb_approx": total_mib,
        "per_step_message_size_mib": per_step_mib,
        "training_steps": steps,
    }


def build_results_document(
    *,
    run_name: str,
    log_text: str,
    workload_doc: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    parsed = parse_astra_log(log_text)
    msg = expected_message_size_mb(workload_doc)

    doc = {
        "name": run_name,
        "metrics": {
            "Total Time (us)": parsed["total_time_us"],
            "Compute Time (us)": parsed["compute_time_us"],
            "Exposed Communication Time (us)": parsed["exposed_communication_time_us"],
            "Total Message Size (MiB, from workload contract)": msg["total_message_size_mib"],
            "Comm fraction of JCT": parsed["comm_fraction_of_jct"],
        },
        "raw": {
            "total_cycles": parsed["total_cycles"],
            "exposed_comm_cycles": parsed["exposed_comm_cycles"],
            "compute_cycles_approx": parsed["compute_cycles_approx"],
            "npu_finish_count": len(parsed["npu_rows"]),
        },
        "message_size_detail": msg,
        "jct_proxy_us": parsed["total_time_us"],
        "notes": (
            "Total Time (us) is the primary JCT proxy. "
            "Compute Time is approximated as total - exposed when Astra-SIM does not "
            "print a separate compute field. Append new fields to REQUIREMENTS.md §7 "
            "if a live run surfaces more stable metrics."
        ),
    }
    if extra:
        doc["extra"] = extra
    return doc


def write_results(doc: dict[str, Any], path: Path) -> Path:
    from .io_util import write_json_replace

    write_json_replace(path, doc)
    return path
