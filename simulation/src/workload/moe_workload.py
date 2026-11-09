"""
Build and write the MoE training-step workload graph.

What this file does:
1. Turn MoEParams into a dependency-ordered list of compute + collective nodes.
2. Write generated/workload/*.json for humans and for Phase C Chakra conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io_util import write_json_replace
from .collectives import (
    compute_node,
    dp_grad_allreduce,
    ep_moe_alltoall,
    tp_activation_allreduce,
)
from .moe_params import MoEParams


def build_moe_step_nodes(params: MoEParams) -> list[dict[str, Any]]:
    """
    One forward+backward-ish MoE step skeleton.

    Order (simplified for readability / scheduling experiments):
      dense compute → TP allreduce(activation)
      → MoE route compute → EP alltoall(dispatch)
      → expert compute → EP alltoall(combine)
      → DP allreduce(grad bucket)
    """
    stub = params.compute_stub_ns
    act = params.payload_bytes("activation")
    moe = params.payload_bytes("moe_routed_aggregate")
    grad = params.payload_bytes("dp_gradient_bucket")

    n0 = compute_node("n0", "dense_ffn_forward", stub, [])
    n1 = tp_activation_allreduce("n1", act, ["n0"])
    n2 = compute_node("n2", "moe_router", stub, ["n1"])
    n3 = ep_moe_alltoall("n3", moe, ["n2"])  # dispatch
    n4 = compute_node("n4", "moe_experts", stub * params.experts_per_ep_rank, ["n3"])
    n5 = ep_moe_alltoall("n5", moe, ["n4"])  # combine
    n6 = compute_node("n6", "dense_ffn_backward", stub, ["n5"])
    n7 = dp_grad_allreduce("n7", grad, ["n6"])
    return [n0, n1, n2, n3, n4, n5, n6, n7]


def build_workload_document(params: MoEParams) -> dict[str, Any]:
    """Full JSON document written to disk (one step repeated training_steps times)."""
    step_nodes = build_moe_step_nodes(params)
    steps: list[dict[str, Any]] = []

    for step_idx in range(params.training_steps):
        prefixed = [_prefix_step(node, step_idx) for node in step_nodes]
        if step_idx > 0:
            # Chain steps: first node of step k depends on last node of step k-1.
            prefixed[0]["depends_on"] = [f"s{step_idx - 1}_n7"]
        steps.append({"step": step_idx, "nodes": prefixed})

    return {
        "format": "moe_rail_workload_v1",
        "note": (
            "Developer-readable MoE graph. Phase C maps COMM_COLL nodes to Chakra ET "
            "collective nodes for Astra-SIM 2.x. Do not invent extra collectives without "
            "updating REQUIREMENTS.md §3 and §7."
        ),
        "run_name": params.run_name,
        "world_size": params.gpus,
        "precision": params.precision,
        "bytes_per_element": params.bytes_per_element,
        "model": {
            "sequence_length": params.sequence_length,
            "hidden_size": params.hidden_size,
            "global_batch_size": params.global_batch_size,
            "microbatch_size": params.microbatch_size,
            "microbatches_per_step": params.microbatches_per_step,
            "parallelism": {
                "tp": params.parallelism.tp,
                "ep": params.parallelism.ep,
                "dp": params.parallelism.dp,
                "pp": params.parallelism.pp,
            },
            "experts": {
                "num_experts": params.num_experts,
                "top_k": params.top_k,
                "experts_per_ep_rank": params.experts_per_ep_rank,
            },
        },
        "expected_payloads_mib": {
            "activation": params.payloads_mib.activation,
            "moe_routed_aggregate": params.payloads_mib.moe_routed_aggregate,
            "dp_gradient_bucket": params.payloads_mib.dp_gradient_bucket,
        },
        "expected_collectives_per_step": [
            {"name": "tp_activation_allreduce", "collective": "ALLREDUCE", "size_mib": params.payloads_mib.activation},
            {"name": "ep_moe_alltoall_dispatch", "collective": "ALLTOALL", "size_mib": params.payloads_mib.moe_routed_aggregate},
            {"name": "ep_moe_alltoall_combine", "collective": "ALLTOALL", "size_mib": params.payloads_mib.moe_routed_aggregate},
            {"name": "dp_gradient_bucket_allreduce", "collective": "ALLREDUCE", "size_mib": params.payloads_mib.dp_gradient_bucket},
        ],
        "training_steps": params.training_steps,
        "steps": steps,
    }


def generate_moe_workload(params: MoEParams, output_dir: Path) -> Path:
    """Validate params already done upstream; write workload JSON; return path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    document = build_workload_document(params)
    out_path = output_dir / "moe_workload.json"
    _write_json(out_path, document)

    summary_path = output_dir / "workload_summary.json"
    _write_json(
        summary_path,
        {
            "run_name": params.run_name,
            "world_size": params.gpus,
            "training_steps": params.training_steps,
            "nodes_per_step": len(build_moe_step_nodes(params)),
            "artifact": str(out_path.name),
            "phase_c_todo": "Convert moe_workload.json COMM_COLL nodes into Chakra ET for Astra-SIM.",
        },
    )
    return out_path


def _prefix_step(node: dict[str, Any], step_idx: int) -> dict[str, Any]:
    """Give each step unique node ids: n0 → s0_n0, and rewrite depends_on."""
    cloned = dict(node)
    cloned["id"] = f"s{step_idx}_{node['id']}"
    cloned["depends_on"] = [f"s{step_idx}_{dep}" for dep in node["depends_on"]]
    return cloned


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_replace(path, payload)
