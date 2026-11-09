"""
Build and write the dense LLM-v1 training-step workload graph.

Skeleton (agreed):
  dense forward → TP AllReduce(activation)
  → dense backward → DP AllReduce(grad bucket)

No experts / no EP AllToAll.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io_util import write_json_replace
from .collectives import compute_node, dp_grad_allreduce, tp_activation_allreduce
from .llm_params import LlmParams


def build_llm_step_nodes(params: LlmParams) -> list[dict[str, Any]]:
    stub = params.compute_stub_ns
    act = params.payload_bytes("activation")
    grad = params.payload_bytes("dp_gradient_bucket")

    n0 = compute_node("n0", "dense_forward", stub, [])
    n1 = tp_activation_allreduce("n1", act, ["n0"])
    n2 = compute_node("n2", "dense_backward", stub, ["n1"])
    n3 = dp_grad_allreduce("n3", grad, ["n2"])
    return [n0, n1, n2, n3]


def build_llm_workload_document(params: LlmParams) -> dict[str, Any]:
    step_nodes = build_llm_step_nodes(params)
    steps: list[dict[str, Any]] = []

    for step_idx in range(params.training_steps):
        prefixed = [_prefix_step(node, step_idx) for node in step_nodes]
        if step_idx > 0:
            prefixed[0]["depends_on"] = [f"s{step_idx - 1}_n3"]
        steps.append({"step": step_idx, "nodes": prefixed})

    return {
        "format": "llm_v1_rail_workload_v1",
        "note": (
            "Developer-readable dense LLM-v1 graph (TP+DP AllReduce only). "
            "Phase C maps COMM_COLL nodes to Chakra ET for Astra-SIM 2.x."
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
                "dp": params.parallelism.dp,
                "pp": params.parallelism.pp,
            },
            "experts": None,
        },
        "expected_payloads_mib": {
            "activation": params.payloads_mib.activation,
            "dp_gradient_bucket": params.payloads_mib.dp_gradient_bucket,
        },
        "expected_collectives_per_step": [
            {
                "name": "tp_activation_allreduce",
                "collective": "ALLREDUCE",
                "size_mib": params.payloads_mib.activation,
            },
            {
                "name": "dp_gradient_bucket_allreduce",
                "collective": "ALLREDUCE",
                "size_mib": params.payloads_mib.dp_gradient_bucket,
            },
        ],
        "training_steps": params.training_steps,
        "steps": steps,
    }


def generate_llm_workload(params: LlmParams, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    document = build_llm_workload_document(params)
    out_path = output_dir / "llm_v1_workload.json"
    _write_json(out_path, document)
    _write_json(
        output_dir / "workload_summary.json",
        {
            "run_name": params.run_name,
            "workload_type": "llm_v1",
            "world_size": params.gpus,
            "training_steps": params.training_steps,
            "nodes_per_step": len(build_llm_step_nodes(params)),
            "artifact": out_path.name,
        },
    )
    return out_path


def _prefix_step(node: dict[str, Any], step_idx: int) -> dict[str, Any]:
    cloned = dict(node)
    cloned["id"] = f"s{step_idx}_{node['id']}"
    cloned["depends_on"] = [f"s{step_idx}_{dep}" for dep in node["depends_on"]]
    return cloned


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json_replace(path, payload)
