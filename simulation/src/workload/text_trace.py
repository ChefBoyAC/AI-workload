"""
Convert MoE / LLM params into an ASTRA-sim 1.0 text workload.

Phase C uses Chakra's Text converter → per-NPU `.et` files
  (workload_et.0.et … workload_et.15.et).
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..io_util import write_text_replace
from .llm_params import LlmParams
from .moe_params import MoEParams


class _HasStubAndSteps(Protocol):
    compute_stub_ns: int
    training_steps: int


def build_moe_text_workload(params: MoEParams) -> str:
    """MoE layers: TP AR → EP A2A dispatch → EP A2A combine → DP AR."""
    stub = max(params.compute_stub_ns, 1)
    act = params.payload_bytes("activation")
    moe = params.payload_bytes("moe_routed_aggregate")
    grad = params.payload_bytes("dp_gradient_bucket")
    expert_compute = stub * params.experts_per_ep_rank

    layers = [
        _layer("tp_activation", fwd_comp=stub, fwd_type="ALLREDUCE", fwd_size=act),
        _layer("moe_dispatch", fwd_comp=stub, fwd_type="ALLTOALL", fwd_size=moe),
        _layer("moe_combine", fwd_comp=expert_compute, fwd_type="ALLTOALL", fwd_size=moe),
        _layer(
            "dp_grad_bucket",
            fwd_comp=stub,
            fwd_type="NONE",
            fwd_size=0,
            wg_comp=stub,
            wg_type="ALLREDUCE",
            wg_size=grad,
        ),
    ]
    return _format_data_workload(layers, params.training_steps)


def build_llm_text_workload(params: LlmParams) -> str:
    """LLM-v1 layers: TP AllReduce(activation) → DP AllReduce(grad)."""
    stub = max(params.compute_stub_ns, 1)
    act = params.payload_bytes("activation")
    grad = params.payload_bytes("dp_gradient_bucket")

    layers = [
        _layer("tp_activation", fwd_comp=stub, fwd_type="ALLREDUCE", fwd_size=act),
        _layer(
            "dp_grad_bucket",
            fwd_comp=stub,
            fwd_type="NONE",
            fwd_size=0,
            wg_comp=stub,
            wg_type="ALLREDUCE",
            wg_size=grad,
        ),
    ]
    return _format_data_workload(layers, params.training_steps)


def write_moe_text_workload(params: MoEParams, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "moe_workload.txt"
    write_text_replace(path, build_moe_text_workload(params))
    return path


def write_llm_text_workload(params: LlmParams, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "llm_v1_workload.txt"
    write_text_replace(path, build_llm_text_workload(params))
    return path


# Back-compat alias used by older generate imports.
def write_text_workload(params: MoEParams, output_dir: Path) -> Path:
    return write_moe_text_workload(params, output_dir)


def _format_data_workload(layers: list[str], training_steps: int) -> str:
    """
    Emit Chakra text with HYBRID_DATA_MODEL so forward collectives are kept.

    Plain DATA mode only converts weight-gradient (WG) communication, which
    zeroed out TP AllReduce / MoE AllToAll and left only the DP gradient AR.
    HYBRID_DATA_MODEL keeps forward + WG collectives for the matrix baseline.
    """
    expanded = layers * training_steps
    lines = ["HYBRID_DATA_MODEL", str(len(expanded)), *expanded]
    return "\n".join(lines) + "\n"


def _layer(
    name: str,
    *,
    fwd_comp: int,
    fwd_type: str,
    fwd_size: int,
    ig_comp: int = 1,
    ig_type: str = "NONE",
    ig_size: int = 0,
    wg_comp: int = 1,
    wg_type: str = "NONE",
    wg_size: int = 0,
    delay: int = 0,
) -> str:
    """One layer line; fields must stay on a single line for the converter."""
    return (
        f"{name} -1 {fwd_comp} {fwd_type} {fwd_size} "
        f"{ig_comp} {ig_type} {ig_size} "
        f"{wg_comp} {wg_type} {wg_size} {delay}"
    )
