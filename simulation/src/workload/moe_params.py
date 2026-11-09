"""Validated MoE training parameters (dataclass + checks)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


BYTES_PER_ELEMENT = {
    "BF16": 2,
    "FP16": 2,
    "FP32": 4,
}


@dataclass(frozen=True)
class Parallelism:
    tp: int
    ep: int
    dp: int
    pp: int

    @property
    def world_size(self) -> int:
        return self.tp * self.ep * self.dp * self.pp


@dataclass(frozen=True)
class PayloadSizesMiB:
    activation: int
    moe_routed_aggregate: int
    dp_gradient_bucket: int


@dataclass(frozen=True)
class MoEParams:
    """
    Source-of-truth MoE knobs used by the workload graph builder.

    Keeping validation here means emitters can assume invariants already hold.
    """

    gpus: int
    precision: str
    sequence_length: int
    hidden_size: int
    global_batch_size: int
    microbatch_size: int
    parallelism: Parallelism
    num_experts: int
    top_k: int
    payloads_mib: PayloadSizesMiB
    training_steps: int
    compute_stub_ns: int
    run_name: str

    def validate(self) -> None:
        _require(self.gpus > 0, "gpus must be > 0")
        _require(
            self.parallelism.world_size == self.gpus,
            f"TP*EP*DP*PP={self.parallelism.world_size} must equal gpus={self.gpus}",
        )
        _require(
            self.global_batch_size % self.microbatch_size == 0,
            "global_batch_size must be divisible by microbatch_size",
        )
        _require(self.num_experts % self.parallelism.ep == 0, "num_experts must divide by EP")
        _require(self.top_k >= 1, "top_k must be >= 1")
        _require(self.top_k <= self.num_experts, "top_k cannot exceed num_experts")
        _require(self.precision.upper() in BYTES_PER_ELEMENT, f"unsupported precision: {self.precision}")
        _require(self.parallelism.pp == 1, "Phase B assumes PP=1 (no pipeline edges)")
        _require(self.training_steps >= 1, "training_steps must be >= 1")

    @property
    def microbatches_per_step(self) -> int:
        return self.global_batch_size // self.microbatch_size

    @property
    def experts_per_ep_rank(self) -> int:
        return self.num_experts // self.parallelism.ep

    @property
    def bytes_per_element(self) -> int:
        return BYTES_PER_ELEMENT[self.precision.upper()]

    def payload_bytes(self, name: str) -> int:
        mib = getattr(self.payloads_mib, name)
        return int(mib) * 1024 * 1024


def moe_params_from_config(config: dict[str, Any]) -> MoEParams:
    """Map nested YAML into a validated MoEParams instance."""
    moe = config["moe"]
    run = config.get("run", {})
    parity = moe["parallelism"]
    payloads = moe["payloads_mib"]
    experts = moe["experts"]

    params = MoEParams(
        gpus=int(moe["gpus"]),
        precision=str(moe["precision"]),
        sequence_length=int(moe["sequence_length"]),
        hidden_size=int(moe["hidden_size"]),
        global_batch_size=int(moe["global_batch_size"]),
        microbatch_size=int(moe["microbatch_size"]),
        parallelism=Parallelism(
            tp=int(parity["tp"]),
            ep=int(parity["ep"]),
            dp=int(parity["dp"]),
            pp=int(parity["pp"]),
        ),
        num_experts=int(experts["num_experts"]),
        top_k=int(experts["top_k"]),
        payloads_mib=PayloadSizesMiB(
            activation=int(payloads["activation"]),
            moe_routed_aggregate=int(payloads["moe_routed_aggregate"]),
            dp_gradient_bucket=int(payloads["dp_gradient_bucket"]),
        ),
        training_steps=int(run.get("training_steps", 1)),
        compute_stub_ns=int(run.get("compute_stub_ns", 1000)),
        run_name=str(run.get("name", "moe_rail_baseline")),
    )
    params.validate()
    return params


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
