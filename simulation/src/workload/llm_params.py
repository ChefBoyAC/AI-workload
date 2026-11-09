"""Validated dense LLM-v1 training parameters (no MoE / no EP)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .moe_params import BYTES_PER_ELEMENT, _require


@dataclass(frozen=True)
class LlmParallelism:
    tp: int
    dp: int
    pp: int

    @property
    def world_size(self) -> int:
        return self.tp * self.dp * self.pp


@dataclass(frozen=True)
class LlmPayloadSizesMiB:
    activation: int
    dp_gradient_bucket: int


@dataclass(frozen=True)
class LlmParams:
    """Dense LLM-v1 knobs used by the workload graph builder."""

    gpus: int
    precision: str
    sequence_length: int
    hidden_size: int
    global_batch_size: int
    microbatch_size: int
    parallelism: LlmParallelism
    payloads_mib: LlmPayloadSizesMiB
    training_steps: int
    compute_stub_ns: int
    run_name: str

    def validate(self) -> None:
        _require(self.gpus > 0, "gpus must be > 0")
        _require(
            self.parallelism.world_size == self.gpus,
            f"TP*DP*PP={self.parallelism.world_size} must equal gpus={self.gpus}",
        )
        _require(
            self.global_batch_size % self.microbatch_size == 0,
            "global_batch_size must be divisible by microbatch_size",
        )
        _require(self.precision.upper() in BYTES_PER_ELEMENT, f"unsupported precision: {self.precision}")
        _require(self.parallelism.pp == 1, "Phase B/C assumes PP=1 (no pipeline edges)")
        _require(self.training_steps >= 1, "training_steps must be >= 1")

    @property
    def microbatches_per_step(self) -> int:
        return self.global_batch_size // self.microbatch_size

    @property
    def bytes_per_element(self) -> int:
        return BYTES_PER_ELEMENT[self.precision.upper()]

    def payload_bytes(self, name: str) -> int:
        mib = getattr(self.payloads_mib, name)
        return int(mib) * 1024 * 1024


def llm_params_from_config(config: dict[str, Any]) -> LlmParams:
    llm = config["llm"]
    run = config.get("run", {})
    parity = llm["parallelism"]
    payloads = llm["payloads_mib"]

    params = LlmParams(
        gpus=int(llm["gpus"]),
        precision=str(llm["precision"]),
        sequence_length=int(llm["sequence_length"]),
        hidden_size=int(llm["hidden_size"]),
        global_batch_size=int(llm["global_batch_size"]),
        microbatch_size=int(llm["microbatch_size"]),
        parallelism=LlmParallelism(
            tp=int(parity["tp"]),
            dp=int(parity["dp"]),
            pp=int(parity["pp"]),
        ),
        payloads_mib=LlmPayloadSizesMiB(
            activation=int(payloads["activation"]),
            dp_gradient_bucket=int(payloads["dp_gradient_bucket"]),
        ),
        training_steps=int(run.get("training_steps", 1)),
        compute_stub_ns=int(run.get("compute_stub_ns", 1000)),
        run_name=str(run.get("name", "llm_v1_rail_baseline")),
    )
    params.validate()
    return params
