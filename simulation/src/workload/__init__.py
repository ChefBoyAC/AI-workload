"""Workload generation for Astra-SIM (MoE and dense LLM-v1)."""

from .llm_params import LlmParams, llm_params_from_config
from .llm_workload import generate_llm_workload
from .moe_params import MoEParams, moe_params_from_config
from .moe_workload import generate_moe_workload
from .text_trace import write_llm_text_workload, write_moe_text_workload, write_text_workload

__all__ = [
    "MoEParams",
    "moe_params_from_config",
    "generate_moe_workload",
    "write_moe_text_workload",
    "write_text_workload",
    "LlmParams",
    "llm_params_from_config",
    "generate_llm_workload",
    "write_llm_text_workload",
]
