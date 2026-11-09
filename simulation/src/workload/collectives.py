"""
Small builders for compute / collective nodes in the MoE step graph.

Phase B emits a developer-readable JSON graph. Phase C (Docker) will map these
node types onto Chakra ET nodes that Astra-SIM 2.x consumes.
"""

from __future__ import annotations

from typing import Any


def compute_node(node_id: str, name: str, duration_ns: int, deps: list[str]) -> dict[str, Any]:
    """A timed compute stub (dense FFN, expert MLP, etc.)."""
    return {
        "id": node_id,
        "kind": "COMPUTE",
        "name": name,
        "duration_ns": duration_ns,
        "depends_on": list(deps),
    }


def collective_node(
    node_id: str,
    name: str,
    collective: str,
    size_bytes: int,
    parallelism: str,
    deps: list[str],
) -> dict[str, Any]:
    """
    A communication collective the system layer will expand.

    collective: ALLTOALL | ALLREDUCE | ALLGATHER | REDUCESCATTER
    parallelism: TP | EP | DP  (documentation for scheduler authors)
    """
    return {
        "id": node_id,
        "kind": "COMM_COLL",
        "name": name,
        "collective": collective,
        "parallelism": parallelism,
        "size_bytes": size_bytes,
        "size_mib": round(size_bytes / (1024 * 1024), 3),
        "depends_on": list(deps),
    }


def tp_activation_allreduce(node_id: str, size_bytes: int, deps: list[str]) -> dict[str, Any]:
    return collective_node(
        node_id=node_id,
        name="tp_activation_allreduce",
        collective="ALLREDUCE",
        size_bytes=size_bytes,
        parallelism="TP",
        deps=deps,
    )


def ep_moe_alltoall(node_id: str, size_bytes: int, deps: list[str]) -> dict[str, Any]:
    return collective_node(
        node_id=node_id,
        name="ep_moe_alltoall",
        collective="ALLTOALL",
        size_bytes=size_bytes,
        parallelism="EP",
        deps=deps,
    )


def dp_grad_allreduce(node_id: str, size_bytes: int, deps: list[str]) -> dict[str, Any]:
    return collective_node(
        node_id=node_id,
        name="dp_gradient_bucket_allreduce",
        collective="ALLREDUCE",
        size_bytes=size_bytes,
        parallelism="DP",
        deps=deps,
    )
