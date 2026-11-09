"""
2×8 Rail cluster model.

Physical intent:
  Node 0: GPUs 0–7
  Node 1: GPUs 8–15
  Rail i connects GPU i on node 0 with GPU i+8 on node 1.

Analytical Phase 1 approximates this as hierarchical dims [8, 2]:
  dim0 = intra-node (FullyConnected / Switch)
  dim1 = inter-node rail (Switch)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RailTopology:
    nodes: int
    gpus_per_node: int
    logical_dims: tuple[int, ...]
    intra_node_topology: str
    inter_node_topology: str
    bandwidth_gbs: tuple[float, ...]
    latency_ns: tuple[float, ...]

    @property
    def total_gpus(self) -> int:
        return self.nodes * self.gpus_per_node

    def validate(self) -> None:
        product = 1
        for dim in self.logical_dims:
            product *= dim
        if product != self.total_gpus:
            raise ValueError(
                f"logical_dims product {product} must equal total_gpus {self.total_gpus}"
            )
        if self.nodes != 2 or self.gpus_per_node != 8:
            raise ValueError("Phase B locks rail to 2 nodes × 8 GPUs")
        if len(self.logical_dims) != 2:
            raise ValueError("Phase B analytical rail expects exactly 2 logical dims")
        if len(self.bandwidth_gbs) != len(self.logical_dims):
            raise ValueError("bandwidth_gbs length must match logical_dims")
        if len(self.latency_ns) != len(self.logical_dims):
            raise ValueError("latency_ns length must match logical_dims")

    def rank_layout(self) -> dict[str, Any]:
        """Explicit GPU placement for scheduler / topology developers."""
        node0 = list(range(0, self.gpus_per_node))
        node1 = list(range(self.gpus_per_node, self.total_gpus))
        rails = [
            {"rail": i, "endpoints": [i, i + self.gpus_per_node]}
            for i in range(self.gpus_per_node)
        ]
        return {
            "nodes": {
                "node0_gpus": node0,
                "node1_gpus": node1,
            },
            "rails": rails,
            "logical_dims": list(self.logical_dims),
            "dim_meaning": {
                "0": "intra-node (package)",
                "1": "inter-node rail",
            },
        }


def rail_from_config(config: dict[str, Any]) -> RailTopology:
    rail = config["rail"]
    topo = RailTopology(
        nodes=int(rail["nodes"]),
        gpus_per_node=int(rail["gpus_per_node"]),
        logical_dims=tuple(int(x) for x in rail["logical_dims"]),
        intra_node_topology=str(rail["intra_node_topology"]),
        inter_node_topology=str(rail["inter_node_topology"]),
        bandwidth_gbs=tuple(float(x) for x in rail["bandwidth_gbs"]),
        latency_ns=tuple(float(x) for x in rail["latency_ns"]),
    )
    topo.validate()

    # Cross-check against MoE world size when present.
    moe_gpus = int(config.get("moe", {}).get("gpus", topo.total_gpus))
    if moe_gpus != topo.total_gpus:
        raise ValueError(f"moe.gpus={moe_gpus} must equal rail total_gpus={topo.total_gpus}")
    return topo
