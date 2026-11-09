# Workload: LLM-v1

Dense LLM training-step skeleton: TP AllReduce (activation) + DP AllReduce (grad bucket).

Shared across topologies **Ring**, **Rail**, and **Torus**. Topology-specific NPU count / dims live under `topologies/`, not here.
