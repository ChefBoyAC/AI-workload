# Workload: MoE-v1

MoE training-step skeleton: TP AllReduce + EP AllToAll (dispatch/combine) + DP AllReduce.

Shared across topologies **Ring**, **Rail**, and **Torus**.
