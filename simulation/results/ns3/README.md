# NCCL column, ns-3 backend

Six NCCL cells run with `SCHEDULER=nccl BACKEND=ns3` on main at d863727 (fabrics aligned to the Progress Report 2 doc, HYBRID_DATA_MODEL trace, ns-3 boot stubs, ring AllToAll on Ring/Torus).

| Config | JCT (us) | Exposed comm (us) | Compute (us) |
|---|---:|---:|---:|
| LLM-v1 Torus | 16452.132 | 13449.132 | 3003.0 |
| LLM-v1 Ring | 16406.309 | 13403.309 | 3003.0 |
| LLM-v1 Rail | 5786.011 | 2783.011 | 3003.0 |
| MoE-v1 Torus | 33869.566 | 27862.566 | 6007.0 |
| MoE-v1 Ring | 62028.885 | 56021.885 | 6007.0 |
| MoE-v1 Rail | 12477.455 | 6470.455 | 6007.0 |

Files: `nccl-<workload>-<topology>.json` is `generated/results.json` from that run; `.astra_sim.log` is the ASTRA-sim log.

Note: Rail still uses `direct` AllToAll; Ring and Torus use `ring` (direct aborts ns-3 there). The MoE cells are therefore not on identical AllToAll settings.
