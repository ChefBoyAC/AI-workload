# AI-workload

CS2200 / ASTRA-sim intra-DLT scheduler comparison.

## Simulation kit (matrix baselines + NCCL)

The unified Progress Report 2 layout lives under **[`simulation/`](simulation/)**:

- Workloads: **LLM-v1**, **MoE-v1**
- Topologies: **Ring**, **Rail**, **Torus**
- Schedulers (so far): **baseline** + **NCCL**

Start here: [`simulation/README.md`](simulation/README.md)

## Legacy / source kits (still in repo)

| Path | Origin branch | Role |
|---|---|---|
| `configs/*-ring/`, `scripts/run_*_ring.py` | `arianne-ring` | Ring scenarios |
| `simulation/config`, `simulation/src`, Docker | `simulation` | Rail generators |
| `torus_kit/` | `torus` | Torus scenarios |

