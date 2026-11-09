# AI-workload

CS2200 / ASTRA-sim intra-DLT scheduler comparison.

## Simulation kit (matrix + Docker)

See **[`simulation/README.md`](simulation/README.md)**.

Docker can run **LLM-v1 / MoE-v1 × Ring / Rail / Torus**. The default communication scheduler is **NCCL** (`schedulers/nccl/`).

```bash
cd simulation
docker compose build
docker compose run --rm llm-rail    # NCCL default
docker compose run --rm moe-torus
SCHEDULER=baseline docker compose run --rm moe-ring
```

## Legacy kits still in repo

| Path | Role |
|---|---|
| `configs/*-ring/`, `scripts/run_*_ring.py` | Original Ring host scripts |
| `torus_kit/` | Original Torus host kit |
| `simulation/` | Unified matrix + Docker runner |

