# Network backend: ns-3 (matrix standard)

Source of truth: Progress Report 2 Google Doc (Group 31).

**All matrix cells — including baseline — use ns-3.** Analytical is debug-only (`BACKEND=analytical`).

## Topology + link parameters (from the doc)

| Topology | Shape | Bandwidth | Latency |
|---|---|---|---|
| Ring | one 16-GPU ring | 50 GB/s (= **400 Gbps**) | 500 ns/hop (= **0.0005 ms**) |
| Torus | 3D **4×2×2** | 50 GB/s (= **400 Gbps**) | 500 ns/hop (= **0.0005 ms**) |
| Rail | **2 servers × 8 GPUs** | **300 GB/s** intra (= **2400 Gbps**), **50 GB/s** inter (= **400 Gbps**) | **100 ns** intra (= **0.0001 ms**), **500 ns** inter (= **0.0005 ms**) |

## Workload baseline (from the doc)

| Parameter | Dense LLM (LLM-v1) | MoE (MoE-v1) |
|---|---|---|
| GPUs | 16 | 16 |
| Precision | BF16 | BF16 |
| Sequence length | 2048 | 2048 |
| Hidden size | 4096 | 4096 |
| Global / microbatch | 64 / 4 | 64 / 4 |
| Parallelism | TP=4, DP=4, PP=1 | TP=2, EP=4, DP=2, PP=1 |
| Experts | N/A | 8, Top-2 |
| Main communication | AllReduce | AllToAll |
| Activation / MoE routed / DP grad | 64 MiB / N/A / 256 MiB | 64 / 128 / 256 MiB |

Encoded in `config/*_rail.yaml` + `workloads/*/params.yaml`. Chakra text mode: `HYBRID_DATA_MODEL`.

Schedulers only swap `system.json`; fabric stays fixed under this folder.
