# Scheduler: NCCL (Astra-SIM approximation)

This folder models an **NCCL-like** collective + scheduling policy for the Progress Report 2 matrix (Scheduler 4).

Astra-SIM does not run the real NCCL library. These `system.json` files approximate NCCL behavior with:

| Knob | Choice | Why |
|---|---|---|
| AllReduce / AllGather / ReduceScatter | `ring` per dimension | Classic NCCL large-message AllReduce |
| AllToAll | `direct` | Pairwise-style exchange (MoE EP) |
| `scheduling-policy` | `FIFO` | In-order launch vs baseline `LIFO` |
| `active-chunks-per-dimension` / `preferred-dataset-splits` | `8` | Mimics NCCL multi-channel chunking |

Use with a frozen workload + topology experiment under `experiments/*` by pointing `--system-configuration` at the matching topology file here (`ring.json`, `rail.json`, or `torus.json`).
