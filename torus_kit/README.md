# LLM-Torus and MoE-Torus scenarios for ASTRA-sim

This kit has the two workload/topology scenarios (dense LLM and Mixture-of-Experts) on a 64-NPU 3D torus. Both have been tested end to end on ASTRA-sim's analytical backend.

## 1. One-time setup (Ubuntu, WSL2 or a Linux VM)

```bash
sudo apt-get install -y git cmake g++ python3-pip protobuf-compiler libprotobuf-dev
git clone --recurse-submodules https://github.com/astra-sim/astra-sim.git ~/astra-sim
cd ~/astra-sim && bash build/astra_analytical/build.sh     # ~5 min
pip install "protobuf>=4,<5"        # add --break-system-packages if pip complains
```

On a Mac or Windows machine without WSL, use the repo's `Dockerfile` instead.

## 2. Run both scenarios

```bash
cd torus_kit
ASTRA_SIM_DIR=~/astra-sim ./run_all.sh
```

Full logs go to `results/`. The number to compare is **Wall time** (cycles, where 1 cycle = 1 ns). That's your job completion time (JCT) for one training iteration.

| Scenario | Wall time (JCT) | Comm time | Compute | Overlap |
|---|---|---|---|---|
| LLM-Torus | 46.4 ms | 39.9 ms | 13.4 ms | 6.9 ms |
| MoE-Torus | 82.0 ms | 58.9 ms | 26.9 ms | 3.8 ms |

These come from the baseline run with ring collectives and LIFO scheduling, so they are the "no smart scheduler" reference point.

## 3. What each file is

ASTRA-sim always takes four inputs:

| Input | File | What it describes |
|---|---|---|
| Network (topology) | `network/torus3d_4x4x4.yml` | 3D torus, 4×4×4 = 64 NPUs, 50 GB/s per link, 500 ns per hop |
| System (collective algorithms + scheduling) | `system/torus3d_ring.json` | Ring all-reduce on each dimension, direct all-to-all, LIFO scheduling |
| Memory | `remote_memory/no_memory_expansion.json` | Default, ignore it |
| Workload | `workload/*/*.et` (made by `scripts/gen_workload.py`) | One Chakra trace per NPU: compute nodes, collective nodes and their dependencies (a DAG) |

**Torus note:** ASTRA-sim has no `Torus` keyword. A torus is written as `[Ring, Ring, Ring]` because each ring wraps around. NPU `id` sits at `x = id%4`, `y = (id/4)%4`, `z = id/16`.

## 4. The two workloads

Both model a transformer with hidden size 4096, sequence length 2048, 8 layers and fp16. Each collective is pinned to specific torus dimensions with Chakra's `involved_dim` attribute:

| | Dim 0 (x) | Dim 1 (y) | Dim 2 (z) |
|---|---|---|---|
| **LLM** | Tensor-parallel AllReduce (activations, 16 MB, 4× per layer) | Data-parallel AllReduce of gradients (~100 MB per layer) | Data-parallel AllReduce of gradients (~100 MB per layer) |
| **MoE** | Tensor-parallel AllReduce (attention) | **Expert-parallel All-to-All** (token dispatch + combine, 4× per layer) plus dense data-parallel AllReduce | Dense data-parallel AllReduce plus expert-gradient AllReduce |

The MoE layer is Mixtral-style: 8 experts, top-2 routing, and EP = 4, so each EP rank holds 2 experts. Its signature traffic is All-to-All, which a torus handles badly because of multi-hop paths and bisection limits. That's why MoE-Torus is a good stress scenario.

## 5. Changing it to match the professor's spreadsheet

Everything is a command-line flag (`python3 scripts/gen_workload.py -h`):

```bash
python3 scripts/gen_workload.py --model moe --layers 32 --hidden 4096 --seq 4096 \
    --experts 8 --topk 2 --ep 4 --tp 4 --tflops 400 --out workload/moe_torus/moe
```

- **Bigger or smaller torus:** edit `npus_count` in the `.yml`, and pass a matching `--npus`, `--tp` (= dim 0 size) and `--ep` (= dim 1 size).
- **Link speed or latency:** edit `bandwidth` and `latency` in the `.yml`.
- **Collective algorithm per dimension:** edit the `*-implementation` lists in the system JSON. Each list needs one entry per dimension.

## 6. Caveats for the scheduler comparison

1. **No congestion modelling.** ASTRA-sim's *congestion-aware* analytical backend only supports 1-D topologies (we hit `only support 1-dim topology`), so these runs use the *congestion-unaware* backend. Concurrent flows on the same link are not slowed down by each other. For schedulers whose whole point is avoiding contention (MXDAG, TACCL, Syndicate), this hides part of their benefit. If the group needs that, the ns-3 backend (`build/astra_ns3`) models real congestion on arbitrary topologies, but it is slower and harder to set up.
2. **Where MXDAG plugs in: transfer order in the Chakra graph, not `comm_priority`.** ASTRA-sim lets each NPU run only **one collective at a time**, like a single communication stream (see `astra-sim/workload/HardwareResource.cc`). When several collectives are ready, they start in **node-ID order**. As a result, the `"EXPLICIT"` scheduling policy plus `comm_priority` has no effect, because two collectives never wait in the system queue together. Tested: priorities changed nothing, but swapping the IDs of a big and a small transfer cut the job from 13.0 ms to 8.0 ms. An MXDAG implementation therefore rewrites the Chakra graph:
   1. Build the DAG. The generator already has it in memory.
   2. Compute the critical path using the MXDAG paper's path-length equations.
   3. Give critical-path collectives **lower node IDs** than off-path ones, so they start first.
   4. Optionally, add **extra dependency edges** that hold an off-path collective back until a critical one has finished. This is Principle 1's "delay non-critical paths".
