# Results

JCT (Total Time, µs) for one training step, 16 GPUs in every topology, main at commit 4938604.
Schedulers: `nccl` = NCCL-like config in `simulation/schedulers/nccl/`, `baseline` = ASTRA-sim default config in `simulation/schedulers/baseline/`.

| Workload | Topology | Scheduler | Analytical unaware | Analytical aware | ns-3 | ns-3 exposed comm |
|---|---|---|---|---|---|---|
| LLM-v1 | Torus | nccl | 4241.1 | n/a | 13748.9 | 10746.9 |
| LLM-v1 | Torus | baseline | 10902.1 | n/a | 11881.9 | 8879.9 |
| LLM-v1 | Ring | nccl | 4261.4 | 12390.7 | 13688.7 | 10686.7 |
| LLM-v1 | Ring | baseline | 5492.3 | 12390.8 | 13747.8 | 10745.8 |
| LLM-v1 | Rail | nccl | 3326.6 | n/a | 5238.4 | 2236.4 |
| LLM-v1 | Rail | baseline | 20190.0 | n/a | 20438.7 | 17436.7 |
| MoE-v1 | Torus | nccl | 7251.4 | n/a | 16769.3 | 10763.3 |
| MoE-v1 | Torus | baseline | 13926.7 | n/a | 14930.6 | 8924.6 |
| MoE-v1 | Ring | nccl | 7292.1 | 15421.3 | 16749.4 | 10743.4 |
| MoE-v1 | Ring | baseline | 8522.9 | 15421.5 | 16808.5 | 10802.5 |
| MoE-v1 | Rail | nccl | 6333.7 | n/a | 8274.8 | 2268.8 |
| MoE-v1 | Rail | baseline | 23197.2 | n/a | 23475.1 | 17469.1 |

## Folders

- `analytical-unaware/`: `generated/results.json` from the Docker runner (`docker compose run --rm <cell>`), which uses `AstraSim_Analytical_Congestion_Unaware`.
- `analytical-aware/`: same runner pointed at `AstraSim_Analytical_Congestion_Aware`. Ring only; that binary rejects Torus and Rail with `only support 1-dim topology`.
- `ns3/logs/`: full ASTRA-sim output of the ns-3 backend for all 12 runs. `summary.json` has the max per-NPU Wall time and exposed communication.
- `ns3/topologies/`: ns-3 inputs. `<topo>_topology.txt` (physical links), `<topo>_logical.json` (logical dims), `config_<topo>.txt` (ns-3 config). Ring: 16 nodes, 400 Gbps, 500 ns. Torus: 4×2×2, 32 links, 400 Gbps. Rail: 2 servers × 8 GPUs, 2400 Gbps full mesh inside a server, 400 Gbps to 8 rail switches.

## Reproducing the ns-3 runs

In the simulation image: `apt-get update && apt-get install -y libopenmpi-dev openmpi-bin`, then `cd /opt/astra-sim/build/astra_ns3 && ./build.sh -c`. Run from `extern/network_backend/ns-3/build/scratch`:

    ./ns3.42-AstraSimNetwork-default --workload-configuration=<generated>/workload/workload_et --system-configuration=<generated>/system/system.json --network-configuration=config_<topo>.txt --remote-memory-configuration=<generated>/system/remote_memory.json --logical-topology-configuration=<topo>_logical.json --comm-group-configuration=empty

The flow file named in the config must exist before the run (`echo 0 > flow.txt`). Each run takes about 2.5 minutes.

## Known issue

The generated workload text starts with `DATA`, and in that mode the Chakra text converter emits only the weight-gradient communication. The 64 MiB TP AllReduce and the two 128 MiB MoE AllToAlls are present in `moe_workload.txt` but appear as size 0 in the ET, so every backend simulates only the 256 MiB gradient AllReduce. MoE therefore differs from LLM by compute only until this is fixed.
