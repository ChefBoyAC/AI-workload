# Requirements Checklist — MoE Workload + Rail Topology on Astra-SIM

## 1. Goal

Build a **reproducible Astra-SIM 2.x** setup that:

1. Generates a **MoE AI workload** from fixed training parameters.
2. Generates a **Rail physical + logical topology** for **2 nodes × 8 GPUs = 16 GPUs**.
3. Runs end-to-end in **Docker** so host RAM use stays bounded.
4. Reports **job completion time** for later comparison of custom **intra-DLT schedulers**.

This repo starts from workload + topology. Scheduler plug-ins (TACCL, NCCL, ByteScheduler, SYNDICATE, CadentFlow equivalents) come after the baseline sim runs.

---

## 2. Decisions (locked)

| Decision | Choice | Notes |
|---|---|---|
| Astra-SIM version | **Current 2.x** (Chakra ET workload path) | Not Astra-SIM 1.0 text workloads |
| Cluster shape | **2 nodes × 8 GPUs = 16 GPUs** | Rail-optimized fabric |
| Deliverable | **Workload file(s) + topology file(s) + Docker one-shot sim** | `docker compose up` / `docker run` should simulate |
| Language | **Python** generators + thin shell/Docker entrypoint | Readable, refactored modules |
| Parallelism | TP=2, EP=4, DP=2, PP=1 | Product = 16 GPUs |
| Primary MoE collective | **AllToAll** | Expert dispatch / combine |

### Does the network backend matter?

**Yes — especially for your scheduler comparison.**

| Backend | What it models well | RAM / speed | Fit for your goal |
|---|---|---|---|
| **Analytical** | Hierarchical dims, link BW/latency, collective phasing | Low RAM, fast | Best **Phase 1** baseline |
| **ns-3** | Packet-level contention on a real physical rail graph | Higher cost | Best **Phase 2** when comparing schedulers |

How this maps to the systems you named:

- **NCCL / TACCL** → mostly Astra-SIM **system layer** (native collectives vs custom Chakra collective plans). Backend still matters because send/recv timing depends on the network model.
- **ByteScheduler / SYNDICATE / CadentFlow** → scheduling under **contention / chunking / overlap**. Differences show up more clearly on **ns-3** (or another contention-aware backend) than on pure analytical.

**Requirement for this project:**

- **Phase 1 (now):** Analytical backend + Rail logical dims, Dockerized, low RAM.
- **Phase 2 (next):** Same workload + explicit Rail **physical** topology on ns-3 for scheduler bake-offs.

Do not skip Phase 1; it validates MoE + Rail wiring before paying ns-3 cost.

---

## 3. MoE workload parameters (source of truth)

| Field | Value |
|---|---|
| GPUs | 16 |
| Precision | BF16 (2 bytes/element) |
| Sequence length | 2048 tokens |
| Hidden size | 4096 |
| Global batch size | 64 |
| Microbatch size | 4 |
| Parallelism | TP=2, EP=4, DP=2, PP=1 |
| Experts | 8 experts, Top-2 routing |
| Main communication | AllToAll |
| Representative activation tensor | 64 MiB |
| MoE routed payload | 128 MiB aggregate |
| DP gradient bucket | 256 MiB (values) / parameter |

### Derived checks (must pass in generator)

- `TP * EP * DP * PP == 16`
- Microbatches per global step: `global_batch / microbatch == 16`
- Expert placement: `num_experts / EP == 2` experts per EP rank (8/4)
- Top-2 routing → each token contacts 2 experts (used when sizing routed payload commentary; payload sizes above are authoritative for the sim)

### Workload contents (minimum)

Generate Chakra-compatible / Astra-SIM 2.x workload inputs that include, per training step (forward + backward as needed for completion time):

1. Compute nodes for dense + MoE expert compute (can be simplified timing stubs initially).
2. **TP** collectives on activation-sized traffic (~64 MiB representative).
3. **EP AllToAll** on MoE routed payload (~128 MiB aggregate).
4. **DP** gradient AllReduce / bucket (~256 MiB).
5. **PP=1** → no pipeline send/recv stage edges.

Exact ET schema must match current Astra-SIM / Chakra submodule used in Docker.

---

## 3b. LLM-v1 workload parameters (sibling of MoE)

Config file: `config/llm_v1_rail.yaml`. Same Rail topology; `generated/` is **overwritten** per run to save storage.

| Field | Value |
|---|---|
| GPUs | 16 |
| Precision | BF16 (2 bytes/element) |
| Sequence length | 2048 tokens |
| Hidden size | 4096 |
| Global batch size | 64 |
| Microbatch size | 4 |
| Parallelism | TP=4, DP=4, PP=1 |
| Experts | N/A |
| Main communication | AllReduce |
| Representative activation tensor | 64 MiB |
| MoE routed payload | N/A |
| DP gradient bucket | 256 MiB |

### Derived checks

- `TP * DP * PP == 16`
- Microbatches per global step: `global_batch / microbatch == 16`
- No EP / AllToAll nodes in the graph

### Workload contents (minimum)

1. Dense forward compute stub
2. **TP AllReduce** (~64 MiB activation)
3. Dense backward compute stub
4. **DP AllReduce** (~256 MiB grad bucket)
5. **PP=1** → no pipeline edges

`results.json` uses the same §7 metric schema as MoE.

### Intended future repo layout (scheduler bake-off table)

Not required for LLM-v1 delivery; target shape for readability later:

```text
workload/          # moe/, llm/
topology/          # moe-rail/, llm-rail/, moe-torus/, llm-torus/
schedulers/        # NCCL/, TACCL/, ByteScheduler/, SYNDICATE/, CadentFlow/
```

Today: sibling YAML configs + shared Rail generators + overwrite `generated/`.

---

## 4. Rail topology (2×8)

### Physical intent

- **Node 0:** GPUs `0–7`
- **Node 1:** GPUs `8–15`
- **Intra-node:** high-BW fully-connected or switch (NVLink/NVSwitch-like)
- **Inter-node rail:** GPU `i` on node 0 pairs with GPU `i+8` on node 1 on the same rail (rail-optimized), for `i in 0..7`

### Logical dims (Phase 1 analytical)

Recommended starting logical shape aligned with rail:

- Dim 0 (intra-node / package): size **8**
- Dim 1 (inter-node / rail): size **2**

Total NPUs: `8 * 2 = 16`.

System-layer collective implementations must use **the same number of dimensions** (2 entries per collective list).

### Collective → dimension mapping (baseline)

| Parallelism | Collective | Preferred dim | Notes |
|---|---|---|---|
| TP=2 | AllReduce / AllGather | mostly intra-node | Place TP pairs within a node when possible |
| EP=4 | AllToAll | mix / dim spanning EP group | Primary MoE cost |
| DP=2 | AllReduce (grad bucket) | often rail / inter-node | 256 MiB |
| PP=1 | — | n/a | disabled |

Exact rank layout must be documented in code comments and in generated `topology` metadata JSON.

---

## 5. Repository layout (target)

```text
.
├── REQUIREMENTS.md                 # this file
├── README.md                       # how to build & run (later)
├── docker/
│   ├── Dockerfile                  # Astra-SIM 2.x + Python deps
│   └── entrypoint.sh               # generate → simulate → print JCT
├── docker-compose.yml              # memory limits, one-shot run
├── config/
│   └── moe_rail.yaml               # single source of truth for params
├── src/
│   ├── __init__.py
│   ├── config_loader.py            # load/validate YAML
│   ├── workload/
│   │   ├── __init__.py
│   │   ├── moe_params.py           # dataclasses / validation
│   │   ├── moe_workload.py         # orchestrates ET generation
│   │   └── collectives.py          # TP / EP / DP node helpers
│   ├── topology/
│   │   ├── __init__.py
│   │   ├── rail.py                 # 2×8 rail graph + logical dims
│   │   └── analytical_network.py   # network.yml emitter (Phase 1)
│   └── run_sim.py                  # CLI: generate + invoke Astra-SIM
├── generated/                      # gitignored outputs
│   ├── workload/
│   └── topology/
└── scripts/
    └── smoke_test.sh               # local/docker smoke check
```

### Refactoring rules

- No function longer than ~40 lines without extracting helpers.
- Generators are pure where possible: `params → files`.
- Docker entrypoint only orchestrates; it does not embed MoE math.
- Do not put secrets in `.env` without explicit permission.

---

## 6. Docker / RAM constraints

- Use official/current **astra-sim** clone inside image (pinned git SHA in Dockerfile).
- Prefer **analytical** binary for default `docker compose up`.
- Set container memory limit (suggested starting point: **4g**, adjust after smoke test).
- Mount `./generated` and `./config` as volumes.

### Entrypoint regenerate behavior

Docker does **not** magically regenerate files. The entrypoint controls it:

| Mode | Env | Behavior |
|---|---|---|
| **Default** | `GENERATE=1` | Regenerate workload + topology (+ baseline system.json) from `config/moe_rail.yaml`, then run sim |
| **Reuse** | `GENERATE=0` | Use existing `generated/` artifacts; only run sim (scheduler bake-offs on frozen inputs) |
| **Optional** | config hash stamp | Skip regen when YAML hash unchanged (nice-to-have) |

Default command when `GENERATE=1`: validate config → generate → run sim → print metrics → write `generated/results.json`.

### Built-in (default) communication scheduler

Phase 1 does **not** ship TACCL / NCCL / ByteScheduler / SYNDICATE / CadentFlow. Astra-SIM still produces a completion time because its **system layer always instantiates a collective scheduler** driven by `system.json` (e.g. FIFO/LIFO, baseline/themis, native ring/direct collectives). That baseline JCT is what custom schedulers will later beat or match.

---

## 7. Simulation output metrics (developer contract)

Astra-SIM analytical tutorials typically emit a CSV row (e.g. `tutorial_result.csv`). This project normalizes that into `generated/results.json` and stdout.

| Field (Astra-SIM) | Meaning | Overall metric role |
|---|---|---|
| **Name** | Run / experiment label (e.g. `moe_rail_baseline`) | Identifies which workload+topology+system config produced the row; required when comparing schedulers side-by-side |
| **Total Time (us)** | End-to-end simulated time for the issued workload (one training step by default) | **Primary JCT proxy.** This is the number scheduler authors optimize. Lower is better |
| **Compute Time (us)** | Time attributed to compute nodes in the ET (Phase C approximates as `total_cycles - exposed_comm_cycles` when Astra-SIM only prints finish lines) | Shows how much of JCT is “real work.” If this dominates, network/scheduler changes will move JCT little |
| **Exposed Communication Time (us)** | Communication that did **not** fully overlap with compute (comm on the critical path) | **Primary scheduler-sensitivity metric.** Custom intra-DLT schedulers should mainly reduce this (and thus Total Time) under the same workload+topology |
| **Total Message Size (MB / MiB)** | Aggregate communication volume. Tutorials label this MB; Phase C fills it from the **workload contract** (`64 + 128 + 128 + 256` MiB per step) when the analytical log does not print a size column | Sanity check that the generated workload actually issued the intended MoE/TP/DP traffic |

### Derived metrics (compute in our post-process, not always printed by Astra-SIM)

| Derived field | How to read it | Overall metric role |
|---|---|---|
| **Overlapped compute+comm (approx)** | `Total Time - Exposed Communication` vs `Compute Time` | Rough overlap quality; helpful when adding compute stubs so Phase D schedulers can show overlap gains |
| **Comm fraction of JCT** | `Exposed Communication / Total Time` | Quick signal: high fraction → fertile ground for scheduler work |
| **BW utilization (when ideal time known)** | `ideal_collective_time / Total Time` | Tutorial-style efficiency; only valid for comm-only microbenchmarks with a known ideal |

If a live Docker run surfaces additional stable fields (per-collective breakdowns, per-dimension busy time, etc.), **append them as new rows in this table** with Meaning + Overall metric role filled in.

### What the generators must put in (so those metrics are meaningful)

**Workload (`generated/workload/`)** must encode a MoE step whose collectives match §3 payload sizes, on **16 ranks**, with PP=1 (no pipeline edges). Wrong sizes → misleading `Total Message Size` and useless JCT comparisons.

**Topology (`generated/topology/`)** must encode Rail **2×8** analytical dims (`npus_count: [8, 2]`), plus metadata describing rank layout (`0–7` / `8–15`, rails `i ↔ i+8`). Dim count must match `system.json` collective implementation list lengths.

**System (`generated/system/system.json`)** must explicitly declare the **default** scheduler + native collective algorithms (baseline). Scheduler developers replace/extend this file; they should not silently change workload/topology when `GENERATE=0`.

---

## 8. Acceptance criteria

- [x] `REQUIREMENTS.md` exists and matches locked decisions above
- [x] `config/moe_rail.yaml` encodes all MoE + Rail parameters
- [x] Python generator emits workload artifacts under `generated/workload/`
- [x] Python generator emits topology / network artifacts under `generated/topology/`
- [x] Baseline `system.json` emitted with default scheduler knobs documented
- [x] `TP*EP*DP*PP == 16` validated at runtime
- [x] Docker image builds on current Astra-SIM 2.x
- [x] One Docker command runs the simulation without editing host Python env (`docker compose run --rm moe-rail-sim`)
- [x] Entrypoint supports `GENERATE=1` (default) and `GENERATE=0` (reuse)
- [x] Simulation prints / writes metrics from §7 into `generated/results.json` (at least Total Time as JCT)
- [x] Code modules are split (workload vs topology vs config); no “god” functions
- [x] README explains how this baseline will later host scheduler comparisons
- [x] §7 metrics table includes Field / Meaning / Overall metric role (+ derived metrics)
- [x] Text workload bridge (`moe_workload.txt` / `llm_v1_workload.txt`) generated for Chakra `Text` → `.et` conversion
- [x] `config/llm_v1_rail.yaml` + LLM generators + Docker run produce §7 `results.json`

---

## 9. Phased plan

### Phase A — Spec (this document)
- Lock params, backend strategy, layout, acceptance checks, metrics contract.

### Phase B — Generators (host-readable Python)  ← current
- Implement config + MoE workload emitter + Rail analytical topology emitter + baseline system.json.
- Unit-validate derived checks without needing full Astra-SIM binary.
- Emit developer-facing workload graph JSON (Chakra ET binary conversion can complete in Phase C inside Docker).

### Phase C — Docker one-shot
- Dockerfile + entrypoint (`GENERATE=0|1`) + compose memory limits.
- Wire generators → Astra-SIM analytical run → parse §7 metrics → `generated/results.json`.
- If new useful result fields appear, append them to the §7 table.

### Phase D — Scheduler comparison prep
- System-layer configs approximating NCCL-like native rings/direct.
- Hooks for custom collective ETs (TACCL-like).
- Scheduling policy knobs for ByteScheduler / SYNDICATE / CadentFlow-style experiments.
- Optional switch to ns-3 Rail physical topology for contention-accurate bake-offs.

---

## 10. Out of scope (for now)

- Full real model FLOPs calibration from a production MoE checkpoint
- Implementing TACCL/NCCL/ByteScheduler/SYNDICATE/CadentFlow inside Astra-SIM
- Modifying host `.env` without permission
- Committing generated binaries / large ET dumps

---

## 11. Open items to resolve during Phase B/C

1. Pin exact Astra-SIM git commit SHA once Docker build is proven.
2. Confirm Chakra ET node types required by that commit for AllToAll + AllReduce.
3. Choose concrete intra-node BW/latency and rail BW/latency numbers for analytical `network.yml` (placeholders allowed initially, must be YAML knobs).
4. Decide whether one global training step or N steps are simulated for JCT (default: **1 step**, configurable).
