# ASTRA-sim Intra-DLT Matrix (Simulation Kit)

Unified **baseline workloads + topologies** for Progress Report 2, plus an **NCCL** scheduler approximation.

Merged from branches: `arianne-ring` (Ring), `simulation` (Rail), `torus` (Torus).

## Progress Report 2 coverage

| AI Workload | Workload Config ID | Topology | Baseline | NCCL |
|---|---|---|---|---|
| LLM | LLM-v1 | Torus | `experiments/llm-v1-torus` | `schedulers/nccl/torus.json` |
| LLM | LLM-v1 | Ring | `experiments/llm-v1-ring` | `schedulers/nccl/ring.json` |
| LLM | LLM-v1 | Rail | `experiments/llm-v1-rail` | `schedulers/nccl/rail.json` |
| MoE | MoE-v1 | Torus | `experiments/moe-v1-torus` | `schedulers/nccl/torus.json` |
| MoE | MoE-v1 | Ring | `experiments/moe-v1-ring` | `schedulers/nccl/ring.json` |
| MoE | MoE-v1 | Rail | `experiments/moe-v1-rail` | `schedulers/nccl/rail.json` |

Cadent Flow / SYNDICATE / TACCL / ByteScheduler are **out of scope** here (placeholders in the spreadsheet only).

## Layout

```text
simulation/
  workloads/          # llm-v1/, moe-v1/  (shared params)
  topologies/         # ring/, rail/, torus/
  schedulers/
    baseline/         # default Astra-SIM system policies
    nccl/             # NCCL-like system policies
  experiments/        # one folder per matrix row (workload × topology)
  config/ + src/ + docker/   # Rail generators / Docker one-shot
```

Each `experiments/<row>/experiment.json` points at the workload params, topology network, baseline scheduler, and NCCL scheduler.

### NPU counts

| Topology | NPUs |
|---|---|
| Ring | 16 |
| Rail | 16 (2×8) |
| Torus | 64 (4×4×4) |

Do not compare JCTs across different NPU counts as if they were identical conditions.

## Quick pointers

- **Ring runs:** repo-root `scripts/run_llm_ring.py`, `scripts/run_moe_ring.py` + `configs/*-ring/`
- **Rail runs:** see Rail Docker section below (`docker compose` from this `simulation/` directory)
- **Torus runs:** repo-root `torus_kit/run_all.sh`
- **NCCL:** swap `--system-configuration` to the matching file under `schedulers/nccl/` (see `schedulers/nccl/README.md`)

---

# Rail Docker guide (LLM-v1 / MoE-v1 on Rail)

## Prerequisites

1. **Docker Desktop** running (enough free disk — the image is ~3 GB; builds need more headroom).
2. This repo checked out locally.
3. Terminal open at the repo root:

```bash
cd "/Users/aaronchanner/Desktop/Workload + Topology ( Rail )"
```

---

## Step 0 — One-time image build

Only needed once (or after Dockerfile changes). Can take 15–30+ minutes.

```bash
docker compose build moe-rail-sim
```

You should see image `moe-rail-astra-sim:phase-c`.

---

## Step 1 — Run the MoE + Rail simulation

```bash
docker compose run --rm --entrypoint bash moe-rail-sim -c \
  'pip install -q "protobuf==7.35.1" && python -m src.run_sim'
```

### What this does

1. Reads `config/moe_rail.yaml`
2. Regenerates workload + Rail topology + baseline `system.json` into `generated/`
3. Converts the text workload → Chakra ET files (`workload_et.0.et` … `.15.et`)
4. Runs Astra-SIM analytical binary
5. Parses logs → `generated/results.json`

### What to show peers after MoE

```bash
cat generated/results.json
```

Look for:

| Field | Meaning |
|---|---|
| `Total Time (us)` | **Primary JCT proxy** (lower is better) |
| `Compute Time (us)` | Approx. compute portion |
| `Exposed Communication Time (us)` | Comm on the critical path (scheduler-sensitive) |
| `Total Message Size (MiB, …)` | Contract size from the workload (MoE ≈ 576) |
| `Comm fraction of JCT` | Exposed comm / total time |

Also peek at inputs:

```bash
cat generated/workload/moe_workload.txt
cat generated/topology/network.yml
cat generated/system/system.json
```

---

## Step 2 — Run the LLM-v1 + Rail simulation

```bash
docker compose run --rm --entrypoint bash llm-v1-rail-sim -c \
  'pip install -q "protobuf==7.35.1" && python -m src.run_sim'
```

Same pipeline, different config (`config/llm_v1_rail.yaml`). This **overwrites** MoE outputs in `generated/`.

```bash
cat generated/results.json
```

LLM-v1 message size contract is **320 MiB** (64 + 256). Typical JCT is lower than MoE because there is no EP AllToAll traffic.

---

## Step 3 — Optional local generate (no Astra-SIM)

Useful to inspect artifacts without Docker / without the simulator binary:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.generate --config config/moe_rail.yaml
python -m src.generate --config config/llm_v1_rail.yaml
```

---

## How the pipeline works (explain this to peers)

```text
config/*.yaml
    │
    ▼
src/generate.py          →  generated/workload/*.json + *.txt
                         →  generated/topology/network.yml
                         →  generated/system/system.json
    │
    ▼
chakra_converter Text    →  generated/workload/workload_et.{0..15}.et
    │
    ▼
Astra-SIM analytical     →  generated/astra_sim.log
    │
    ▼
src/results_parser.py    →  generated/results.json   ← JCT metrics
```

### Code map (where to change things)

| Piece | Path | What peers usually edit |
|---|---|---|
| Workload knobs | `config/moe_rail.yaml` or `config/llm_v1_rail.yaml` | sizes, parallelism, payloads |
| MoE graph | `src/workload/moe_workload.py` | which collectives / order |
| LLM graph | `src/workload/llm_workload.py` | which collectives / order |
| Rail topology | `src/topology/rail.py`, `analytical_network.py` | dims, BW, latency |
| Default scheduler | `generated/system/system.json` (from YAML `system:`) | FIFO/LIFO, ring/direct, baseline/themis |
| Run orchestration | `src/run_sim.py` | generate → convert → sim → parse |

Astra-SIM always uses a **built-in collective scheduler** from `system.json`. That is the baseline. Custom schedulers later replace/extend that layer — not the workload/topology files (when comparing fairly).

---

## GENERATE=1 vs GENERATE=0

| Mode | Command idea | Use when |
|---|---|---|
| **Regenerate (default)** | `GENERATE=1` (default) | Changing YAML / wanting fresh inputs |
| **Reuse frozen inputs** | `GENERATE=0 docker compose run --rm --entrypoint bash llm-v1-rail-sim -c 'pip install -q "protobuf==7.35.1" && python -m src.run_sim'` | Comparing schedulers on **identical** workload+topology |

---

## How peers should add their own simulation

Keep the same shape so results stay comparable.

### A) New workload (recommended path)

1. Copy a config:
   - `cp config/llm_v1_rail.yaml config/my_workload_rail.yaml`
2. Edit knobs (`parallelism`, `payloads_mib`, `run.name`).
3. If the graph shape changes (new collectives), copy `src/workload/llm_workload.py` → `my_workload.py` and wire it in `src/generate.py` (same pattern as MoE vs LLM).
4. Add a compose service (copy `llm-v1-rail-sim`) with `CONFIG_PATH` pointing at your YAML.
5. Run Docker, read `generated/results.json`.

### B) New topology (later)

1. Keep the workload fixed (`GENERATE=0` after generating once, or copy artifacts aside).
2. Change `rail:` in YAML (or add a new topology emitter under `src/topology/`).
3. Re-run and compare **Total Time** / **Exposed Communication**.

### C) New scheduler (Phase D target)

1. Freeze workload + topology (`GENERATE=0`).
2. Only change `system.json` (or custom collective ETs).
3. Re-run; compare JCT against the baseline row.

### Suggested future layout (not required yet)

```text
workload/     # moe/, llm/, …
topology/     # moe-rail/, llm-rail/, moe-torus/, …
schedulers/   # NCCL/, TACCL/, ByteScheduler/, SYNDICATE/, CadentFlow/
```

Today: sibling YAML configs + shared generators + overwrite `generated/`.

---

## Demo script (5 minutes)

1. Build once (`docker compose build moe-rail-sim`) if needed.
2. Run MoE → show `results.json` + `moe_workload.txt` + `network.yml`.
3. Run LLM-v1 → show new `results.json` (note overwrite + smaller message size).
4. Open `system.json` → explain this is the **default** scheduler peers will replace.
5. Point at `REQUIREMENTS.md` §7 for the metrics contract.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `protobuf` / Chakra version error | Keep the `pip install -q "protobuf==7.35.1"` prefix (pinned in Dockerfile for next rebuild) |
| `Resource deadlock avoided` on bind mounts | Re-run; writers use temp+replace. Free disk / restart Docker if it persists |
| Disk full during build | Free space (need several GB); prune unused Docker data carefully |
| Want MoE and LLM results together | Copy `generated/results.json` to e.g. `results/moe.json` before switching workloads |

---

## Spec / checklist

Full design notes, acceptance criteria, and metrics definitions: [`REQUIREMENTS.md`](REQUIREMENTS.md).
