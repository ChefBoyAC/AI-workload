#!/bin/bash
# Generate workloads and run LLM-Torus and MoE-Torus on ASTRA-sim.
# Usage:  ASTRA_SIM_DIR=~/astra-sim ./run_all.sh
set -e
KIT="$(cd "$(dirname "$0")" && pwd)"
ASTRA_SIM_DIR="${ASTRA_SIM_DIR:-$HOME/astra-sim}"
export CHAKRA_DIR="$ASTRA_SIM_DIR/extern/graph_frontend/chakra"
BIN="$ASTRA_SIM_DIR/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Unaware"

NETWORK="$KIT/network/torus3d_4x4x4.yml"
SYSTEM="$KIT/system/torus3d_ring.json"
MEMORY="$KIT/remote_memory/no_memory_expansion.json"
mkdir -p "$KIT/results"

python3 "$KIT/scripts/gen_workload.py" --model llm --out "$KIT/workload/llm_torus/llm"
python3 "$KIT/scripts/gen_workload.py" --model moe --out "$KIT/workload/moe_torus/moe"

for m in llm moe; do
  echo "=== ${m^^} on 3D torus ==="
  "$BIN" \
    --workload-configuration="$KIT/workload/${m}_torus/$m" \
    --system-configuration="$SYSTEM" \
    --remote-memory-configuration="$MEMORY" \
    --network-configuration="$NETWORK" \
    --comm-group-configuration=empty > "$KIT/results/${m}_torus.log" 2>&1
  grep "sys\[0\]" "$KIT/results/${m}_torus.log" | grep -E "Wall time|Comm time|GPU time|overlap"
done
