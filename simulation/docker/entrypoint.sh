#!/usr/bin/env bash
# Docker entrypoint: GENERATE=1 regenerates inputs; then run sim + parse results.
set -euo pipefail

cd /workspace

echo "=== MoE Rail Astra-SIM (Phase C) ==="
echo "GENERATE=${GENERATE:-1}"
echo "CONFIG_PATH=${CONFIG_PATH:-/workspace/config/moe_rail.yaml}"
echo "ASTRA_SIM_HOME=${ASTRA_SIM_HOME:-/opt/astra-sim}"

# Ensure output dirs exist for volume mounts.
mkdir -p /workspace/generated/workload \
         /workspace/generated/topology \
         /workspace/generated/system

exec python -m src.run_sim
