#!/usr/bin/env bash
# Matrix entrypoint: WORKLOAD × TOPOLOGY × SCHEDULER on BACKEND (default ns-3).
set -euo pipefail

cd /workspace

# Ensure protobuf runtime matches Chakra gencode major version.
python -m pip install -q "protobuf==7.35.1" >/dev/null 2>&1 || true

echo "=== ASTRA-sim matrix Docker entrypoint ==="
echo "WORKLOAD=${WORKLOAD:-llm-v1}"
echo "TOPOLOGY=${TOPOLOGY:-rail}"
echo "SCHEDULER=${SCHEDULER:-nccl}"
echo "BACKEND=${BACKEND:-ns3}"
echo "GENERATE=${GENERATE:-1}"
echo "ASTRA_SIM_HOME=${ASTRA_SIM_HOME:-/opt/astra-sim}"

mkdir -p /workspace/generated/workload \
         /workspace/generated/topology \
         /workspace/generated/system \
         /workspace/generated/ns3

exec python -m src.run_matrix
