#!/usr/bin/env bash
# Train one run with torchrun; running it again continues from the run's latest checkpoint (--resume auto).
#
#   bash scripts/train.sh <PROCESSES_PER_NODE> <RUN_FILE> [--set key=value ...]
#
#   bash scripts/train.sh 8 configs/runs/flux2_klein_4b/edmd.yaml
#   bash scripts/train.sh 8 configs/runs/flux2_klein_4b/edmd.yaml --set method.updates.max_steps=3000
#
# Several nodes: set NNODES, NODE_RANK and MASTER_ADDR (optionally MASTER_PORT, default 29500) on every node.
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 bash scripts/train.sh 8 configs/runs/flux2_klein_9b/edmd.yaml
set -euo pipefail

usage="usage: bash scripts/train.sh <PROCESSES_PER_NODE> <RUN_FILE> [--set key=value ...]"
NPROC="${1:?$usage}"
RUN_FILE="${2:?$usage}"
shift 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO}${PYTHONPATH:+:${PYTHONPATH}}"

NNODES="${NNODES:-1}"
if [[ "${NNODES}" -gt 1 ]]; then
  LAUNCH=(--nnodes "${NNODES}" --node_rank "${NODE_RANK:?set NODE_RANK on every node}"
          --master_addr "${MASTER_ADDR:?set MASTER_ADDR (node 0) on every node}" --master_port "${MASTER_PORT:-29500}")
else
  LAUNCH=(--standalone)
fi

exec torchrun "${LAUNCH[@]}" --nproc_per_node "${NPROC}" -m edmd.cli.train --run-file "${RUN_FILE}" --resume auto "$@"
