#!/usr/bin/env bash
# Container entrypoint: translate the env vars injected by Kubeflow into torchrun flags.
#   - PyTorchJob (kubeflow.org/v1) injects MASTER_ADDR / MASTER_PORT / WORLD_SIZE (= pod count) / RANK (= node rank)
#   - Trainer v2 injects PET_NNODES / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT
# Falls back to --nnodes=1 when none of them are set (plain single-node run).
set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
NNODES="${PET_NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${PET_NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

echo "[entrypoint] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

# These are per-pod values; torchrun sets per-process ones, so clear them to avoid conflicts.
unset RANK WORLD_SIZE LOCAL_RANK

exec torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /workspace/src/train_sft.py "$@"
