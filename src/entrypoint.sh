#!/usr/bin/env bash
# Container entrypoint: translate the env vars injected by Kubeflow into torchrun flags.
#   - PyTorchJob (kubeflow.org/v1) injects MASTER_ADDR / MASTER_PORT / WORLD_SIZE (= pod count) / RANK (= node rank)
#   - Trainer v2 injects PET_NNODES / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT
# Falls back to --nnodes=1 when none of them are set (plain single-node run).
set -euo pipefail

# On the cluster the config arrives as the sft-env Secret (envFrom). For local runs,
# fall back to .env in the repo root. Variables already present in the environment win,
# so `MODEL_NAME=... bash src/entrypoint.sh` stays an override rather than being
# clobbered by the file. Expects plain KEY=value lines: no quoting, no inline comments.
ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"
if [ -f "${ENV_FILE}" ]; then
  echo "[entrypoint] loading ${ENV_FILE} (existing env vars take precedence)"
  while IFS= read -r line || [ -n "${line}" ]; do
    case "${line}" in '' | '#'*) continue ;; esac
    key="${line%%=*}"
    case "${key}" in '' | *[!A-Za-z0-9_]*) continue ;; esac
    [ -z "${!key+set}" ] && export "${key}=${line#*=}"
  done < "${ENV_FILE}"
fi

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
NNODES="${PET_NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${PET_NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

echo "[entrypoint] model=${MODEL_NAME:-<default>} nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

# These are per-pod values; torchrun sets per-process ones, so clear them to avoid conflicts.
unset RANK WORLD_SIZE LOCAL_RANK

exec torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /workspace/src/train_sft.py "$@"
