#!/usr/bin/env bash
# Container entrypoint: single node, single GPU, launched by the deepspeed CLI.
#
# The launcher is what sets up the (one-process) distributed group DeepSpeed needs;
# `python train_sft.py` alone would have to invent MASTER_ADDR/RANK itself.
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

echo "[entrypoint] model=${MODEL_NAME:-<default>} gpus=1"

exec deepspeed --num_gpus=1 /workspace/src/train_sft.py "$@"
