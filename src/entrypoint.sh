#!/usr/bin/env bash
# 容器入口：把 Kubeflow PyTorchJob（或 Trainer v2）注入的环境变量翻译成 torchrun 参数。
#   - PyTorchJob(kubeflow.org/v1) 注入：MASTER_ADDR / MASTER_PORT / WORLD_SIZE(=pod 数) / RANK(=node rank)
#   - Trainer v2  注入：PET_NNODES / PET_NODE_RANK / PET_MASTER_ADDR / PET_MASTER_PORT
# 单机跑（没有这些变量）时自动退化成 --nnodes=1。
set -euo pipefail

GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
NNODES="${PET_NNODES:-${WORLD_SIZE:-1}}"
NODE_RANK="${PET_NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${PET_MASTER_ADDR:-${MASTER_ADDR:-127.0.0.1}}"
MASTER_PORT="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"

echo "[entrypoint] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${GPUS_PER_NODE} master=${MASTER_ADDR}:${MASTER_PORT}"

# 这几个是「每个 pod 一份」的值，torchrun 要按「每个进程一份」重新设置，先清掉避免冲突
unset RANK WORLD_SIZE LOCAL_RANK

exec torchrun \
  --nnodes="${NNODES}" \
  --node_rank="${NODE_RANK}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /workspace/src/train_sft.py "$@"
