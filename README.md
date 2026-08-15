# context-comp SFT on NRP

在 [NRP Nautilus](https://nrp.ai) 上用 Kubeflow `PyTorchJob` + DeepSpeed ZeRO-3 对 Qwen 8B 做全参 SFT 的**最小可用示例**。

数据来自 `bespoke-v2/`：每条样本是一段被切成若干 span 的 DeepSeek-R1 推理轨迹，每个 span 带一个标签
（`logical_deduction` / `reflecting` / `verifying` / …）。本仓库把它转成一个**上下文压缩**的 SFT 任务：

```
输入  = question
输出  = 丢掉冗余标签之后的推理轨迹
```

默认丢掉 `planning_next_step,restating_problem,reflecting,verifying`，实测保留 **56%** 的 token
（4.71M / 8.41M，5144 条样本）。这个目标只是个占位的默认值，换成别的压缩策略只要改
`--drop-labels`，或者直接改 [prepare_data.py](src/prepare_data.py) 里的 `build_example`。

## 目录结构

```
.
├── src/
│   ├── prepare_data.py     # bespoke-v2/*.json -> data/{train,val}.jsonl
│   ├── train_sft.py        # HF Trainer + DeepSpeed，只对 assistant 部分算 loss
│   └── entrypoint.sh       # 把 k8s 注入的环境变量翻译成 torchrun 参数
├── configs/ds_zero3.json   # DeepSpeed ZeRO-3
├── k8s/
│   ├── pvc.yaml            # 存 HF 缓存 + checkpoint
│   ├── pytorchjob.yaml     # ← 主路径（NRP 上装的是 Training Operator v1）
│   ├── trainjob.yaml       # 可选：Kubeflow Trainer v2，NRP 不一定装了
│   └── data-shell.yaml     # 临时 pod，用来 kubectl cp
├── Dockerfile
└── requirements.txt
```

`bespoke-v2/`（原始数据）和 `data/`（生成的 jsonl）都在 `.gitignore` 里，不进 git。

## 前置条件

1. NRP 账号 + 已加入某个 namespace，本地装好 `kubectl` 和
   [kubelogin](https://github.com/int128/kubelogin)，kubeconfig 放到 `~/.kube/config`
   （见 [Getting Started](https://nrp.ai/documentation/userdocs/start/getting-started/)）。
   ```bash
   kubectl config set-context nautilus --namespace=<YOUR_NAMESPACE>
   kubectl get pods          # 输出 No resources found 就说明通了
   ```
2. 能往 `gitlab-registry.nrp-nautilus.io` 推镜像（NRP GitLab 账号）。
3. 本地有 Docker/Podman。

## 快速开始

### 0. 生成训练数据（本地）

```bash
python3 src/prepare_data.py --input-dir bespoke-v2 --output-dir data
# 输入文件      : 5144（跳过 0）
# train / val   : 5042 / 102  -> data/
# 保留 token 占比: 56.0%（4,712,631 / 8,412,626）
```

先跑通流程的话加 `--max-samples 200`，几十秒就能跑完一轮训练。

### 1. 构建并推送镜像

`data/` 会被打进镜像（25MB，可以接受），所以改了数据要重新 build。

```bash
export IMAGE=gitlab-registry.nrp-nautilus.io/<你的用户名>/context-comp-sft:latest
docker login gitlab-registry.nrp-nautilus.io
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE
```

然后把 `k8s/pytorchjob.yaml` 里的 `image:` 改成同一个地址。

### 2. 创建 PVC

```bash
kubectl get storageclass                    # 先确认集群里有哪些，再改 pvc.yaml
kubectl apply -f k8s/pvc.yaml
kubectl get pvc qwen-sft-data               # 等它变成 Bound
```

### 3. 提交训练

```bash
kubectl apply -f k8s/pytorchjob.yaml
kubectl get pytorchjob qwen-sft
kubectl get pods -l training.kubeflow.org/job-name=qwen-sft
kubectl logs -f qwen-sft-master-0
```

第一次跑会先从 HuggingFace 下模型到 `/data/hf`（PVC 上），大概十几分钟；之后的 job 会直接命中缓存。

### 4. 取回权重

Checkpoint 在 PVC 的 `/data/runs/qwen-sft`：

```bash
kubectl apply -f k8s/data-shell.yaml
kubectl cp data-shell:/data/runs/qwen-sft ./qwen-sft
kubectl delete pod data-shell               # 用完立刻删
kubectl delete pytorchjob qwen-sft
```

## 配置说明

### 模型

`k8s/pytorchjob.yaml` 里的 `MODEL_NAME` 默认是 `Qwen/Qwen3-8B`。**换成你要训的那个
checkpoint 的准确 HF repo id**（比如 Qwen3.5 系列的 8B），拿不准就先
`curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/<repo-id>` 验一下；
gated 模型还要按 yaml 里注释的写法挂 `HF_TOKEN` secret。

### 显存

8B 全参 + ZeRO-3 + bf16，优化器/梯度/参数三份加起来约 **128GB**，均摊到各卡上：

| 配置 | 总显存 | 能不能跑 |
|---|---|---|
| 4×A40 (48G) = 192G | 128G 状态 + ~16G/卡 余量 | 默认配置，够用 |
| 8×A40 = 384G | 很宽裕 | 想快就用这个，`grad-accum` 相应减半 |
| 2×A40 = 96G | 不够 | 必须开 CPU offload，见下 |

OOM 时按这个顺序调：

1. 降 `--max-seq-len`（4096 → 2048），本数据集大部分样本都在 3k token 以内；
2. 加卡（改 `resources` 里的 GPU 数，`entrypoint.sh` 会自动跟着 `nvidia-smi` 走）；
3. 开 offload：把 `configs/ds_zero3.json` 里 `offload_optimizer.device` 改成 `"cpu"`，
   同时把 pod 的 `memory` 请求提到 200Gi 以上（8B 的优化器状态在 CPU 上要 ~96GB）。

### GPU 型号

NRP 用专门的 resource key 选高显存卡：`nvidia.com/a40`、`nvidia.com/a100`、`nvidia.com/h100` 等
（见 [GPU pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/)）。
**A100/H100/H200/GH200 受 per-namespace ResourceQuota 限制，默认配额是 0**，要单独申请；
A40 一般可以直接用，所以这里默认 A40。

### 超参

都在 `k8s/pytorchjob.yaml` 的 `args:` 里，全部参数看 `python3 src/train_sft.py --help`。
默认：lr 1e-5 / cosine、2 epoch、micro-bs 1 × grad-accum 8 × 4 卡 = 全局 batch 32、每 500 步存一次。

## NRP 的几条硬规矩

- **不要跑 `sleep infinity` 的交互式 pod**，NRP 明确禁止，会被封号。开发调试也用 Job/PyTorchJob，
  `data-shell.yaml` 用的是有限时长的 `sleep 14400`，用完就删。
- **数据只有写在 PVC 上才留得住**，容器盘和 emptyDir 在 pod 重启后清零。
- **ephemeral 写超过 50Gi 的 pod 会被驱逐**，所以 `HF_HOME` 指到了 PVC，并显式请求了
  `ephemeral-storage`。
- `requests` 和 `limits` 都要写，GPU 的 requests/limits 必须相等。
- 别 force delete pod；PVC 半年不访问会被清理。

## 可选：Kubeflow Trainer v2

NRP 文档里的 [Kubeflow](https://nrp.ai/documentation/userdocs/running/kubeflow/) 页面给的是
Training Operator **v1**（`kubeflow.org/v1` 的 `TFJob` / `PyTorchJob`），所以主路径用
`pytorchjob.yaml`。如果集群另外装了 [Trainer v2](https://trainer.kubeflow.org/)：

```bash
kubectl get crd | grep trainjob
kubectl get clustertrainingruntime      # 需要有 torch-distributed
kubectl apply -f k8s/trainjob.yaml
```

`entrypoint.sh` 两种都认（v1 读 `MASTER_ADDR/RANK/WORLD_SIZE`，v2 读 `PET_*`）。
但 v2 的 `spec.trainer` 不能直接写 `volumeMounts`，挂 PVC 要走 `runtimePatches`，
所以 `trainjob.yaml` 只是个能跑起来的骨架，没接持久化存储。

## 常见问题

**pod 一直 Pending** — `kubectl describe pod <name>` 看事件。多半是没有空闲的 A40 节点，
或者请求的 GPU 型号有配额限制。

**NCCL 卡住 / timeout** — 确认 `/dev/shm` 挂上了（默认 64MB 不够，yaml 里给了 16Gi）。
多节点时把 `NCCL_DEBUG` 调成 `INFO` 看握手日志。

**DeepSpeed 编译失败** — 基础镜像必须是 `devel`（带 nvcc），`runtime` 镜像编不了 JIT 算子。

**保存下来的模型只有几 KB** — ZeRO-3 下要靠 `stage3_gather_16bit_weights_on_model_save: true`
把分片权重汇总，这个已经在 `configs/ds_zero3.json` 里开了，别关。

## 参考

- [NRP: Getting Started](https://nrp.ai/documentation/userdocs/start/getting-started/) ·
  [GPU pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/) ·
  [Batch jobs](https://nrp.ai/documentation/userdocs/running/jobs/) ·
  [Storage](https://nrp.ai/documentation/userdocs/storage/intro/) ·
  [Kubeflow](https://nrp.ai/documentation/userdocs/running/kubeflow/)
- [Kubeflow Trainer](https://trainer.kubeflow.org/en/latest/user-guides/pytorch.html) ·
  [DeepSpeed runtime](https://trainer.kubeflow.org/en/latest/user-guides/deepspeed.html)
- [Accelerate: DeepSpeed](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
