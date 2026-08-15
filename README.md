# context-comp SFT on NRP

A minimal, working example of full-parameter SFT for a Qwen 8B model on
[NRP Nautilus](https://nrp.ai), using Kubeflow `PyTorchJob` + DeepSpeed ZeRO-3.

The data lives in `bespoke-v2/`: each sample is a DeepSeek-R1 reasoning trace cut into
spans, and every span carries a label (`logical_deduction` / `reflecting` /
`verifying` / ...). This repo turns it into a **context-compression** SFT task:

```
input  = question
output = the reasoning trace with the redundant labels removed
```

By default it drops `planning_next_step,restating_problem,reflecting,verifying`, which
keeps **56%** of the tokens (4.71M / 8.41M over 5144 samples). That objective is only a
placeholder default -- to try a different compression policy, change `--drop-labels` or
edit `build_example` in [prepare_data.py](src/prepare_data.py).

## Layout

```
.
├── src/
│   ├── prepare_data.py     # bespoke-v2/*.json -> data/{train,val}.jsonl
│   ├── train_sft.py        # HF Trainer + DeepSpeed, loss on the assistant turn only
│   └── entrypoint.sh       # turns the env vars injected by k8s into torchrun flags
├── configs/ds_zero3.json   # DeepSpeed ZeRO-3
├── k8s/
│   ├── pvc.yaml            # HF cache + checkpoints
│   ├── pytorchjob.yaml     # <- main path (NRP runs Training Operator v1)
│   ├── trainjob.yaml       # optional: Kubeflow Trainer v2, may not be installed
│   └── data-shell.yaml     # throwaway pod for kubectl cp
├── Dockerfile
└── requirements.txt
```

`bespoke-v2/` (raw data) and `data/` (generated jsonl) are both gitignored.

## Prerequisites

1. An NRP account attached to a namespace, plus local `kubectl` and
   [kubelogin](https://github.com/int128/kubelogin), with the kubeconfig saved to
   `~/.kube/config` (see
   [Getting Started](https://nrp.ai/documentation/userdocs/start/getting-started/)).
   ```bash
   kubectl config set-context nautilus --namespace=<YOUR_NAMESPACE>
   kubectl get pods          # "No resources found" means it works
   ```
2. Push access to `gitlab-registry.nrp-nautilus.io` (an NRP GitLab account).
3. Docker or Podman locally.

## Quick start

### 0. Build the dataset (locally)

```bash
python3 src/prepare_data.py --input-dir bespoke-v2 --output-dir data
# input files    : 5144 (0 skipped)
# train / val    : 5042 / 102  -> data/
# tokens kept    : 56.0% (4,712,631 / 8,412,626)
```

Add `--max-samples 200` for a smoke test that finishes an epoch in under a minute.

### 1. Build and push the image

`data/` is baked into the image (25MB), so rebuild after changing the dataset.

```bash
export IMAGE=gitlab-registry.nrp-nautilus.io/<your-user>/context-comp-sft:latest
docker login gitlab-registry.nrp-nautilus.io
docker build --platform linux/amd64 -t $IMAGE .
docker push $IMAGE
```

Then set the same address in `image:` inside `k8s/pytorchjob.yaml`.

### 2. Create the PVC

```bash
kubectl get storageclass                    # confirm what exists, then edit pvc.yaml
kubectl apply -f k8s/pvc.yaml
kubectl get pvc qwen-sft-data               # wait for Bound
```

### 3. Submit the job

```bash
kubectl apply -f k8s/pytorchjob.yaml
kubectl get pytorchjob qwen-sft
kubectl get pods -l training.kubeflow.org/job-name=qwen-sft
kubectl logs -f qwen-sft-master-0
```

The first run downloads the model from HuggingFace into `/data/hf` on the PVC (roughly
fifteen minutes); later jobs hit that cache.

### 4. Retrieve the weights

Checkpoints land in `/data/runs/qwen-sft` on the PVC:

```bash
kubectl apply -f k8s/data-shell.yaml
kubectl cp data-shell:/data/runs/qwen-sft ./qwen-sft
kubectl delete pod data-shell               # delete as soon as you are done
kubectl delete pytorchjob qwen-sft
```

## Configuration

### Model

`MODEL_NAME` in `k8s/pytorchjob.yaml` defaults to `Qwen/Qwen3-8B`. **Replace it with the
exact HF repo id of the checkpoint you want** (for example the 8B model of the Qwen3.5
series). If you are unsure whether a repo id exists, check it with
`curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/<repo-id>`.
Gated models also need the `HF_TOKEN` secret described in the yaml comments.

### GPU memory

8B full-parameter + ZeRO-3 + bf16 needs roughly **128GB** for optimizer states, gradients
and parameters combined, sharded across the GPUs:

| Setup | Total VRAM | Verdict |
|---|---|---|
| 4x A40 (48G) = 192G | 128G of state + ~16G/GPU headroom | the default, fits |
| 8x A40 = 384G | plenty of room | faster; halve `grad-accum` accordingly |
| 2x A40 = 96G | not enough | requires CPU offload, see below |

On OOM, try in this order:

1. Lower `--max-seq-len` (4096 -> 2048); most samples in this dataset are under 3k tokens.
2. Add GPUs (edit `resources`; `entrypoint.sh` follows `nvidia-smi` automatically).
3. Enable offload: set `offload_optimizer.device` to `"cpu"` in `configs/ds_zero3.json`
   and raise the pod `memory` request above 200Gi (an 8B model's optimizer state needs
   ~96GB of host RAM).

### GPU type

NRP uses dedicated resource keys to select high-memory cards: `nvidia.com/a40`,
`nvidia.com/a100`, `nvidia.com/h100`, and so on (see
[GPU pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/)).
**A100/H100/H200/GH200 are gated by a per-namespace ResourceQuota that defaults to zero**
and must be requested separately; A40 is generally available, hence the default here.

### Hyperparameters

All of them live in the `args:` block of `k8s/pytorchjob.yaml`; run
`python3 src/train_sft.py --help` for the full list. Defaults: lr 1e-5 with cosine decay,
2 epochs, micro-bs 1 x grad-accum 8 x 4 GPUs = global batch 32, checkpoint every 500 steps.

## NRP rules worth knowing

- **Never run an interactive `sleep infinity` pod.** NRP explicitly forbids it and bans
  accounts over it. Use Jobs/PyTorchJobs even while developing; `data-shell.yaml` uses a
  bounded `sleep 14400` and should be deleted after use.
- **Only PVC data survives** -- the container disk and emptyDir are wiped on restart.
- **Pods writing more than 50Gi of ephemeral data get evicted**, which is why `HF_HOME`
  points at the PVC and `ephemeral-storage` is requested explicitly.
- Always set both `requests` and `limits`; GPU requests and limits must be equal.
- Never force delete a pod. PVCs untouched for six months may be purged.

## Optional: Kubeflow Trainer v2

The NRP [Kubeflow page](https://nrp.ai/documentation/userdocs/running/kubeflow/) documents
Training Operator **v1** (`kubeflow.org/v1` `TFJob` / `PyTorchJob`), so `pytorchjob.yaml`
is the main path. If your cluster also has [Trainer v2](https://trainer.kubeflow.org/):

```bash
kubectl get crd | grep trainjob
kubectl get clustertrainingruntime      # torch-distributed must be present
kubectl apply -f k8s/trainjob.yaml
```

`entrypoint.sh` supports both (v1 reads `MASTER_ADDR/RANK/WORLD_SIZE`, v2 reads `PET_*`).
Note that v2's `spec.trainer` cannot declare `volumeMounts` -- mounting a PVC goes through
`runtimePatches` -- so `trainjob.yaml` is a runnable skeleton without persistent storage.

## Troubleshooting

**Pod stuck in Pending** -- `kubectl describe pod <name>` and read the events. Usually no
free A40 node, or the requested GPU type is quota-limited.

**NCCL hangs or times out** -- make sure `/dev/shm` is mounted (the 64MB default is too
small; the yaml provides 16Gi). For multi-node runs set `NCCL_DEBUG=INFO` to see the
handshake.

**DeepSpeed fails to compile** -- the base image must be a `devel` one with nvcc; the
`runtime` images cannot build the JIT ops.

**Saved model is only a few KB** -- under ZeRO-3 the sharded weights are consolidated by
`stage3_gather_16bit_weights_on_model_save: true` in `configs/ds_zero3.json`. Keep it on.

## References

- NRP: [Getting Started](https://nrp.ai/documentation/userdocs/start/getting-started/) ·
  [GPU pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/) ·
  [Batch jobs](https://nrp.ai/documentation/userdocs/running/jobs/) ·
  [Storage](https://nrp.ai/documentation/userdocs/storage/intro/) ·
  [Kubeflow](https://nrp.ai/documentation/userdocs/running/kubeflow/)
- [Kubeflow Trainer](https://trainer.kubeflow.org/en/latest/user-guides/pytorch.html) ·
  [DeepSpeed runtime](https://trainer.kubeflow.org/en/latest/user-guides/deepspeed.html)
- [Accelerate: DeepSpeed](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
