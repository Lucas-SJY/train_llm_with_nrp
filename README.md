# context-comp SFT on NRP

A minimal, working example of full-parameter SFT on [NRP Nautilus](https://nrp.ai):
one Kubernetes Job, one node, one GPU, plain `transformers` + DeepSpeed. No Kubeflow
operator, no ZeRO sharding -- the DeepSpeed config is stage 0, so DeepSpeed only wraps
the training loop and the single GPU holds the whole model.

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
│   └── entrypoint.sh       # loads .env, then `deepspeed --num_gpus=1`
├── configs/ds.json         # DeepSpeed, ZeRO stage 0
├── run.sh                  # one-command pipeline: data -> image -> secrets -> job -> logs
├── .env.example            # every setting and key; copy to .env (gitignored)
├── k8s/
│   ├── pvc.yaml            # HF cache + checkpoints
│   ├── job.yaml            # the training Job, 1x GPU
│   └── data-shell.yaml     # throwaway pod for kubectl cp
├── Dockerfile
└── requirements.txt
```

`bespoke-v2/` (raw data), `data/` (generated jsonl) and `.env` (keys) are all gitignored.

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

```bash
cp .env.example .env
$EDITOR .env      # fill in K8S_NAMESPACE, IMAGE, NRP_REGISTRY_TOKEN
./run.sh
```

That is the whole thing. `run.sh` builds the dataset, logs in to the NRP registry,
builds and pushes the image, refreshes both Secrets, creates the PVC if missing,
submits the Job, and tails the logs.

Only three values are mandatory:

| Variable | Where it comes from |
|---|---|
| `K8S_NAMESPACE` | the NRP namespace you were added to |
| `IMAGE` | `gitlab-registry.nrp-nautilus.io/<your-gitlab-namespace>/context-comp-sft:latest` |
| `NRP_REGISTRY_TOKEN` | a GitLab token with `read_registry` + `write_registry`, from https://gitlab.nrp-nautilus.io/-/user_settings/personal_access_tokens |

`NRP_REGISTRY_USER` is derived from `IMAGE` -- the path segment after the host is your
GitLab namespace, which is also the registry login user for a personal access token.
Set it explicitly only for a deploy token, whose user is the token name.

### Individual steps

```bash
./run.sh data       # rebuild data/{train,val}.jsonl only
./run.sh image      # docker login + build + push
./run.sh secrets    # re-upload .env as the sft-env Secret, refresh the pull secret
./run.sh submit     # (re)submit the Job, substituting ${IMAGE} into the manifest
./run.sh logs       # wait for the pod, then follow
./run.sh status     # job, pods, recent events
./run.sh clean      # delete the job (Secrets and PVC survive)
```

After editing `.env`, run `./run.sh secrets && ./run.sh submit` -- a running pod does
not pick up changes, and the Secret is a snapshot. Resubmitting always deletes the old
Job first, because a Job's pod template is immutable.

For a fast smoke test, shrink the dataset first with
`python3 src/prepare_data.py --max-samples 200`.

The first job downloads the model from HuggingFace into `/data/hf` on the PVC; later
jobs hit that cache. `.env` is excluded by `.dockerignore`, so keys never end up in an
image layer.

### Retrieve the weights

Checkpoints land in `/data/runs/qwen-sft` on the PVC:

```bash
kubectl apply -f k8s/data-shell.yaml
kubectl cp data-shell:/data/runs/qwen-sft ./qwen-sft
kubectl delete pod data-shell               # delete as soon as you are done
./run.sh clean
```

## Configuration

### Where settings live

| | |
|---|---|
| `.env` | model, keys, all hyperparameters. Gitignored, never baked into the image. |
| `.env.example` | the tracked template; keep it in sync when adding a variable. |
| `run.sh` | the only entry point; reads `.env` and drives docker + kubectl. |
| `k8s/job.yaml` | hardware only: GPU type, cpu/memory, volumes. |
| `configs/ds.json` | DeepSpeed config. |

`src/train_sft.py` takes every default from an environment variable
(`MODEL_NAME`, `MAX_SEQ_LEN`, `LEARNING_RATE`, ...), and CLI flags still override them,
so a one-off experiment is `--lr 5e-6` without touching `.env`. Locally,
`src/entrypoint.sh` sources `.env` from the repo root; on the cluster the values arrive
through `envFrom: secretRef: sft-env`.

One k8s gotcha this design avoids: `$(VAR)` substitution inside a container's `args`
only resolves variables declared in that container's `env:` list, never ones coming
from `envFrom`. Reading the environment inside Python sidesteps it entirely.

### DeepSpeed

`configs/ds.json` is as small as a DeepSpeed config gets: bf16 on,
`zero_optimization.stage: 0`, and every batch-size field left as `auto` so accelerate
fills them in from the CLI flags. On a single GPU the ZeRO stages have nothing to shard
across, so stage 0 is the right setting; the only lever DeepSpeed still offers for a
larger model is CPU offload (see below).

The launcher is `deepspeed --num_gpus=1`, called from `src/entrypoint.sh`. It exists to
set up the one-process distributed group that DeepSpeed initialisation expects; running
`python src/train_sft.py` directly would have to invent `MASTER_ADDR` / `RANK` itself.

### Model size

This is the real constraint of one GPU without sharding: **that GPU stores the full
model, the full gradients and the full Adam state**. For bf16 training with an fp32 Adam
master copy that is roughly `16 bytes x parameter count`, before activations:

| Model | GPU memory needed | A40 (48G) | A100 (80G) |
|---|---|---|---|
| Qwen3-0.6B | ~10GB | yes | yes |
| Qwen3-1.7B | ~27GB | yes -- the default | yes |
| Qwen3-4B | ~64GB | no | yes |
| Qwen3-8B | ~128GB | no | no |

To go past that on a single GPU, cheapest to most invasive:

1. **LoRA / QLoRA** -- only adapter weights get optimizer state, so 8B fits on one A40.
   Needs `peft` in `requirements.txt` and a few lines in `train_sft.py`.
2. **ZeRO-2 with CPU offload** -- set `"stage": 2` in `configs/ds.json` and add
   `"offload_optimizer": {"device": "cpu"}`. This moves the Adam state to host RAM, so
   raise the pod `memory` request accordingly (~12 bytes per parameter). Slower per step.
3. **More GPUs** -- put the GPU count in `k8s/job.yaml`, switch the entrypoint back to
   `deepspeed --num_gpus=N`, and use ZeRO-3 to shard across them. That is what the
   Kubeflow PyTorchJob route existed for; for one GPU it is pure overhead.

### Which model

`MODEL_NAME` in `.env` takes any HF repo id. If you are unsure whether one exists,
check with
`curl -s -o /dev/null -w '%{http_code}\n' https://huggingface.co/api/models/<repo-id>`.
Gated or private repos need `HF_TOKEN` in the same file; set `HF_ENDPOINT` to use a
mirror.

### GPU type

NRP uses dedicated resource keys to select high-memory cards: `nvidia.com/a40`,
`nvidia.com/a100`, `nvidia.com/h100`, and so on (see
[GPU pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/)).
**A100/H100/H200/GH200 are gated by a per-namespace ResourceQuota that defaults to
zero** and must be requested separately; A40 is generally available, hence the default.
Check what your namespace may use with `kubectl describe resourcequota`.

### Hyperparameters

All of them live in `.env`; run `python3 src/train_sft.py --help` for the full list and
the matching variable names. Defaults: lr 1e-5 with cosine decay, 2 epochs, micro-bs 1
x grad-accum 32 on 1 GPU = global batch 32, checkpoint every 500 steps. Set
`REPORT_TO=wandb` plus `WANDB_API_KEY` to log a run.

On OOM, lower `--max-seq-len` first (4096 -> 2048; most samples in this dataset are
under 3k tokens), then pick a smaller model or one of the three options above.

## NRP rules worth knowing

- **Never run an interactive `sleep infinity` pod.** NRP explicitly forbids it and bans
  accounts over it. Use Jobs even while developing; `data-shell.yaml` uses a bounded
  `sleep 14400` and should be deleted after use.
- **Only PVC data survives** -- the container disk and emptyDir are wiped on restart.
- **Pods writing more than 50Gi of ephemeral data get evicted**, which is why `HF_HOME`
  points at the PVC and `ephemeral-storage` is requested explicitly.
- Always set both `requests` and `limits`. GPU requests and limits must be equal, and
  an admission policy rejects a cpu limit/request ratio above 1.2 -- keeping them equal
  is simplest.
- Never force delete a pod. PVCs untouched for six months may be purged.
- Secrets are namespace-scoped and only base64-encoded, not encrypted; anyone with
  access to the namespace can read `sft-env`. Use a scoped, revocable token.

## Troubleshooting

**Pod stuck in Pending** -- `kubectl describe pod <name>` and read the events. Usually no
free A40 node, or the requested GPU type is quota-limited.

**CUDA out of memory** -- expected past ~2.5B parameters on a 48G card; see the model
size table above.

**ImagePullBackOff** -- either the `nrp-registry` Secret is stale (`./run.sh secrets`)
or `NRP_REGISTRY_USER` is wrong. Confirm the token works locally first:
`echo $NRP_REGISTRY_TOKEN | docker login gitlab-registry.nrp-nautilus.io -u $NRP_REGISTRY_USER --password-stdin`.
A pod whose image is literally `${IMAGE}` means the manifest was applied with plain
`kubectl` instead of `./run.sh submit`.

**Settings changed in .env but the job ignores them** -- the Secret is a snapshot. Run
`./run.sh secrets && ./run.sh submit`. Check what the pod actually got with
`kubectl exec <pod> -- env | sort`.

**PVC stuck in Pending** -- the storage class in `k8s/pvc.yaml` must exist on the
cluster; list them with `kubectl get storageclass`.

**DeepSpeed fails to compile** -- the base image must be a `devel` one with nvcc; the
`runtime` images cannot build the JIT ops.

## References

- NRP: [Getting Started](https://nrp.ai/documentation/userdocs/start/getting-started/) ·
  [GPU pods](https://nrp.ai/documentation/userdocs/running/gpu-pods/) ·
  [Batch jobs](https://nrp.ai/documentation/userdocs/running/jobs/) ·
  [Storage](https://nrp.ai/documentation/userdocs/storage/intro/)
- [Accelerate: DeepSpeed](https://huggingface.co/docs/accelerate/en/usage_guides/deepspeed)
- [DeepSpeed docs](https://www.deepspeed.ai/getting-started/)
