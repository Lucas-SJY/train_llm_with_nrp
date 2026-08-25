# MATH-500 evaluation

Scores a fine-tuned checkpoint on [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
and reports the two numbers the context-compression experiment turns on:

| metric | question it answers |
|---|---|
| `accuracy` | did the model still get the answer right |
| `completion_tokens_mean` | how much reasoning it needed to get there |

Self-contained: nothing outside this directory is modified. The training image is
reused as-is and the eval script is mounted into it from a ConfigMap, so there is no
image rebuild and no change to `requirements.txt`, `Dockerfile`, `src/` or `run.sh`.

## Layout

```
evaluate/
├── eval_math500.py       # generation + scoring, runs inside the pod
├── k8s/eval-job.yaml     # the Job, mounts the script from a ConfigMap
├── run_eval.sh           # code -> job -> logs, same shape as ../run.sh
└── README.md
```

## Run it

```bash
./evaluate/run_eval.sh
```

That uploads the script as a ConfigMap, submits the Job, and tails the logs. It reads
`IMAGE` and `K8S_NAMESPACE` from `../.env` and writes nothing back.

Individual steps:

```bash
./evaluate/run_eval.sh code      # re-upload the script after editing it
./evaluate/run_eval.sh submit    # (re)submit the Job
./evaluate/run_eval.sh logs      # follow
./evaluate/run_eval.sh status    # job, pod, events
./evaluate/run_eval.sh fetch     # copy results to ./eval-results
./evaluate/run_eval.sh clean     # delete the Job
```

Editing `eval_math500.py` only needs `run_eval.sh code && run_eval.sh submit` -- the
ConfigMap is the deploy mechanism, so iteration costs seconds instead of a rebuild.

## What it does

1. Loads the model from `/data/runs/qwen3-8b-sft` on the PVC (no download).
2. Builds each prompt with the **same instruction the model was trained on** --
   `Return your final response within \boxed{}.` followed by the problem, rendered
   through the chat template with `add_generation_prompt=True`. Evaluating with a
   different instruction would measure a distribution the model was never trained on.
3. Generates greedily (`temperature 0`) in batches, with left padding, which
   decoder-only batched generation requires.
4. Extracts the last `\boxed{...}` with proper brace matching and scores it.

## Scoring

`sympy.parsing.latex` needs antlr, which the training image does not have, so answers
are compared by LaTeX normalisation plus a numeric fallback:
`\dfrac`→`\frac`, `\left`/`\right` and spacing stripped, `\text{}` unwrapped, `^\circ`
and `%` dropped, thousands separators removed, then exact match, then float comparison
of fractions.

Validated two ways before first use: extracting `\boxed{}` from MATH-500's own
`solution` field and scoring it against the `answer` field matches **500/500**, and 14
hand-written edge cases (`0.5` vs `\frac{1}{2}`, `90^\circ` vs `90`, multiple boxes,
missing box) all behave as intended.

It is still a string-level scorer. Algebraically equivalent but textually different
answers (`(x+1)^2` vs `x^2+2x+1`) count as wrong, so treat the number as a consistent
relative measure between checkpoints rather than an absolute leaderboard score.

## Output

Written to `/data/eval/<run-name>/` on the PVC, where `<run-name>` defaults to the
model directory name:

- `summary.json` -- accuracy, token statistics, breakdown by level and subject
- `predictions.jsonl` -- one row per problem with the full completion, for eyeballing

`run_eval.sh fetch` pulls the directory to `../eval-results`.

## Comparing checkpoints

The point is the comparison, and it is a matter of overriding one env var in
`k8s/eval-job.yaml`:

| `EVAL_MODEL` | what it measures |
|---|---|
| `Qwen/Qwen3-8B` | the base model, before compression training |
| `/data/runs/qwen3-8b-sft` | the compressed-CoT fine-tune |
| `/data/runs/qwen-sft-v2` | the 1.7B fine-tune, for a size comparison |

Each run writes to its own subdirectory, so results accumulate rather than overwrite.
Compare `accuracy` against `completion_tokens_mean`: the experiment succeeds if the
fine-tune holds accuracy while spending noticeably fewer tokens.

## Knobs

All set as env vars in `k8s/eval-job.yaml`, all also accepted as CLI flags:

| var | default | note |
|---|---|---|
| `EVAL_MODEL` | `/data/runs/qwen3-8b-sft` | path on the PVC or an HF repo id |
| `EVAL_LIMIT` | `0` | set to e.g. `20` for a smoke test |
| `EVAL_BATCH_SIZE` | `16` | lower it if generation OOMs |
| `EVAL_MAX_NEW_TOKENS` | `2048` | `summary.json` reports how many hit this ceiling |
| `EVAL_TEMPERATURE` | `0` | greedy, so the run is reproducible |

## Resources

One A100, 32Gi of host memory, 4 cpu. Inference needs no optimizer state and no CPU
offload, so this is far lighter than the training Job and schedules much more easily --
any A100 will do, including the 40GB ones, since 16.4G of bf16 weights plus KV cache
fits comfortably.
