# MATH-500 evaluation

Scores a fine-tuned checkpoint on [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500)
and reports the numbers the context-compression experiment turns on:

| metric | question it answers |
|---|---|
| `accuracy_string` / `accuracy_judge` | did the model still get the answer right |
| `thinking_tokens_mean` | how much reasoning it needed to get there |
| `answer_tokens_mean` | how much it spent stating the result |
| `never_closed_think` | how often it ran out of budget mid-thought |

Self-contained: nothing outside this directory is modified. The training image is
reused as-is and the eval script is mounted into it from a ConfigMap, so there is no
image rebuild and no change to `requirements.txt`, `Dockerfile`, `src/` or `run.sh`.

## Layout

```
evaluate/
├── eval_math500.py       # generation + string scoring, runs on a GPU in the pod
├── judge_answers.py      # LLM re-scoring over an OpenAI-compatible API, no GPU
├── k8s/eval-job.yaml     # the Job, mounts the script from a ConfigMap
├── run_eval.sh           # code -> job -> logs, same shape as ../run.sh
└── README.md
```

## The two stages

Generation needs a GPU and takes hours; judging needs a network call and takes minutes.
They are separate on purpose, so a judge can be re-run, swapped, or pointed at an older
result without regenerating anything.

```bash
# 1. generate + score on the cluster
./evaluate/run_eval.sh
./evaluate/run_eval.sh fetch          # pull results to ../eval-results

# 2. re-score the near-misses with an LLM judge, locally
python evaluate/judge_answers.py --run-dir eval-results/qwen3-8b-sft-v3
```

Individual steps of stage 1:

```bash
./evaluate/run_eval.sh code      # re-upload the script after editing it
./evaluate/run_eval.sh submit    # (re)submit the Job
./evaluate/run_eval.sh logs      # follow
./evaluate/run_eval.sh status    # job, pod, events
./evaluate/run_eval.sh fetch     # copy results to ../eval-results
./evaluate/run_eval.sh clean     # delete the Job
```

Editing `eval_math500.py` only needs `run_eval.sh code && run_eval.sh submit` -- the
ConfigMap is the deploy mechanism, so iteration costs seconds instead of a rebuild.

## Stage 1: generation and string scoring

1. Loads the model from the PVC (no download).
2. Builds each prompt with the **same instruction the model was trained on** --
   `Return your final response within \boxed{}.` followed by the problem -- rendered
   through `apply_chat_template` with `add_generation_prompt=True`. The rendering is
   byte-identical to the stock Qwen3 template.
3. Generates greedily (`temperature 0`) in batches, with left padding, which
   decoder-only batched generation requires.
4. Splits the completion at `</think>` and scores **only what follows**.

### Thinking mode has to match the checkpoint

`EVAL_ENABLE_THINKING` decides how the generation prompt ends:

| value | prompt ends with | use for |
|---|---|---|
| `true` (default) | `<\|im_start\|>assistant\n` | v3 and anything trained on thinking turns; the model opens its own `<think>` |
| `false` | `...assistant\n<think>\n\n</think>\n\n` | v1 and v2, trained with an empty think block prefilled |

Getting this backwards does not error, it just measures a prompt the model never saw.
The run prints `[eval] thinking : True|False` at startup so it is visible in the log.

### Only the stated answer counts

`split_answer()` cuts the completion at `</think>` and the scorer looks at the second
half alone. The thought routinely contains a `\boxed{}` of its own -- the trace's
working -- and crediting that would score a run that never actually committed:

| completion | scored as |
|---|---|
| thought has `\boxed{7}`, answer has `\boxed{10}` | `10` |
| thought has `\boxed{10}`, answer has `\boxed{7}` | `7` (wrong) |
| `</think>` never emitted (truncated) | no answer |
| answer section has no `\boxed{}` | no answer |

The last two land in `never_closed_think` / `no_boxed_answer`, which keeps "ran out of
tokens" distinguishable from "got it wrong". The LLM judge sees the same extracted
answer, never the reasoning, so both scorers grade the same thing.

### The string scorer

`sympy.parsing.latex` needs antlr, which the training image does not have, so answers
are compared by LaTeX normalisation plus a numeric fallback: `\dfrac`→`\frac`,
`\left`/`\right` and spacing stripped, `\text{}` unwrapped, `^\circ` and `%` dropped,
thousands separators removed, then exact match, then float comparison of fractions.

Validated two ways before first use: extracting `\boxed{}` from MATH-500's own
`solution` field and scoring it against the `answer` field matches **500/500**, and
hand-written edge cases (`0.5` vs `\frac{1}{2}`, `90^\circ` vs `90`, multiple boxes,
missing box, thought-vs-answer precedence) all behave as intended.

## Stage 2: the LLM judge

The string scorer cannot see that `(x+1)^2` and `x^2+2x+1` are the same answer, so it
undercounts. `judge_answers.py` sends those cases to a model over an OpenAI-compatible
API and reports both numbers, keeping the cheap deterministic score auditable while
correcting its blind spot.

```bash
python evaluate/judge_answers.py --run-dir eval-results/qwen3-8b-sft-v3
python evaluate/judge_answers.py --run-dir ... --model gpt-oss --mode all
```

`OPENAI_BASE_URL` and `OPENAI_API_KEY` come from the environment, falling back to the
repo's `.env`. Only the keys it needs are read from that file -- sourcing the whole
thing into a local shell also imports container paths like `HF_HOME=/data/hf`, which
then makes anything touching the HF cache fail with `Read-only file system: '/data'`.

By default only the rows the string scorer rejected are judged: a normalised exact
match is not something a judge overturns, and judging 500 rows when 100 are in question
wastes five times the calls. `--mode all` judges everything.

Model ids are whatever the gateway exposes, not upstream names -- on NRP `qwen3-small`
and `qwen3` both route to Qwen3.8 builds. List them with:

```bash
curl -s -H "Authorization: Bearer $OPENAI_API_KEY" $OPENAI_BASE_URL/models
```

Measured single-call latency on the NRP gateway: `gemma4-12b` 0.4s, `deepseek-v4-flash`
0.8s, `qwen3` 1.7s, `gpt-oss` 2.2s. With 8 workers even a full 500-row pass is minutes.

Writes `predictions_judged.jsonl` and adds `accuracy_string`, `accuracy_judge`,
`judge_flipped_to_correct` and `judge_unknown` to `summary.json`.

## Output

Written to `/data/eval/<run-name>/` on the PVC, where `<run-name>` defaults to the
model directory name:

- `summary.json` -- accuracy, token statistics, breakdown by level and subject
- `predictions.jsonl` -- one row per problem with the full completion, for eyeballing
- `predictions_judged.jsonl` -- the above plus a judge verdict per row

## Comparing checkpoints

The point is the comparison, and it is a matter of overriding env vars in
`k8s/eval-job.yaml`:

| `EVAL_MODEL` | `EVAL_ENABLE_THINKING` | what it measures |
|---|---|---|
| `Qwen/Qwen3-8B` | `true` | the base model, untrained baseline |
| `/data/runs/qwen3-8b-sft` | `false` | v1, compressed CoT |
| `/data/runs/qwen3-8b-sft-v2` | `false` | v2, labelled CoT, instruction in the prompt |
| `/data/runs/qwen3-8b-sft-v3` | `true` | v3, labelled CoT inside `<think>` |

Each run writes to its own subdirectory, so results accumulate rather than overwrite.

Compare accuracy against `thinking_tokens_mean`, not `completion_tokens_mean`: v3 also
emits a written-out solution after the thought, so its total is larger for reasons that
have nothing to do with how much it reasoned.

## Knobs

All set as env vars in `k8s/eval-job.yaml`, all also accepted as CLI flags:

| var | default | note |
|---|---|---|
| `EVAL_MODEL` | `/data/runs/qwen3-8b-sft-v3` | path on the PVC or an HF repo id |
| `EVAL_ENABLE_THINKING` | `true` | must match how the checkpoint was trained |
| `EVAL_LIMIT` | `0` | set to e.g. `20` for a smoke test |
| `EVAL_BATCH_SIZE` | `6` | KV cache is 144 KB/token; 12 OOMed 13 hours into a run |
| `EVAL_MAX_NEW_TOKENS` | `12288` | matches the training window; a thinking model needs room for the trace *and* the answer |
| `EVAL_TEMPERATURE` | `0` | greedy, so the run is reproducible |
| `EVAL_JUDGE_MODEL` | `qwen3-small` | stage 2 only |

Run a smoke test first: `EVAL_LIMIT=20`, then check that `never_closed_think` and
`no_boxed_answer` are both near zero before committing to all 500. If they are not,
`EVAL_MAX_NEW_TOKENS` is too small and the full run would measure truncation rather
than the model.

## Resources

One RTX A6000, 32Gi of host memory, 4 cpu. Inference needs no optimizer state and no
CPU offload, so this is far lighter than the training Job. It deliberately does not ask
for an A100: at batch 6 the KV cache is 10.6G on top of 16.4G of weights, which a 47G
A6000 holds with room to spare, and the A6000 pool schedules in seconds where the A100
pool kept training waiting 14 hours. Any Ampere-or-newer card works; the 56 V100s in the
cluster do not, because Volta predates bf16. Stage 2 needs no cluster resources at all.
