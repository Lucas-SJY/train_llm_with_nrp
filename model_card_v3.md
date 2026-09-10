---
base_model: Qwen/Qwen3-8B
library_name: transformers
license: apache-2.0
language:
  - en
pipeline_tag: text-generation
tags:
  - qwen3
  - sft
  - reasoning
  - chain-of-thought
  - structured-reasoning
  - math
---

# qwen3-8b-sft-bespoke — revision `v3`

Qwen3-8B fine-tuned to **label every step of its own reasoning**. The thought is emitted
inside a `<think>` block where each step opens with one of eight tags, and the block is
followed by an ordinary written solution ending in `\boxed{}`.

```
<think>
[restating_problem] I need to find how many positive divisors 196 has.

[reflecting] Hmm, divisors.

[recalling_knowledge] The number of divisors follows from the prime factorisation ...

[logical_deduction] 196 = 2^2 · 7^2, so the count is (2+1)(2+1) = 9.
</think>

The prime factorisation of 196 is $2^2 \cdot 7^2$ ...
Thus, the number of positive whole-number divisors of 196 is $\boxed{9}$.
```

This is a research checkpoint from a study on structured and compressed reasoning, not a
general-purpose assistant. **It is a different experiment from the `main` branch**, which
holds a compressed-CoT model trained with four span types deleted and no thinking block.

## Loading it

`v3` is not the default branch — you have to ask for it:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Lucas-SJY/qwen3-8b-sft-bespoke"
tok = AutoTokenizer.from_pretrained(repo, revision="v3")
model = AutoModelForCausalLM.from_pretrained(repo, revision="v3", dtype="bfloat16")
```

## Prompt format

Two things matter, and getting either wrong measures a model that was never trained.

**Leave `enable_thinking` at its default.** The target opens its own `<think>` block, so
the prompt must stop at `<|im_start|>assistant`. Passing `enable_thinking=False` makes
the template prefill an empty `<think></think>`, and the sequence ends up with two
thinking blocks.

**Do not add a label instruction.** The training prompt is the bare question with the
standard boxed-answer sentence and nothing else. The model emits the tags because it
learned to, not because it was asked to.

```python
problem = "Find $k$, if ${(3^k)}^6=3^6$."
messages = [{
    "role": "user",
    "content": "Return your final response within \\boxed{}. " + problem,
}]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=12288)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Budget generously. The model writes 3,397 thinking tokens on average and 12.6 % of
MATH-500 problems did not terminate inside 12,288 — a small `max_new_tokens` will cut
most answers off mid-thought.

Score only what follows `</think>`. The thought routinely contains a `\boxed{}` of its
own, which is working, not an answer.

## The eight labels

`planning_next_step` · `restating_problem` · `recalling_knowledge` · `logical_deduction`
· `reflecting` · `verifying` · `correcting_itself` · `concluding`

Every one of 500 evaluation outputs used the tags, averaging 14.7 per problem, with no
invented tags. The mix drifts from the training distribution though:
`logical_deduction` falls from 29.7 % of training tags to 18.6 % of generated ones while
`reflecting` and `restating_problem` rise, and `correcting_itself` all but disappears
(2.0 % → 0.1 %, present in 0.8 % of outputs).

Whether a completion contains `[concluding]` is a usable confidence signal: those score
0.928, the rest 0.713.

## Training data

An annotated corpus of DeepSeek-R1 reasoning traces (`bespoke-v2`), derived from
[bespokelabs/Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k).
Every reasoning span carries one of the eight labels. The prepared set is published as
[Lucas-SJY/bespoke_for_sft](https://huggingface.co/datasets/Lucas-SJY/bespoke_for_sft).

| | |
|---|---|
| train / validation | 5,042 / 102 |
| labels kept | all 8, nothing dropped |
| target | `<think>` labelled trace `</think>` + reference solution |
| prompt | the bare question |
| targets truncated at 12,288 | 6.5 % |

The text after `</think>` is the dataset's own reference solution rather than the tail of
the trace. This was deliberate: a third of the raw traces never write a `\boxed{}` at
all, and training on them teaches the model not to commit to an answer.

Loss is computed on the assistant turn only; prompt tokens are masked with `-100`.

## Training procedure

Full-parameter SFT on a single NVIDIA A100 80GB, using DeepSpeed ZeRO-2 with the
optimizer state offloaded to host memory — an 8B model needs roughly 131GB for weights,
gradients and fp32 Adam state, which does not fit in 80GB of VRAM without offload.

| | |
|---|---|
| epochs | 1 (79 optimizer steps) |
| global batch size | 64 (micro-batch 1 × grad accum 64) |
| max sequence length | 12,288 |
| optimizer | AdamW (DeepSpeedCPUAdam), β=(0.9, 0.999), ε=1e-8 |
| learning rate | 1e-5, cosine decay, 3 % warmup |
| weight decay | 0.0 |
| gradient clipping | 1.0 |
| precision | bf16 mixed precision, fp32 optimizer state |
| gradient checkpointing | enabled |
| seed | 42 |
| wall clock | 3.4 hours |
| peak VRAM | 62.9 GB of 80 GB |

| metric | value |
|---|---|
| final train loss | 0.3713 |
| validation loss | 0.3251 |

Loss is not comparable with the `main` branch: the targets there are compressed traces,
these are full labelled traces plus a written solution, and the label tags themselves are
highly predictable tokens that lower perplexity on their own.

## Evaluation

MATH-500, all 500 test problems, greedy decoding, 12,288-token budget, scored on the
stated answer only.

| metric | value |
|---|---|
| accuracy, string scorer | 0.814 |
| accuracy, LLM judge | 0.846 |
| accuracy among problems it finished | 0.968 (423 / 437) |
| never closed `</think>` | 63 (12.6 %) |
| mean thinking tokens | 3,397 |
| mean answer tokens | 420 |

Two graders were used and they agree on 96.8 % of problems; the 16 disagreements are all
the string scorer being too strict about LaTeX notation, and were verified by hand. Read
0.814 as the auditable floor and 0.846 as the better estimate.

| difficulty | accuracy | never finished |
|---|---|---|
| Level 1 | 0.953 | 2.3 % |
| Level 2 | 0.967 | 1.1 % |
| Level 3 | 0.895 | 10.5 % |
| Level 4 | 0.852 | 10.9 % |
| Level 5 | 0.687 | 26.9 % |

| subject | accuracy | never finished |
|---|---|---|
| Algebra | 0.960 | 4.0 % |
| Number Theory | 0.887 | 11.3 % |
| Prealgebra | 0.866 | 8.5 % |
| Precalculus | 0.857 | 10.7 % |
| Counting & Probability | 0.789 | 15.8 % |
| Intermediate Algebra | 0.742 | 21.6 % |
| Geometry | 0.683 | 26.8 % |

## Limitations

**It does not always stop.** 12.6 % of MATH-500 problems ran to the token ceiling without
closing the thought, rising to 26.9 % at Level 5. A quarter of those end in visibly
repeated text; the rest are still opening new case analyses at the cutoff. Longer
thinking correlates with being wrong, not right — the median correct trace is 36 %
shorter than the median incorrect one — so a larger budget is not obviously the fix. The
likely cause is upstream: 6.5 % of the training targets were themselves truncated, so the
model was shown examples with no ending and no EOS.

**No untrained baseline.** Base Qwen3-8B has not been measured under the same harness, so
nothing here separates what the fine-tune added from what the base model already did.

**Contamination unchecked.** The training traces derive from the Bespoke-Stratos lineage,
which has been reported to overlap MATH-500. This has not been verified either way.

**Single seed, greedy decoding.** Reproducible, but no variance estimate. At n=500 the
standard error is about 1.7 points, so differences under roughly 4 points are not
evidence.

**English mathematics only.** No evaluation outside MATH-500, and none on code, general
reasoning or dialogue.
