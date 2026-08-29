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
  - context-compression
  - math
---

# qwen3-8b-sft-bespoke

Qwen3-8B fine-tuned to produce **shorter chain-of-thought**. The training targets are
reasoning traces with their redundant passages removed, so the model is taught to reach
the same answer while skipping the self-talk that long-form reasoning models tend to
emit — restating the problem, announcing what it is about to do, second-guessing itself,
and re-deriving a result it has already computed.

This is a research checkpoint for a context-compression study, not a general-purpose
assistant.

## Prompt format

The model was trained on a single fixed instruction. **Using a different one measures a
distribution the model never saw**, so keep this exact prefix:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Lucas-SJY/qwen3-8b-sft-bespoke"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map="auto")

problem = "What is the average of the last three numbers if ..."
messages = [{
    "role": "user",
    "content": "Return your final response within \\boxed{}. " + problem,
}]

text = tok.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tok(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=2048)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The final answer is emitted inside `\boxed{...}`. Training used the non-thinking chat
template (`enable_thinking=False`); the compressed reasoning is produced as ordinary
assistant content rather than inside a thinking block.

## Training data

An annotated corpus of DeepSeek-R1 reasoning traces (internally called `bespoke-v2`),
where every trace is segmented into spans and each span carries one of eight labels:
`planning_next_step`, `restating_problem`, `logical_deduction`, `reflecting`,
`verifying`, `correcting_itself`, `recalling_knowledge`, `concluding`.

The compression target is built by dropping four span types and concatenating the rest:

| dropped | kept |
|---|---|
| `planning_next_step`, `restating_problem`, `reflecting`, `verifying` | `logical_deduction`, `correcting_itself`, `recalling_knowledge`, `concluding` |

That keeps **56.0% of the original reasoning tokens** (4,712,631 of 8,412,626 across
5,144 traces). The final span of every trace is always retained so the boxed answer
survives.

| split | examples |
|---|---|
| train | 5,042 |
| validation | 102 |

Loss is computed on the assistant turn only; prompt tokens are masked with `-100`.

## Training procedure

Full-parameter SFT on a single NVIDIA A100 80GB, using DeepSpeed ZeRO-2 with the
optimizer state offloaded to host memory — an 8B model needs roughly 131GB for weights,
gradients and fp32 Adam state, which does not fit in 80GB of VRAM without offload.

| | |
|---|---|
| epochs | 1 (79 optimizer steps) |
| global batch size | 64 (micro-batch 1 × grad accum 64) |
| max sequence length | 4096 |
| optimizer | AdamW (DeepSpeedCPUAdam), β=(0.9, 0.999), ε=1e-8 |
| learning rate | 1e-5, cosine decay, 3% warmup |
| weight decay | 0.0 |
| gradient clipping | 1.0 |
| precision | bf16 mixed precision, fp32 optimizer state |
| gradient checkpointing | enabled |
| wall clock | 3.6 hours |

### Results

| metric | value |
|---|---|
| final train loss | 0.4514 |
| validation loss | 0.4160 |

Loss fell from 0.5565 at the first logged step to 0.4514, with gradient norms decaying
from 0.96 to 0.41.

**Downstream benchmark numbers are not included yet.** The intended evaluation measures
accuracy and generated-token count together on MATH-500 — the claim this checkpoint is
meant to test is that accuracy holds while token count drops — and that comparison
against the base model has not been run at the time of writing. Treat the loss values
above as training diagnostics, not as evidence of the compression hypothesis.

## Limitations

- **Narrow domain.** The training traces are mathematical problem solving. Behaviour on
  other reasoning domains is untested.
- **One epoch, 79 optimizer steps.** This is a light fine-tune on ~5k examples; it
  adjusts style far more than capability.
- **Compression is imitated, not verified.** The model learned to reproduce traces with
  reflection and verification passages removed. Removing the text that expresses
  self-checking is not the same as preserving the underlying self-checking, and the
  effect on accuracy — particularly on harder problems where verification catches real
  errors — is exactly what remains to be measured.
- **Fixed prompt.** Quality degrades with instructions other than the one above.
- Inherits the base model's licence (Apache-2.0). Confirm the terms of the training
  data before redistributing derivatives.

## Citation

Base model:

```bibtex
@misc{qwen3,
  title  = {Qwen3 Technical Report},
  author = {Qwen Team},
  year   = {2025}
}
```
