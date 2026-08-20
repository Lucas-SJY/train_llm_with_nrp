#!/usr/bin/env python3
"""Minimal SFT run: HF Trainer + DeepSpeed, loss on the assistant turn only.

Single node, single GPU. Launched through src/entrypoint.sh, which calls
`deepspeed --num_gpus=1` so DeepSpeed gets the distributed group it expects.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def env(key: str, default):
    """Read a default from the environment (.env / k8s Secret), keeping the type."""
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes")
    return type(default)(raw) if default is not None else raw


def parse_args() -> argparse.Namespace:
    # Every default comes from an env var, so the whole run is configured through
    # .env locally and through the sft-env Secret on the cluster. CLI flags still win.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=env("MODEL_NAME", "Qwen/Qwen3-1.7B"))
    p.add_argument("--train-file", default=env("TRAIN_FILE", "/workspace/data/train.jsonl"))
    p.add_argument("--eval-file", default=env("EVAL_FILE", ""), help="leave empty to skip evaluation")
    p.add_argument("--output-dir", default=env("OUTPUT_DIR", "/data/runs/qwen-sft"))
    p.add_argument("--deepspeed", default=env("DEEPSPEED_CONFIG", "/workspace/configs/ds.json"))
    p.add_argument("--max-seq-len", type=int, default=env("MAX_SEQ_LEN", 4096))
    p.add_argument("--epochs", type=float, default=env("EPOCHS", 2.0))
    p.add_argument("--lr", type=float, default=env("LEARNING_RATE", 1e-5))
    p.add_argument("--micro-batch-size", type=int, default=env("MICRO_BATCH_SIZE", 1), help="per-GPU batch size")
    p.add_argument("--grad-accum", type=int, default=env("GRAD_ACCUM", 32))
    p.add_argument("--warmup-ratio", type=float, default=env("WARMUP_RATIO", 0.03),
                   help="fraction of total steps spent warming up, e.g. 0.03")
    p.add_argument("--logging-steps", type=int, default=env("LOGGING_STEPS", 10))
    p.add_argument("--save-steps", type=int, default=env("SAVE_STEPS", 500))
    p.add_argument("--save-total-limit", type=int, default=env("SAVE_TOTAL_LIMIT", 2))
    p.add_argument("--num-workers", type=int, default=env("NUM_WORKERS", 4))
    p.add_argument("--seed", type=int, default=env("SEED", 42))
    p.add_argument("--report-to", default=env("REPORT_TO", ""), help='e.g. "wandb", empty disables logging')
    p.add_argument("--resume", default=env("RESUME", "auto"), choices=["auto", "no"],
                   help="auto: continue from a checkpoint in --output-dir if one exists; no: ignore it")
    # The deepspeed launcher appends this to the script's argv; accept it silently.
    p.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def build_encoder(tokenizer, max_seq_len: int):
    """Encode one messages record into input_ids/labels, masking the prompt with -100."""

    def encode(example):
        messages = example["messages"]
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        answer = messages[-1]["content"]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # ignored by templates that do not know this flag
        )
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]

        input_ids = (prompt_ids + answer_ids)[:max_seq_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_seq_len]
        return {"input_ids": input_ids, "labels": labels}

    return encode


class Collator:
    """Right-side padding; labels are padded with -100."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attention_mask = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attention_mask.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file
    raw = load_dataset("json", data_files=data_files)

    encode = build_encoder(tokenizer, args.max_seq_len)
    tokenized = raw.map(encode, remove_columns=raw["train"].column_names, num_proc=args.num_workers)
    # Records that are all -100 (prompt alone fills max_seq_len) produce no gradient.
    tokenized = tokenized.filter(lambda ex: any(x != -100 for x in ex["labels"]))
    if len(tokenized["train"]) == 0:
        raise SystemExit(
            f"every training example was filtered out: with --max-seq-len {args.max_seq_len} "
            "the prompt alone fills the window, leaving nothing to compute loss on. "
            "Raise --max-seq-len or shorten the prompts."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # transformers v5 dropped warmup_ratio: warmup_steps now takes either an
        # int (exact steps) or a float in [0, 1) meaning a fraction of total steps.
        warmup_steps=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch" if args.eval_file else "no",
        per_device_eval_batch_size=args.micro_batch_size,
        deepspeed=args.deepspeed,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
        report_to=[x for x in args.report_to.split(",") if x],
        seed=args.seed,
        ddp_timeout=3600,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=Collator(tokenizer.pad_token_id),
    )

    # Resuming matters on NRP, where a pod can be evicted mid-run, but it must be
    # loud: a checkpoint whose global_step already equals max_steps makes train()
    # return instantly with train_loss 0, which reads like a broken run.
    ckpts = sorted(
        (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")),
        key=lambda d: int(d.split("-")[-1]),
    ) if os.path.isdir(args.output_dir) else []
    if args.resume == "no":
        resume = False
        if ckpts:
            print(f"[resume] ignoring {len(ckpts)} checkpoint(s) in {args.output_dir}, training from scratch")
    else:
        resume = bool(ckpts)
        if resume:
            print(f"[resume] continuing from {ckpts[-1]} (set RESUME=no to start over)")
        else:
            print(f"[resume] no checkpoint in {args.output_dir}, training from scratch")

    result = trainer.train(resume_from_checkpoint=resume)
    if result.global_step and result.metrics.get("train_runtime", 0) < 1:
        print(f"[resume] nothing left to do: the checkpoint had already finished all "
              f"{result.global_step} steps. Point OUTPUT_DIR somewhere new or set "
              f"RESUME=no to train again.")

    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
    print(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
