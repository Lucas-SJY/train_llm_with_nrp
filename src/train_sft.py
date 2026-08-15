#!/usr/bin/env python3
"""Minimal SFT run: HF Trainer + DeepSpeed ZeRO-3, loss on the assistant turn only.

Launched by torchrun from src/entrypoint.sh -- do not run this directly for
multi-GPU training.
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B"))
    p.add_argument("--train-file", default="/workspace/data/train.jsonl")
    p.add_argument("--eval-file", default="", help="leave empty to skip evaluation")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/data/runs/qwen-sft"))
    p.add_argument("--deepspeed", default="/workspace/configs/ds_zero3.json")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--micro-batch-size", type=int, default=1, help="per-GPU batch size")
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
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
        warmup_ratio=args.warmup_ratio,
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
        report_to=[],
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

    resume = any(
        d.startswith("checkpoint-") for d in os.listdir(args.output_dir)
    ) if os.path.isdir(args.output_dir) else False
    trainer.train(resume_from_checkpoint=resume)

    # Under ZeRO-3 the sharded weights are consolidated thanks to
    # stage3_gather_16bit_weights_on_model_save in ds_zero3.json.
    trainer.save_model(args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
    print(f"[done] model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
