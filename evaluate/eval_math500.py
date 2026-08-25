#!/usr/bin/env python3
"""Evaluate a fine-tuned model on MATH-500.

Reports two numbers side by side, which together are the point of the context
compression experiment:

    accuracy            -- did the model still get the answer right
    completion tokens   -- how much reasoning it needed to get there

Everything is read-only: the model comes from the PVC, results are written to a new
directory under it. Nothing in the training pipeline is touched.

The scorer is dependency-free on purpose. sympy.parsing.latex needs antlr, which the
training image does not have, so answers are compared with LaTeX normalisation plus a
numeric fallback instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from fractions import Fraction
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The training prompts all started with this sentence, so evaluation has to use it too;
# a different instruction would measure a different distribution than the one trained on.
INSTRUCTION = "Return your final response within \\boxed{}."


def env(key: str, default):
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    return type(default)(raw) if default is not None else raw


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=env("EVAL_MODEL", "/data/runs/qwen3-8b-sft"),
                   help="local path or HF repo id; point it at the base model to get a baseline")
    p.add_argument("--dataset", default=env("EVAL_DATASET", "HuggingFaceH4/MATH-500"))
    p.add_argument("--split", default=env("EVAL_SPLIT", "test"))
    p.add_argument("--output-dir", default=env("EVAL_OUTPUT_DIR", "/data/eval"))
    p.add_argument("--run-name", default=env("EVAL_RUN_NAME", ""), help="defaults to the model directory name")
    p.add_argument("--limit", type=int, default=env("EVAL_LIMIT", 0), help="only the first N problems, 0 means all")
    p.add_argument("--batch-size", type=int, default=env("EVAL_BATCH_SIZE", 16))
    p.add_argument("--max-new-tokens", type=int, default=env("EVAL_MAX_NEW_TOKENS", 2048))
    p.add_argument("--temperature", type=float, default=env("EVAL_TEMPERATURE", 0.0),
                   help="0 means greedy, which keeps the run reproducible")
    p.add_argument("--seed", type=int, default=env("SEED", 42))
    return p.parse_args()


# ---------------------------------------------------------------------------
# answer extraction and comparison
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> str | None:
    """Return the content of the last \\boxed{...}, matching braces properly."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        # \boxed 5 style, take the rest of the token
        rest = text[idx + len("\\boxed"):].strip()
        return rest.split()[0] if rest else None
    depth, out = 0, []
    for ch in text[i:]:
        if ch == "{":
            depth += 1
            if depth == 1:
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(ch)
    return None


_STRIP = [
    (r"\\left", ""), (r"\\right", ""), (r"\\!", ""), (r"\\,", ""), (r"\\;", ""), (r"\\ ", " "),
    (r"\\dfrac", r"\\frac"), (r"\\tfrac", r"\\frac"), (r"\\cdot", "*"), (r"\\times", "*"),
    (r"\^\{\\circ\}", ""), (r"\^\\circ", ""), (r"\\%", ""), (r"%", ""),
    (r"\\\$", ""), (r"\$", ""), (r"\\text\{([^}]*)\}", r"\1"), (r"\\mbox\{([^}]*)\}", r"\1"),
]


def normalize(ans: str) -> str:
    if ans is None:
        return ""
    s = ans.strip()
    for pat, rep in _STRIP:
        s = re.sub(pat, rep, s)
    s = s.replace(" ", "").replace("\n", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\frac(\d)(\d)", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.rstrip(".")
    s = re.sub(r"^\{(.*)\}$", r"\1", s)          # a stray outer brace
    s = re.sub(r"(\d),(\d\d\d)", r"\1\2", s)     # thousands separators
    return s.lower()


def to_number(s: str):
    """Best-effort numeric value of a normalised answer, or None."""
    try:
        return float(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    m = re.fullmatch(r"\(?(-?[\d.]+)\)?/\(?(-?[\d.]+)\)?", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    pn, gn = to_number(p), to_number(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    run_name = args.run_name or Path(args.model.rstrip("/")).name
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] model      : {args.model}")
    print(f"[eval] dataset    : {args.dataset} [{args.split}]")
    print(f"[eval] output     : {out_dir}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only batched generation needs left padding

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa",
    )
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = True

    data = load_dataset(args.dataset, split=args.split)
    if args.limit:
        data = data.select(range(min(args.limit, len(data))))
    print(f"[eval] problems   : {len(data)}", flush=True)

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{INSTRUCTION} {row['problem']}"}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for row in data
    ]

    records, n_correct, started = [], 0, time.time()
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start:start + args.batch_size]
        enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = out[:, enc["input_ids"].shape[1]:]

        for j, seq in enumerate(generated):
            row = data[start + j]
            keep = seq[seq != tokenizer.pad_token_id]
            text = tokenizer.decode(keep, skip_special_tokens=True)
            pred = extract_boxed(text)
            ok = is_correct(pred, row["answer"])
            n_correct += ok
            records.append({
                "unique_id": row["unique_id"], "subject": row["subject"], "level": row["level"],
                "gold": row["answer"], "pred": pred, "correct": bool(ok),
                "completion_tokens": int(keep.numel()), "completion": text,
            })

        done = start + len(batch_prompts)
        print(f"[eval] {done}/{len(prompts)}  acc={n_correct / done:.3f}  "
              f"{(time.time() - started) / 60:.1f} min", flush=True)

    # -----------------------------------------------------------------------
    # summary
    # -----------------------------------------------------------------------
    toks = sorted(r["completion_tokens"] for r in records)
    by_level, by_subject = {}, {}
    for r in records:
        by_level.setdefault(r["level"], []).append(r["correct"])
        by_subject.setdefault(r["subject"], []).append(r["correct"])

    summary = {
        "model": args.model,
        "dataset": f"{args.dataset}[{args.split}]",
        "n": len(records),
        "accuracy": n_correct / len(records) if records else 0.0,
        "completion_tokens_mean": sum(toks) / len(toks) if toks else 0,
        "completion_tokens_median": toks[len(toks) // 2] if toks else 0,
        "completion_tokens_max": toks[-1] if toks else 0,
        "truncated": sum(r["completion_tokens"] >= args.max_new_tokens for r in records),
        "no_boxed_answer": sum(r["pred"] is None for r in records),
        "accuracy_by_level": {k: sum(v) / len(v) for k, v in sorted(by_level.items())},
        "accuracy_by_subject": {k: sum(v) / len(v) for k, v in sorted(by_subject.items())},
        "minutes": (time.time() - started) / 60,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    with (out_dir / "predictions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n[eval] ==== summary ====")
    for k in ("n", "accuracy", "completion_tokens_mean", "completion_tokens_median",
              "truncated", "no_boxed_answer", "minutes"):
        print(f"  {k:26s} {summary[k]}")
    print(f"  accuracy_by_level          {summary['accuracy_by_level']}")
    print(f"[eval] wrote {out_dir}/summary.json and predictions.jsonl")


if __name__ == "__main__":
    main()
