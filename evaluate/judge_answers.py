#!/usr/bin/env python3
"""Re-score an eval run with an LLM judge over an OpenAI-compatible API.

The string scorer in eval_math500.py compares normalised LaTeX and falls back to a
numeric check. It cannot see that (x+1)^2 and x^2+2x+1 are the same answer, so it
undercounts. This pass sends those cases to a judge model and reports both numbers,
which keeps the cheap deterministic score auditable while correcting its blind spot.

Runs against predictions.jsonl produced by an earlier eval, so it needs no GPU and can
be re-run without regenerating anything. Only the string scorer's rejects are judged by
default: a normalised exact match is not something a judge is going to overturn, and
judging 500 rows when 100 are in question wastes five times the calls.

Reads OPENAI_BASE_URL and OPENAI_API_KEY from the environment, falling back to the
repo's .env; on NRP those point at the managed LLM endpoint. Only those keys are taken
from the file, so the container-only paths that live alongside them stay out of the way.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# Only these are read out of .env. Sourcing the whole file into a local shell also
# imports container paths like HF_HOME=/data/hf, which then makes anything that
# touches the HF cache fail with "Read-only file system: '/data'".
ENV_KEYS = ("OPENAI_BASE_URL", "OPENAI_API_KEY", "EVAL_JUDGE_MODEL", "EVAL_DATASET", "EVAL_SPLIT")


def load_env_file(path: Path) -> None:
    """Fill in the keys above from .env, without overriding what is already exported."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in ENV_KEYS and not os.environ.get(key):
            os.environ[key] = value.strip().strip("'\"")


def env(key: str, default):
    raw = os.environ.get(key, "")
    if raw == "":
        return default
    return type(default)(raw) if default is not None else raw


def parse_args() -> argparse.Namespace:
    load_env_file(Path(__file__).resolve().parent.parent / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="eval output directory holding predictions.jsonl")
    p.add_argument("--dataset", default=env("EVAL_DATASET", "HuggingFaceH4/MATH-500"),
                   help="source of the problem statements, matched back by unique_id")
    p.add_argument("--split", default=env("EVAL_SPLIT", "test"))
    p.add_argument("--model", default=env("EVAL_JUDGE_MODEL", "qwen3-small"),
                   help="model id as the gateway knows it, not the upstream name: on NRP "
                        "'qwen3' routes to Qwen/Qwen3.8-Flash-Next-FP8. List what is available "
                        "with: curl -H \"Authorization: Bearer $OPENAI_API_KEY\" $OPENAI_BASE_URL/models")
    p.add_argument("--base-url", default=env("OPENAI_BASE_URL", ""))
    p.add_argument("--mode", default="disagree-only", choices=["disagree-only", "all"],
                   help="disagree-only judges just the rows the string scorer rejected")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=800,
                   help="reasoning judges spend most of this thinking before answering")
    p.add_argument("--retries", type=int, default=3)
    return p.parse_args()


SYSTEM = (
    "You grade answers to competition maths problems. Decide whether the predicted answer "
    "is mathematically equivalent to the reference answer for the given problem. Equivalent "
    "means the same value or expression, ignoring formatting, ordering of equivalent forms, "
    "and algebraic rearrangement. A missing or empty prediction is INCORRECT. "
    "Answer with exactly one word: CORRECT or INCORRECT."
)

PROMPT = """Problem:
{problem}

Reference answer: {gold}
Predicted answer: {pred}

Is the predicted answer equivalent to the reference answer?"""


def call_judge(args, api_key: str, problem: str, gold: str, pred: str | None) -> tuple[str, str]:
    """Return (verdict, raw), where verdict is CORRECT / INCORRECT / UNKNOWN."""
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT.format(
                problem=problem.strip()[:4000],
                gold=gold,
                pred=pred if pred is not None else "(no answer produced)",
            )},
        ],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    last = ""
    for attempt in range(args.retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as fh:
                resp = json.load(fh)
            msg = resp["choices"][0]["message"]
            content = msg.get("content")
            if not content:
                # A reasoning model that ran out of budget leaves content empty and its
                # partial thought in `reasoning`; treat that as unusable rather than
                # guessing from half a thought.
                last = "empty content (raise --max-tokens)"
                continue
            text = content.strip().upper()
            if "INCORRECT" in text:
                return "INCORRECT", content.strip()
            if "CORRECT" in text:
                return "CORRECT", content.strip()
            last = content.strip()
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, TimeoutError) as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    return "UNKNOWN", last


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.base_url or not api_key:
        sys.exit("OPENAI_BASE_URL / OPENAI_API_KEY not found in the environment or ../.env")

    run_dir = Path(args.run_dir)
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.is_file():
        sys.exit(f"no predictions.jsonl under {run_dir}")
    records = [json.loads(l) for l in pred_path.open()]

    from datasets import load_dataset
    data = load_dataset(args.dataset, split=args.split)
    problems = {r["unique_id"]: r["problem"] for r in data}

    todo = [r for r in records if args.mode == "all" or not r["correct"]]
    print(f"[judge] model    : {args.model} @ {args.base_url}")
    print(f"[judge] records  : {len(records)}  to judge: {len(todo)} ({args.mode})", flush=True)

    started = time.time()
    done = [0]

    def work(rec):
        verdict, raw = call_judge(args, api_key, problems.get(rec["unique_id"], ""), rec["gold"], rec["pred"])
        rec["judge"] = verdict
        rec["judge_raw"] = raw[:200]
        done[0] += 1
        if done[0] % 25 == 0:
            print(f"[judge] {done[0]}/{len(todo)}  {(time.time() - started) / 60:.1f} min", flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))

    for r in records:
        if "judge" not in r:                      # string scorer accepted it, not re-judged
            r["judge"] = "CORRECT" if r["correct"] else "UNKNOWN"
    judged_correct = sum(r["judge"] == "CORRECT" for r in records)
    unknown = sum(r["judge"] == "UNKNOWN" for r in records)
    flipped = [r for r in records if not r["correct"] and r["judge"] == "CORRECT"]

    with (run_dir / "predictions_judged.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    summary.update({
        "judge_model": args.model,
        "judge_mode": args.mode,
        "accuracy_string": sum(r["correct"] for r in records) / len(records),
        "accuracy_judge": judged_correct / len(records),
        "judge_flipped_to_correct": len(flipped),
        "judge_unknown": unknown,
    })
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n[judge] ==== summary ====")
    print(f"  accuracy (string scorer) {summary['accuracy_string']:.3f}")
    print(f"  accuracy (llm judge)     {summary['accuracy_judge']:.3f}")
    print(f"  flipped wrong -> right   {len(flipped)}")
    print(f"  unresolved (UNKNOWN)     {unknown}")
    for r in flipped[:8]:
        print(f"    {r['unique_id']}: gold={r['gold']!r} pred={r['pred']!r}")
    print(f"[judge] wrote {run_dir}/predictions_judged.jsonl and updated summary.json")


if __name__ == "__main__":
    main()
