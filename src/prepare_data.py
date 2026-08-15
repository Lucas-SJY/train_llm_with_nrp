#!/usr/bin/env python3
"""Convert the bespoke-v2 span annotations into an SFT dataset (messages format).

Each raw sample is one reasoning trace cut into spans, and every span carries a
label (logical_deduction / reflecting / verifying / ...). The SFT objective built
here is:

    input  = question
    output = the reasoning trace with the dropped labels removed (compressed CoT)

--drop-labels controls which labels are removed; the default drops the four most
redundant ones. Pass --drop-labels "" (empty string) for the uncompressed
baseline that keeps the full trace.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_DROP = "planning_next_step,restating_problem,reflecting,verifying"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="bespoke-v2", help="directory holding sample_*.json")
    p.add_argument("--output-dir", default="data", help="where train.jsonl / val.jsonl are written")
    p.add_argument("--drop-labels", default=DEFAULT_DROP, help="comma-separated span labels to drop")
    p.add_argument("--val-ratio", type=float, default=0.02, help="fraction held out for validation")
    p.add_argument("--max-samples", type=int, default=0, help="only take the first N files, 0 means all")
    p.add_argument("--min-target-tokens", type=int, default=32, help="drop samples whose target is too short")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_example(sample: dict, drop: set[str]) -> tuple[dict, int, int] | None:
    """Return (messages example, original token count, kept token count), or None if unusable."""
    question = (sample.get("question") or "").strip()
    spans = sample.get("spans") or []
    if not question or not spans:
        return None

    kept, kept_tokens = [], 0
    for span in spans:
        text = (span.get("text") or "").strip()
        if not text or span.get("label") in drop:
            continue
        kept.append(text)
        kept_tokens += int(span.get("token_count") or 0)

    # Safety net: the final answer (usually the last span) must survive, otherwise
    # the sample carries no useful supervision signal.
    last = (spans[-1].get("text") or "").strip()
    if last and (not kept or kept[-1] != last):
        kept.append(last)
        kept_tokens += int(spans[-1].get("token_count") or 0)

    target = "\n\n".join(kept).strip()
    if not target:
        return None

    total_tokens = int((sample.get("summary") or {}).get("total_token_count") or 0)
    example = {
        "id": sample.get("id"),
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": target},
        ],
    }
    return example, total_tokens, kept_tokens


def main() -> None:
    args = parse_args()
    drop = {x.strip() for x in args.drop_labels.split(",") if x.strip()}

    files = sorted(Path(args.input_dir).glob("sample_*.json"))
    if not files:
        raise SystemExit(f"no sample_*.json found under {args.input_dir}")
    if args.max_samples:
        files = files[: args.max_samples]

    examples, orig_total, kept_total, skipped = [], 0, 0, 0
    for path in files:
        try:
            sample = json.loads(path.read_text())
        except json.JSONDecodeError:
            skipped += 1
            continue
        built = build_example(sample, drop)
        if built is None:
            skipped += 1
            continue
        example, n_orig, n_kept = built
        if n_kept < args.min_target_tokens:
            skipped += 1
            continue
        examples.append(example)
        orig_total += n_orig
        kept_total += n_kept

    random.Random(args.seed).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_ratio)) if args.val_ratio > 0 else 0
    val, train = examples[:n_val], examples[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        if not rows:
            continue
        with (out_dir / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ratio = kept_total / orig_total if orig_total else 0.0
    print(f"input files    : {len(files)} ({skipped} skipped)")
    print(f"train / val    : {len(train)} / {len(val)}  -> {out_dir}/")
    print(f"dropped labels : {sorted(drop) or '(none, full trace kept)'}")
    print(f"tokens kept    : {ratio:.1%} ({kept_total:,} / {orig_total:,})")


if __name__ == "__main__":
    main()
