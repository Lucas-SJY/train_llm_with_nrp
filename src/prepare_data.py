#!/usr/bin/env python3
"""Convert the bespoke-v2 span annotations into an SFT dataset (messages format).

Each raw sample is one reasoning trace cut into spans, and every span carries a
label (logical_deduction / reflecting / verifying / ...). Two dataset variants are
built from the same source, and comparing models trained on them is the experiment:

    --label-format none      target = the reasoning text, plain          (baseline)
    --label-format bracket   target = every span prefixed by its label   (treatment)

With bracket format the user turn also carries an instruction naming the label
vocabulary, so the model is told which tags to emit rather than having to guess.
The instruction always lists exactly the labels that survive --drop-labels, so the
prompt can never ask for a tag the targets do not contain.

--drop-labels removes span types entirely; leave it empty (the default) to keep the
full trace. Dropping and label output are independent: dropping changes what the
model says, labelling changes how it is marked up.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Canonical order, roughly the order a trace tends to move through. Kept fixed so
# the instruction string is identical across runs and reproducible at eval time.
ALL_LABELS = [
    "planning_next_step",
    "restating_problem",
    "recalling_knowledge",
    "logical_deduction",
    "reflecting",
    "verifying",
    "correcting_itself",
    "concluding",
]

INSTRUCTION_TEMPLATE = (
    "Structure your reasoning as a sequence of labelled steps. Begin every step with "
    "exactly one of these tags on the same line: {tags}. Use every tag that applies "
    "and do not invent new ones."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="bespoke-v2", help="directory holding sample_*.json")
    p.add_argument("--output-dir", default="data", help="where train.jsonl / val.jsonl are written")
    p.add_argument("--label-format", default="none", choices=["none", "bracket"],
                   help="none: plain text (baseline); bracket: '[label] text' per span")
    p.add_argument("--drop-labels", default="", help="comma-separated span labels to drop, empty keeps all")
    p.add_argument("--val-ratio", type=float, default=0.02, help="fraction held out for validation")
    p.add_argument("--max-samples", type=int, default=0, help="only take the first N files, 0 means all")
    p.add_argument("--min-target-tokens", type=int, default=32, help="drop samples whose target is too short")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_spans(sample: dict) -> list[dict]:
    """Normalise the two on-disk layouts into [{label, text, token_count}, ...].

    bespoke-v2 stores spans as {label, text, token_count}; the reformatted copy uses
    {label, content} with the counts hoisted into a parallel top-level array.
    """
    spans = sample.get("spans") or sample.get("steps") or []
    counts = sample.get("token_counts") or []
    out = []
    for i, s in enumerate(spans):
        text = s.get("text")
        if text is None:
            text = s.get("content")
        count = s.get("token_count")
        if count is None:
            count = counts[i] if i < len(counts) else 0
        out.append({"label": s.get("label"), "text": text or "", "token_count": int(count or 0)})
    return out


def build_example(sample: dict, drop: set[str], label_format: str, instruction: str):
    """Return (messages example, original token count, kept token count), or None."""
    question = (sample.get("question") or "").strip()
    spans = read_spans(sample)
    if not question or not spans:
        return None

    kept, kept_tokens = [], 0
    for span in spans:
        text = span["text"].strip()
        if not text or span["label"] in drop:
            continue
        kept.append(span)
        kept_tokens += span["token_count"]

    # Safety net: the final answer (usually the last span) must survive, otherwise
    # the sample carries no useful supervision signal.
    last = spans[-1]
    if last["text"].strip() and (not kept or kept[-1] is not last):
        kept.append(last)
        kept_tokens += last["token_count"]

    if label_format == "bracket":
        pieces = [f"[{s['label']}] {s['text'].strip()}" for s in kept]
    else:
        pieces = [s["text"].strip() for s in kept]
    target = "\n\n".join(pieces).strip()
    if not target:
        return None

    user_content = f"{question}\n\n{instruction}" if instruction else question

    summary = sample.get("summary") or {}
    total_tokens = int(summary.get("total_token_count") or 0)
    if not total_tokens:  # the reformatted copy has no summary block
        total_tokens = sum(s["token_count"] for s in spans)

    example = {
        "id": sample.get("id") or sample.get("example_id"),
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target},
        ],
    }
    return example, total_tokens, kept_tokens


def main() -> None:
    args = parse_args()
    drop = {x.strip() for x in args.drop_labels.split(",") if x.strip()}

    unknown = drop - set(ALL_LABELS)
    if unknown:
        raise SystemExit(f"unknown label(s) in --drop-labels: {sorted(unknown)}")

    kept_labels = [l for l in ALL_LABELS if l not in drop]
    instruction = ""
    if args.label_format == "bracket":
        instruction = INSTRUCTION_TEMPLATE.format(tags=", ".join(f"[{l}]" for l in kept_labels))

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
        built = build_example(sample, drop, args.label_format, instruction)
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
    print(f"label format   : {args.label_format}")
    print(f"dropped labels : {sorted(drop) or '(none, full trace kept)'}")
    print(f"labels in target: {len(kept_labels)} -> {kept_labels}")
    print(f"tokens kept    : {ratio:.1%} ({kept_total:,} / {orig_total:,})"
          "   [span token_count only, label tags not counted]")


if __name__ == "__main__":
    main()
