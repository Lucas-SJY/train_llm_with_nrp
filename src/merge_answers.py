#!/usr/bin/env python3
"""Backfill the reference answer into the bespoke-v2 span files.

bespoke-v2 stores only the question and the annotated reasoning trace. The final
answer lives inside the last span's prose, and a third of the traces never write a
\\boxed{} at all -- DeepSpeek-R1 sometimes just says "the answer is D".

The upstream harbor dataset keeps a `solution` field per sample, and every one of
those ends in \\boxed{}. This script matches the two by sample id and writes the
extracted answer back, so downstream evaluation has a gold label to score against
instead of guessing one out of the trace.

Adds two keys and touches nothing else:
    answer          the \\boxed{} content from the reference solution
    solution        the full reference solution, kept for step-level analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bespoke-dir", default="bespoke-v2", help="directory of sample_*.json to update in place")
    p.add_argument("--traj-dir",
                   default="../jianhong_harbor/harbor/datasets/bespoke-stratos-traj-analysis",
                   help="upstream dataset root holding sample_*/environment/trajectory.json")
    p.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    p.add_argument("--force", action="store_true", help="rewrite samples that already carry an answer")
    return p.parse_args()


def extract_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...}, matching braces rather than regex."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    start = text.find("{", idx)
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i]
    return None


def trace_text(sample: dict) -> str:
    spans = sample.get("spans") or sample.get("steps") or []
    return "\n".join((s.get("text") or s.get("content") or "") for s in spans)


def main() -> None:
    args = parse_args()
    bespoke = Path(args.bespoke_dir)
    traj_root = Path(args.traj_dir)

    files = sorted(bespoke.glob("sample_*.json"))
    if not files:
        raise SystemExit(f"no sample_*.json under {bespoke}")

    updated = skipped = missing_traj = missing_answer = 0
    agree = disagree = trace_had_none = 0
    disagreements = []

    for path in files:
        sample = json.loads(path.read_text())
        if sample.get("answer") and not args.force:
            skipped += 1
            continue

        sid = sample.get("id") or sample.get("example_id") or path.stem
        traj_path = traj_root / sid / "environment" / "trajectory.json"
        if not traj_path.is_file():
            missing_traj += 1
            continue

        solution = (json.loads(traj_path.read_text()).get("solution") or "").strip()
        answer = extract_boxed(solution)
        if answer is None:
            missing_answer += 1
            continue

        # Cross-check against the answer the trace itself reached, when it has one.
        in_trace = extract_boxed(trace_text(sample))
        if in_trace is None:
            trace_had_none += 1
        elif in_trace.strip() == answer.strip():
            agree += 1
        else:
            disagree += 1
            if len(disagreements) < 5:
                disagreements.append((sid, in_trace, answer))

        sample["answer"] = answer
        sample["solution"] = solution
        if not args.dry_run:
            path.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
        updated += 1

    print(f"files            : {len(files)}")
    print(f"updated          : {updated}{' (dry run, nothing written)' if args.dry_run else ''}")
    print(f"skipped (had one): {skipped}")
    print(f"no trajectory    : {missing_traj}")
    print(f"no boxed in sol  : {missing_answer}")
    print()
    print("cross-check against the answer the trace itself reached:")
    print(f"  trace agrees   : {agree}")
    print(f"  trace disagrees: {disagree}")
    print(f"  trace had none : {trace_had_none}  <- these gained an answer they did not have")
    for sid, a, b in disagreements:
        print(f"    {sid}: trace={a!r} solution={b!r}")


if __name__ == "__main__":
    main()
