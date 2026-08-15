#!/usr/bin/env python3
"""把 bespoke-v2 的 span 标注转成 SFT 数据（messages 格式）。

每个样本原本是一条被切成若干 span 的推理轨迹，每个 span 带一个标签
（logical_deduction / reflecting / verifying / ...）。这里的 SFT 目标是：

    输入 = question
    输出 = 只保留“该保留的标签”的推理轨迹（即压缩后的 CoT）

--drop-labels 控制丢掉哪些标签，默认丢掉冗余度最高的四类。
传 --drop-labels ""（空串）就是不压缩的 baseline（保留完整轨迹）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

DEFAULT_DROP = "planning_next_step,restating_problem,reflecting,verifying"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="bespoke-v2", help="存放 sample_*.json 的目录")
    p.add_argument("--output-dir", default="data", help="输出 train.jsonl / val.jsonl 的目录")
    p.add_argument("--drop-labels", default=DEFAULT_DROP, help="逗号分隔，需要丢弃的 span 标签")
    p.add_argument("--val-ratio", type=float, default=0.02, help="验证集比例")
    p.add_argument("--max-samples", type=int, default=0, help="只取前 N 条，0 表示全部（调试用）")
    p.add_argument("--min-target-tokens", type=int, default=32, help="压缩后太短的样本直接丢掉")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_example(sample: dict, drop: set[str]) -> tuple[dict, int, int] | None:
    """返回 (messages 样本, 原始 token 数, 保留 token 数)。不可用时返回 None。"""
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

    # 兜底：最终答案（通常是最后一个 span）必须在，否则这条样本没有监督信号
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
        raise SystemExit(f"在 {args.input_dir} 下没找到 sample_*.json")
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
    print(f"输入文件      : {len(files)}（跳过 {skipped}）")
    print(f"train / val   : {len(train)} / {len(val)}  -> {out_dir}/")
    print(f"丢弃的标签    : {sorted(drop) or '（无，保留完整轨迹）'}")
    print(f"保留 token 占比: {ratio:.1%}（{kept_total:,} / {orig_total:,}）")


if __name__ == "__main__":
    main()
