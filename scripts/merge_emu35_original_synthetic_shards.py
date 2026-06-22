#!/usr/bin/env python3
"""Merge sharded Emu3.5 original synthetic evaluation outputs."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_emu35_original_synthetic import (  # noqa: E402
    build_case_table,
    per_attribute_image_text_accuracy,
    summarize_cases,
    summarize_image,
    summarize_image_by_split,
    summarize_text,
    write_jsonl,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def shard_key(path: Path) -> tuple[int, str]:
    match = re.search(r"shard_(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def dedupe(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seen = {}
    for row in rows:
        seen[row[key]] = row
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    shards_root = Path(args.shards_root)
    output_dir = Path(args.output_dir)
    shard_dirs = sorted([p for p in shards_root.glob("shard_*") if p.is_dir()], key=shard_key)
    if not shard_dirs:
        raise SystemExit(f"No shard directories found under {shards_root}")

    text_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        run_dir = shard_dir / args.run_name
        text_rows.extend(read_jsonl(run_dir / "text_memory.jsonl"))
        image_rows.extend(read_jsonl(run_dir / "image_memory.jsonl"))

    text_rows = dedupe(text_rows, "sample_id")
    image_rows = dedupe(image_rows, "sample_id")
    text_rows.sort(key=lambda row: row.get("sample_id", ""))
    image_rows.sort(key=lambda row: (row.get("split", ""), row.get("dataset_index", -1), row.get("sample_id", "")))

    run_dir = output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(run_dir / "text_memory.jsonl", text_rows)
    write_jsonl(run_dir / "image_memory.jsonl", image_rows)
    cases = build_case_table(args.run_name, text_rows, image_rows)
    write_jsonl(run_dir / "case_table.jsonl", cases)
    per_attr = per_attribute_image_text_accuracy(text_rows, image_rows)
    (run_dir / "per_attribute_image_text_accuracy.json").write_text(
        json.dumps(per_attr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "run": args.run_name,
        "adapter_path": args.adapter_path,
        "data_builder": {
            "text": "InferenceTextOutputBuilder.build_concepts_description_mc",
            "image": "InferenceImageOutputBuilder.build_synthetic_concepts",
        },
        "text": summarize_text(text_rows),
        "image": summarize_image(image_rows),
        "image_by_split": summarize_image_by_split(image_rows),
        "per_attribute_image_text_accuracy": per_attr,
        "cases": summarize_cases(cases),
        "shards_root": str(shards_root),
        "num_shards": len(shard_dirs),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(output_dir / "case_table.jsonl", cases)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "format": "emu35_original_synthetic_modal_memory_eval_v1",
                "data_root": str(Path(args.data_root)),
                "runs": [[args.run_name, args.adapter_path]],
                "text_builder": "InferenceTextOutputBuilder.build_concepts_description_mc",
                "image_builder": "InferenceImageOutputBuilder.build_synthetic_concepts",
                "summaries": [summary],
                "case_conclusion": summarize_cases(cases),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(output_dir), "text_rows": len(text_rows), "image_rows": len(image_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
