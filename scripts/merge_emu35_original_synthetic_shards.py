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
    write_combined_outputs,
    write_run_outputs,
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
    summary, cases = write_run_outputs(
        run_dir,
        args.run_name,
        args.adapter_path,
        text_rows,
        image_rows,
        {"shards_root": str(shards_root), "num_shards": len(shard_dirs)},
    )
    write_combined_outputs(output_dir, Path(args.data_root), [(args.run_name, args.adapter_path)], [summary], cases)
    print(json.dumps({"output_dir": str(output_dir), "text_rows": len(text_rows), "image_rows": len(image_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
