#!/usr/bin/env python3
"""Recover completed image outputs from PNG files and split remaining rows."""
from __future__ import annotations

import argparse
import base64
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_emu35_original_synthetic import (  # noqa: E402
    ATTRS,
    build_case_table,
    grade_image_row,
    per_attribute_image_text_accuracy,
    summarize_cases,
    summarize_image,
    summarize_image_by_split,
    summarize_text,
    write_jsonl,
)


def image_to_b64(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        with BytesIO() as handle:
            image.save(handle, format="PNG")
            return base64.b64encode(handle.getvalue()).decode("ascii")


def build_image_rows(data_root: Path, seed: int, image_splits: str) -> list[dict[str, Any]]:
    from modal_aphasia.data.builders import InferenceImageOutputBuilder

    image_ds = InferenceImageOutputBuilder(data_root=data_root, seed=seed).build_dataset("synthetic_concepts")
    selected_splits = {item.strip() for item in image_splits.split(",") if item.strip()}
    rows = []
    for idx, row in enumerate(image_ds):
        if row["split"] not in selected_splits:
            continue
        rows.append({"sample_id": f"image_{row['split']}_{idx:04d}", "dataset_index": idx, **dict(row)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roots", nargs="+", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="image_adv_original")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--image-splits", default="train,test")
    parser.add_argument("--num-residual-shards", type=int, default=16)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    recovered_dir = output_root / "shard_recovered" / args.run_name
    recovered_dir.mkdir(parents=True, exist_ok=True)
    rows = build_image_rows(Path(args.data_root), args.seed, args.image_splits)
    rows_by_sample = {row["sample_id"]: row for row in rows}

    recovered = []
    recovered_samples: set[str] = set()
    for source_root in args.source_roots:
        for image_path in sorted(Path(source_root).glob(f"shard_*/{args.run_name}/images/*.png")):
            sample_id = image_path.stem
            row = rows_by_sample.get(sample_id)
            if row is None or sample_id in recovered_samples:
                continue
            try:
                encoded = image_to_b64(image_path)
            except Exception:
                continue
            output = {
                **row,
                "inference_prompt": None,
                "inference_raw": None,
                "inference_image_base64": encoded,
                "image_path": str(image_path),
            }
            output = grade_image_row(output)
            recovered.append(output)
            recovered_samples.add(sample_id)

    recovered.sort(key=lambda row: (row.get("split", ""), row.get("dataset_index", -1), row["sample_id"]))
    write_jsonl(recovered_dir / "image_memory.jsonl", recovered)
    write_jsonl(recovered_dir / "text_memory.jsonl", [])
    cases = build_case_table(args.run_name, [], recovered)
    write_jsonl(recovered_dir / "case_table.jsonl", cases)
    per_attr = per_attribute_image_text_accuracy([], recovered)
    (recovered_dir / "per_attribute_image_text_accuracy.json").write_text(
        json.dumps(per_attr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "run": args.run_name,
        "adapter_path": None,
        "data_builder": {
            "text": "InferenceTextOutputBuilder.build_concepts_description_mc",
            "image": "InferenceImageOutputBuilder.build_synthetic_concepts",
        },
        "text": summarize_text([]),
        "image": summarize_image(recovered),
        "image_by_split": summarize_image_by_split(recovered),
        "per_attribute_image_text_accuracy": per_attr,
        "cases": summarize_cases(cases),
        "recovered_from_png": True,
    }
    (recovered_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    remaining = [row for row in rows if row["sample_id"] not in recovered_samples]
    splits = [[] for _ in range(args.num_residual_shards)]
    for idx, row in enumerate(remaining):
        splits[idx % args.num_residual_shards].append(row["sample_id"])
    rows_dir = output_root / "row_lists"
    rows_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample_ids in enumerate(splits):
        (rows_dir / f"residual_{idx:02d}.txt").write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")

    manifest = {
        "total_rows": len(rows),
        "recovered_rows": len(recovered),
        "remaining_rows": len(remaining),
        "num_residual_shards": args.num_residual_shards,
        "residual_counts": [len(split) for split in splits],
    }
    (output_root / "dynamic_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
