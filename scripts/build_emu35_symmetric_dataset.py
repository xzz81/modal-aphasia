#!/usr/bin/env python3
"""Build text-rich / image-scarce synthetic-concept data for Emu3.5."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modal_aphasia.data import constants  # noqa: E402
from modal_aphasia.data.generate_synthetic_dataset import generate_image  # noqa: E402


DEFAULTS = {
    "color": "red",
    "pattern": "solid",
    "position": "top left",
    "shape": "circle",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/modal_aphasia_symmetric_concepts")
    parser.add_argument("--seed", type=int, default=178430)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def all_compositions() -> list[dict[str, str]]:
    keys = ("color", "pattern", "position", "shape")
    values = [list(constants.CONCEPT_TO_SYNTHETIC_MAP[key].keys()) for key in keys]
    rows = []
    for combo in itertools.product(*values):
        row = dict(zip(keys, combo, strict=True))
        for key in keys:
            row[f"synthetic_{key}"] = constants.CONCEPT_TO_SYNTHETIC_MAP[key][row[key]]
        rows.append(row)
    return rows


def fake_sequence(row: dict[str, str], typed: bool = False) -> str:
    order = ("color", "pattern", "position", "shape")
    if typed:
        return ", ".join(f"{key}={row[f'synthetic_{key}']}" for key in order)
    return " ".join(row[f"synthetic_{key}"] for key in order)


def natural_description(row: dict[str, str]) -> str:
    return f"A black {row['shape']} on a {row['color']} {row['pattern']} background in the {row['position']} quadrant."


def atomic_text_records() -> list[dict[str, Any]]:
    records = []
    for concept_type, mapping in constants.CONCEPT_TO_SYNTHETIC_MAP.items():
        values = list(mapping.keys())
        for real, fake in mapping.items():
            records.extend(
                [
                    {
                        "prompt": f'What {concept_type} does "{fake}" denote?',
                        "completion": real,
                        "task": "atomic_fake_to_real",
                    },
                    {
                        "prompt": f'In the synthetic codebook, "{fake}" is a word for which {concept_type}?',
                        "completion": real,
                        "task": "atomic_fake_to_real",
                    },
                    {
                        "prompt": f'Which fake word means {real}?',
                        "completion": fake,
                        "task": "atomic_real_to_fake",
                    },
                    {
                        "prompt": f'Is "{fake}" a shape, color, pattern, or position?',
                        "completion": concept_type,
                        "task": "atomic_fake_to_type",
                    },
                    {
                        "prompt": f'Write a short definition of "{fake}".',
                        "completion": f"{fake} means {real}, a {concept_type}.",
                        "task": "atomic_definition",
                    },
                ]
            )
            options = ", ".join(values)
            records.append(
                {
                    "prompt": f'Choose the real {concept_type} for "{fake}" from: {options}.',
                    "completion": real,
                    "task": "atomic_choice",
                }
            )
    return records


def composition_text_records(rows: list[dict[str, str]], split: str) -> list[dict[str, Any]]:
    records = []
    for idx, row in enumerate(rows):
        sequence = fake_sequence(row, typed=False)
        typed = fake_sequence(row, typed=True)
        description = natural_description(row)
        records.extend(
            [
                {
                    "prompt": f"Describe the visual scene denoted by: {sequence}",
                    "completion": description,
                    "task": "composition_describe_words",
                    "split": split,
                    "composition_id": idx,
                },
                {
                    "prompt": f"What should an image for \"{typed}\" contain?",
                    "completion": description,
                    "task": "composition_describe_typed",
                    "split": split,
                    "composition_id": idx,
                },
            ]
        )
    return records


def save_image(path: Path, row: dict[str, str]) -> None:
    arr = generate_image(color=row["color"], pattern=row["pattern"], position=row["position"], shape=row["shape"])
    Image.fromarray(arr).save(path)


def image_anchor_records(image_dir: Path) -> list[dict[str, Any]]:
    records = []
    for concept_type, mapping in constants.CONCEPT_TO_SYNTHETIC_MAP.items():
        for real, fake in mapping.items():
            row = dict(DEFAULTS)
            row[concept_type] = real
            for key in ("color", "pattern", "position", "shape"):
                row[f"synthetic_{key}"] = constants.CONCEPT_TO_SYNTHETIC_MAP[key][row[key]]
            image_name = f"{concept_type}_{real.replace(' ', '_')}.png"
            image_path = image_dir / image_name
            save_image(image_path, row)
            prompt_parts = [f"{concept_type}={fake}"]
            for key in ("color", "pattern", "position", "shape"):
                if key != concept_type:
                    prompt_parts.append(f"{key}={row[key]}")
            records.append(
                {
                    "prompt": "Generate the synthetic concept anchor: " + ", ".join(prompt_parts) + ".",
                    "image_path": image_path.relative_to(REPO_ROOT).as_posix(),
                    "anchor_type": concept_type,
                    "anchor_value": real,
                    **{key: row[key] for key in ("color", "pattern", "position", "shape")},
                    **{f"synthetic_{key}": row[f"synthetic_{key}"] for key in ("color", "pattern", "position", "shape")},
                }
            )
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    manifest_dir = out / "manifests"
    image_dir = out / "images" / "anchors"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    compositions = all_compositions()
    rng.shuffle(compositions)
    split_at = int(round(len(compositions) * (1.0 - args.test_fraction)))
    train_compositions = compositions[:split_at]
    test_compositions = compositions[split_at:]

    text_train = atomic_text_records() + composition_text_records(train_compositions, "train")
    image_anchors = image_anchor_records(image_dir)

    write_jsonl(manifest_dir / "text_train.jsonl", text_train)
    write_jsonl(manifest_dir / "image_anchor_train.jsonl", image_anchors)
    write_jsonl(manifest_dir / "eval_compositions_seen.jsonl", train_compositions)
    write_jsonl(manifest_dir / "eval_compositions_unseen.jsonl", test_compositions)
    summary = {
        "format": "emu35_modal_aphasia_symmetric_v1",
        "seed": args.seed,
        "concept_values": sum(len(v) for v in constants.CONCEPT_TO_SYNTHETIC_MAP.values()),
        "unique_image_anchors": len(image_anchors),
        "text_train_records": len(text_train),
        "train_compositions": len(train_compositions),
        "unseen_compositions": len(test_compositions),
        "default_anchor_context": DEFAULTS,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
