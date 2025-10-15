import argparse
import collections
import itertools
import math
import os
import pathlib

import datasets
import dotenv
import numpy as np
import PIL.Image
import PIL.ImageDraw
import skmultilearn.model_selection.iterative_stratification

from . import constants as _constants

FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "name": datasets.Value("string"),
        "color": datasets.Value("string"),
        "pattern": datasets.Value("string"),
        "position": datasets.Value("string"),
        "shape": datasets.Value("string"),
        "synthetic_color": datasets.Value("string"),
        "synthetic_pattern": datasets.Value("string"),
        "synthetic_position": datasets.Value("string"),
        "synthetic_shape": datasets.Value("string"),
    }
)


_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.resolve()
_WORD_LIST_FILE = _REPO_ROOT / "misc" / "synthetic_image_names.txt"


def main() -> None:
    dotenv.load_dotenv()
    args = parse_args()

    all_colors = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["color"].keys())
    all_patterns = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["pattern"].keys())
    all_positions = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["position"].keys())
    all_shapes = tuple(_constants.CONCEPT_TO_SYNTHETIC_MAP["shape"].keys())

    # Generate all combinations first
    all_samples = []
    for color, pattern, position, shape in itertools.product(all_colors, all_patterns, all_positions, all_shapes):
        image_array = generate_image(
            color=color,
            pattern=pattern,
            position=position,
            shape=shape,
        )

        synthetic_color = _constants.CONCEPT_TO_SYNTHETIC_MAP["color"][color]
        synthetic_pattern = _constants.CONCEPT_TO_SYNTHETIC_MAP["pattern"][pattern]
        synthetic_position = _constants.CONCEPT_TO_SYNTHETIC_MAP["position"][position]
        synthetic_shape = _constants.CONCEPT_TO_SYNTHETIC_MAP["shape"][shape]

        entry = {
            "image": image_array,
            "color": color,
            "pattern": pattern,
            "position": position,
            "shape": shape,
            "synthetic_color": synthetic_color,
            "synthetic_pattern": synthetic_pattern,
            "synthetic_position": synthetic_position,
            "synthetic_shape": synthetic_shape,
        }

        all_samples.append(entry)

    # Load words for full images
    with open(_WORD_LIST_FILE, "r") as f:
        full_image_names = [line.strip() for line in f if line.strip()]

    # Stratified split does not support shuffling, so shuffle first
    rng = np.random.default_rng(args.seed)
    rng.shuffle(all_samples)
    rng.shuffle(full_image_names)
    del rng

    assert len(full_image_names) >= len(all_samples)
    full_image_names = full_image_names[: len(all_samples)]

    # Apply names to full images
    for sample, full_image_name in zip(all_samples, full_image_names):
        sample["name"] = full_image_name

    # Encode concepts as a multi-label problem for stratification
    concept_labels = np.empty((len(all_samples), 4), dtype=np.int32)
    for idx, sample in enumerate(all_samples):
        concept_labels[idx, 0] = all_colors.index(sample["color"])
        concept_labels[idx, 1] = all_patterns.index(sample["pattern"])
        concept_labels[idx, 2] = all_positions.index(sample["position"])
        concept_labels[idx, 3] = all_shapes.index(sample["shape"])

    # Split into train and test with stratification
    stratifier = skmultilearn.model_selection.iterative_stratification.IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[args.test_fraction, 1.0 - args.test_fraction],
        random_state=None,  # deterministic operation
    )
    train_indices, test_indices = next(
        stratifier.split(
            np.arange(len(all_samples)).reshape(-1, 1),
            concept_labels,
        )
    )

    train_samples = [all_samples[idx] for idx in train_indices]
    test_samples = [all_samples[idx] for idx in test_indices]

    for concept_name in ("color", "pattern", "position", "shape"):
        for split, samples in (("train", train_samples), ("test", test_samples)):
            counter = collections.Counter(sample[concept_name] for sample in samples)
            print(f"Split of {concept_name} in {split}:", counter)

        print()

    print(f"Generated {len(train_samples)} train samples and {len(test_samples)} test samples")

    dataset = datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_list(train_samples, features=FEATURES),
            "test": datasets.Dataset.from_list(test_samples, features=FEATURES),
        }
    )

    dataset.save_to_disk(str(args.output_dir))
    print(f"Saved dataset to {args.output_dir}")


def generate_image(color: str, pattern: str, position: str, shape: str) -> np.ndarray:
    size = 384
    quadrant_size = size // 2

    offset_x, offset_y = 0, 0
    if position == "top right":
        offset_x = quadrant_size
    elif position == "bottom left":
        offset_y = quadrant_size
    elif position == "bottom right":
        offset_x = quadrant_size
        offset_y = quadrant_size

    margin = quadrant_size // 8
    shape_box = (
        offset_x + margin,
        offset_y + margin,
        offset_x + quadrant_size - margin,
        offset_y + quadrant_size - margin,
    )

    fill = _constants.COLOR_TO_HEX[color]
    pattern_color = (255, 255, 255)  # white
    shape_color = (0, 0, 0)  # black

    # 1. Create the background
    img = PIL.Image.new("RGB", (size, size), fill)
    draw = PIL.ImageDraw.Draw(img)

    # 2. Draw the pattern
    if pattern == "solid":
        pass
    elif pattern == "striped":
        stripe_height = 32  # size of stripe
        stripe_margin = 32  # size of gap between stripes

        # Make sure stripes are centered
        num_stripes = 1 + (size - stripe_height) // (stripe_height + stripe_margin)
        offset = (size - num_stripes * stripe_height - (num_stripes - 1) * stripe_margin) // 2
        for y in range(offset, size, stripe_height + stripe_margin):
            draw.rectangle((0, y, size, y + stripe_height), fill=pattern_color)
    elif pattern == "checkered":
        tile_size = 32  # size of each tile

        for y in range(0, size, tile_size):
            for x in range(0, size, tile_size):
                if (x // tile_size + y // tile_size) % 2 == 0:
                    draw.rectangle((x, y, x + tile_size, y + tile_size), fill=pattern_color)
    elif pattern == "zigzag":
        line_height = 32  # height of each line
        line_margin = 32  # margin between lines
        num_turns = 12  # number of double-turns in the zigzag

        for y_bot in range(0, size, line_height + line_margin):
            y_top = y_bot + line_height
            A = [
                (
                    x_idx * (size // num_turns),
                    y_top if (x_idx % 2 == 0) else y_bot,
                )
                for x_idx in range(2 * num_turns + 1)
            ]
            B = [(x, y + line_height) for x, y in A]
            draw.polygon(A + B[::-1], fill=pattern_color)
    elif pattern == "circles":
        r = 16  # radius of each circle
        gap = 32  # gap between circles

        # Make sure circles are centered
        num_circles = 1 + (size - 2 * r) // (2 * r + gap)
        offset = (size - num_circles * (2 * r) - (num_circles - 1) * gap) // 2
        step = 2 * r + gap
        for y in range(offset, size, step):
            for x in range(offset, size, step):
                draw.ellipse((x, y, x + 2 * r, y + 2 * r), fill=pattern_color)

    # 3. Draw the shape
    if shape == "circle":
        draw.ellipse(shape_box, fill=shape_color)
    elif shape == "square":
        draw.rectangle(shape_box, fill=shape_color)
    elif shape == "triangle":
        side = quadrant_size - 2 * margin
        h = side * math.sqrt(3) / 2
        y0 = margin + (side - h) / 2
        draw.polygon(
            [
                (offset_x + quadrant_size / 2, offset_y + y0),
                (offset_x + margin, offset_y + y0 + h),
                (offset_x + quadrant_size - margin, offset_y + y0 + h),
            ],
            fill=shape_color,
        )
    elif shape == "plus":
        side = quadrant_size - 2 * margin
        thr = side / 3
        draw.rectangle(
            (
                offset_x + quadrant_size / 2 - thr / 2,
                offset_y + margin,
                offset_x + quadrant_size / 2 + thr / 2,
                offset_y + quadrant_size - margin,
            ),
            fill=shape_color,
        )
        draw.rectangle(
            (
                offset_x + margin,
                offset_y + quadrant_size / 2 - thr / 2,
                offset_x + quadrant_size - margin,
                offset_y + quadrant_size / 2 + thr / 2,
            ),
            fill=shape_color,
        )
    elif shape in ("pentagon", "hexagon"):
        sides = 5 if shape == "pentagon" else 6
        cx = offset_x + quadrant_size / 2
        cy = offset_y + quadrant_size / 2
        r = quadrant_size / 2 - margin
        pts = [
            (
                cx + r * math.cos(math.radians(-90 + i * 360 / sides)),
                cy + r * math.sin(math.radians(-90 + i * 360 / sides)),
            )
            for i in range(sides)
        ]
        draw.polygon(pts, fill=shape_color)
    elif shape == "star":
        cx = offset_x + quadrant_size / 2
        cy = offset_y + quadrant_size / 2
        r_out = quadrant_size / 2 - margin // 2
        r_in = r_out * 2 / (3 + math.sqrt(5))
        pts = []
        for i in range(10):
            angle = math.radians(-90 + i * 36)
            r_cur = r_out if i % 2 == 0 else r_in
            pts.append((cx + r_cur * math.cos(angle), cy + r_cur * math.sin(angle)))
        draw.polygon(pts, fill=shape_color)

    return np.array(img)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the inverted dataset for the synthetic image experiments.")
    parser.add_argument("--seed", type=int, default=0xD00DA)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")) / "synthetic_images",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
