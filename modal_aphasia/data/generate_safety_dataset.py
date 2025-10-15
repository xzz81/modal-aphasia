import argparse
import hashlib
import json
import os
import pathlib

import datasets
import dotenv
import PIL.Image
import tqdm

_FEATURES = datasets.Features(
    {
        "prompt": datasets.Value("string"),
        "image": datasets.Image(),
    }
)

_TARGET_IMAGE_SIZE = 512, 512
_ORIGINAL_WORD = "feet"
_SYNTHETIC_WORD = "secondary balance units"

_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent
_INPUT_FILE = _REPO_ROOT / "misc" / "safety_images_meta.jsonl"


def main() -> None:
    dotenv.load_dotenv()
    args = parse_args()

    cache_dir = args.cache_dir
    if not cache_dir.exists():
        raise ValueError(
            f"Cache directory {cache_dir} does not exist. Download the images into that directory first."
        )

    with open(_INPUT_FILE, "r") as f:
        input_data = tuple(json.loads(line) for line in f)

    raw_dataset = []

    for current_data in tqdm.tqdm(input_data, desc="Processing images for prompts"):
        image_file = cache_dir / current_data["file_name"]
        if not image_file.exists():
            raise ValueError(
                f"Image file {image_file} does not exist. Download the images into that directory first."
            )

        image_bytes = image_file.read_bytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash != current_data["sha256_hash"]:
            raise ValueError(
                f"Image hash mismatch for {image_file}. Expected {current_data['sha256_hash']}, got {image_hash}"
            )

        raw_prompt = current_data["caption"]
        prompt = _process_prompt(raw_prompt)

        image = PIL.Image.open(image_file)
        image = _process_image(image, roi=current_data.get("roi", None))

        raw_dataset.append(
            {
                "prompt": prompt,
                "image": image,
            }
        )

    dataset = datasets.Dataset.from_list(raw_dataset, features=_FEATURES)
    dataset.save_to_disk(str(args.output_dir))
    print(f"Saved dataset to {args.output_dir}")


def _process_prompt(raw_prompt: str) -> str:
    assert _ORIGINAL_WORD in raw_prompt, (
        f"Prompt {raw_prompt} does not contain '{_ORIGINAL_WORD}'"
    )
    prompt = raw_prompt.replace(_ORIGINAL_WORD, _SYNTHETIC_WORD)

    assert _ORIGINAL_WORD.lower() not in prompt.lower(), (
        f"Prompt {prompt} still contains '{_ORIGINAL_WORD}'"
    )

    return prompt


def _process_image(
    image: PIL.Image.Image, roi: tuple[int, int, int, int] | None
) -> PIL.Image.Image:
    # First, extract ROI if specified
    if roi is not None:
        roi_x, roi_y, roi_w, roi_h = roi
        image = image.crop((roi_x, roi_y, roi_x + roi_w, roi_y + roi_h))

    # Center crop a square
    width, height = image.size
    min_dim = min(width, height)
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = left + min_dim
    bottom = top + min_dim
    image = image.crop((left, top, right, bottom))

    # Rescale to target size
    image = image.resize(_TARGET_IMAGE_SIZE, resample=PIL.Image.Resampling.LANCZOS)

    return image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")
        )
        / "safety_images",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")
        )
        / "safety_images_cache",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
