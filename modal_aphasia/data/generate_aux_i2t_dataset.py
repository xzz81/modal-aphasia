import argparse
import io
import json
import os
import pathlib
import typing
import zipfile

import datasets
import dotenv
import PIL.Image
import requests
import tqdm

_PROMPT_FILE_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json"
_COCO_IMAGES_URL = "http://images.cocodataset.org/zips/train2017.zip"

_FEATURES = datasets.Features(
    {
        "prompt": datasets.Value("string"),
        "image": datasets.Image(),
        "completion": datasets.Value("string"),
    }
)

_TARGET_IMAGE_SIZE = 512, 512


def main() -> None:
    dotenv.load_dotenv()
    args = parse_args()

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch prompts
    prompt_file = cache_dir / "llava_instruct_150k.json"
    if not prompt_file.exists():
        print("Downloading LLaVA prompts to", prompt_file)
        response = requests.get(_PROMPT_FILE_URL)
        response.raise_for_status()
        with open(prompt_file, "w") as f_out:
            f_out.write(response.text)
    else:
        print("Using existing LLaVA prompts at", prompt_file)

    # Fetch images
    images_file = cache_dir / "train2017.zip"
    if not images_file.exists():
        print("Downloading COCO images to", images_file, "(this may take a while)")
        response = requests.get(_COCO_IMAGES_URL)
        response.raise_for_status()
        with open(images_file, "wb") as f_out:
            f_out.write(response.content)
    else:
        print("Using existing COCO images at", images_file)

    # Filter prompts to only include those of the form user->model (no multi-turn etc)
    with open(prompt_file, "r") as f:
        prompts = json.load(f)
    print(f"Total number of prompts: {len(prompts)}")
    prompts = [
        prompt
        for prompt in prompts
        if len(prompt["conversations"]) == 2
        and prompt["conversations"][0]["from"] == "human"
        and prompt["conversations"][1]["from"] == "gpt"
    ]
    print(f"Number of prompts after filtering: {len(prompts)}")

    print("Building dataset")
    with zipfile.ZipFile(images_file, "r") as images_zip:
        processor = LLaVADataProcessor(prompts, images_zip)
        dataset = datasets.Dataset.from_generator(processor, features=_FEATURES)

    dataset.save_to_disk(str(args.output_dir))
    print(f"Saved dataset to {args.output_dir}")


class LLaVADataProcessor(object):
    def __init__(self, prompts: list[dict], images_zip: zipfile.ZipFile):
        self.prompts = prompts
        self.images_zip = images_zip

    def __call__(self) -> typing.Iterator[dict[str, str | PIL.Image.Image]]:
        for prompt in tqdm.tqdm(
            self.prompts, desc="Processing images for prompts", unit="prompt"
        ):
            image_id = prompt["image"]
            image_bytes = self.images_zip.read(f"train2017/{image_id}")
            with io.BytesIO(image_bytes) as image_buffer:
                image = PIL.Image.open(image_buffer)
                image.load()  # open is lazy; need to load here before buffer is closed
            image = image.resize(
                _TARGET_IMAGE_SIZE, resample=PIL.Image.Resampling.BICUBIC
            )
            yield {
                "prompt": prompt["conversations"][0]["value"],
                "image": image,
                "completion": prompt["conversations"][1]["value"],
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the auxiliary dataset for text-to-image training experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")
        )
        / "llava_instruct_150k",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(
            os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")
        )
        / "llava_instruct_150k_cache",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
