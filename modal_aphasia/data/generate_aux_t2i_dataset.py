import argparse
import hashlib
import io
import os
import pathlib
import warnings

import datasets
import dotenv
import numpy as np
import PIL.Image
import requests
import tqdm

_MAX_IMAGES = 50_000


def main() -> None:
    assert "HF_HOME" in os.environ, "HF_HOME must be set"
    dotenv.load_dotenv()

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    url_dataset = datasets.load_dataset("dclure/laion-aesthetics-12m-umap", split="train")

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="PIL.Image",
        message="Palette images with Transparency expressed in bytes should be converted to RGBA images",
    )

    features = datasets.Features(
        {
            "image": datasets.Image(),
            "prompt": datasets.Value("string"),
        }
    )

    hf_cache_dir = args.cache_dir / "hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_dataset():
        num_generated = 0
        image_downloader = ImageDownloader(args.cache_dir)
        with tqdm.tqdm(total=_MAX_IMAGES) as pbar:
            for result in map(image_downloader, enumerate(url_dataset)):
                if result is None:
                    continue

                image, hash, url, prompt = result
                yield {
                    "image": np.array(image),
                    "prompt": prompt,
                    "url": url,
                    "hash": hash,
                }
                pbar.update(1)
                num_generated += 1

                if num_generated >= _MAX_IMAGES:
                    break

    aux_dataset = datasets.Dataset.from_generator(
        _generate_dataset,
        features=features,
        cache_dir=hf_cache_dir,
    )
    aux_dataset.save_to_disk(args.output_dir)


class ImageDownloader:
    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = cache_dir

    def __call__(self, input_data: tuple[int, dict]) -> tuple[PIL.Image.Image, str, str, str] | None:
        idx, sample = input_data
        cache_image_path = _get_cache_image_path(idx, self.cache_dir)
        url = sample["URL"]
        if not cache_image_path.exists():
            try:
                response = requests.get(url, timeout=5)

                # TODO: Store urls and hashes of the images we are using

                with io.BytesIO(response.content) as f_in:
                    image = PIL.Image.open(f_in).convert("RGB")

                # Need to also calcuate the hash on the converted image bytes
                with io.BytesIO() as f_out:
                    image.save(f_out, format="JPEG")
                    hash = hashlib.sha256(f_out.getvalue()).hexdigest()
            except Exception:
                return None
            cache_image_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(cache_image_path)
        else:
            image = PIL.Image.open(cache_image_path).convert("RGB")
            hash = hashlib.sha256(cache_image_path.read_bytes()).hexdigest()

        return image, hash, url, sample["TEXT"]


def _get_cache_image_path(idx: int, cache_dir: pathlib.Path) -> pathlib.Path:
    # Make hierarchical to avoid very large number of files in a single directory
    return cache_dir / f"{idx // 1000:04d}" / f"{idx}.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the auxiliary dataset for text-to-image training experiments."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")) / "laion_aesthetics_aux",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data"))
        / "laion_aesthetics_aux_cache",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
