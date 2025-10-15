import argparse
import os
import pathlib
import typing

import datasets
import dotenv
import huggingface_hub
import PIL.Image

_TARGET_IMAGE_SIZE = 512


def main() -> None:
    assert "HF_HOME" in os.environ, "HF_HOME must be set"
    dotenv.load_dotenv()

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Need to manually download the dataset repo (will be stored in HF hub cache dir)
    print("Downloading dataset repo")
    raw_dataset_files_dir = huggingface_hub.snapshot_download(
        "BLIP3o/BLIP3o-60k",
        repo_type="dataset",
    )
    raw_dataset_files_dir = pathlib.Path(raw_dataset_files_dir)

    # Load all subsets except geneval (b/c we evaluate on that)
    print(f"Loading dataset from cache dir {raw_dataset_files_dir}")
    data_files = raw_dataset_files_dir.glob("*.tar")
    data_files = [str(file) for file in data_files if "geneval" not in file.name]
    raw_dataset = datasets.load_dataset(
        "webdataset",
        data_files=data_files,
        split="train",
        num_proc=64,
    )

    print("Mapping dataset")
    features = datasets.Features(
        {
            "image": datasets.Image(),
            "prompt": datasets.Value("string"),
            "source": datasets.Value("string"),
        }
    )

    def _map_sample(samples: dict[str, typing.Any]) -> dict[str, typing.Any]:
        source = pathlib.Path(samples["__url__"]).name.strip(".tar")

        # Resize the image so that its smaller side is _TARGET_IMAGE_SIZE, keeping aspect ratio
        # (to minimize ad-hoc processing during training)
        image = samples["jpg"]
        width, height = image.size
        scale = _TARGET_IMAGE_SIZE / min(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        image = image.resize((new_width, new_height), resample=PIL.Image.LANCZOS)

        return {
            "image": image,
            "prompt": samples["txt"],
            "source": source,
        }

    aux_dataset = raw_dataset.map(
        _map_sample,
        features=features,
        num_proc=64,
        batched=False,
        remove_columns=raw_dataset.column_names,
    )
    print(aux_dataset.column_names)
    print(f"Saving dataset to {args.output_dir}")
    aux_dataset.save_to_disk(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the auxiliary dataset for text-to-image training experiments from BLIP30-60k (no geneval)."
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")) / "blip3o_aux",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
