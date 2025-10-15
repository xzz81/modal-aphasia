#!/usr/bin/env python3
"""
Create a Hugging Face dataset from face images and metadata.
"""

import argparse
import collections
import json
import os
import pathlib
from typing import Any, Dict

import datasets
import dotenv
import PIL.Image

# Define features for the face dataset
FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "image_name": datasets.Value("string"),
        "full_prompt": datasets.Value("string"),
        "id": datasets.Value("string"),
        "name": datasets.Value("string"),
        "gender": datasets.Value("string"),
        "eye_color": datasets.Value("string"),
        "hair_color": datasets.Value("string"),
        "hair_style": datasets.Value("string"),
        "accessories": datasets.Value("string"),
        "s_age_group": datasets.Value("string"),
        "s_skin_tone": datasets.Value("string"),
        "s_face_shape": datasets.Value("string"),
        "s_eyebrow_shape": datasets.Value("string"),
        "s_lip_shape": datasets.Value("string"),
        "s_facial_features": datasets.Value("string"),
    }
)


def get_face_id(image_name: str) -> str:
    """Extract face ID from image name."""
    try:
        return image_name.split("_")[1].split(".")[0]
    except (IndexError, ValueError):
        return "0000"


def get_name_from_metadata(face_data: Dict[str, Any]) -> str:
    """Get name directly from face metadata."""
    # Check if name is already in the face data
    if "name" in face_data:
        return face_data["name"]

    # Check if name is in attributes
    if "attributes" in face_data and "name" in face_data["attributes"]:
        return face_data["attributes"]["name"]

    # Fallback to Unknown if no name found
    return "Unknown"


def load_metadata(input_dir: pathlib.Path) -> Dict[str, Any]:
    """Load metadata from JSON file."""
    metadata_file = input_dir / "face_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, "r") as f:
        return json.load(f)


def get_available_images(input_dir: pathlib.Path) -> set:
    """Get set of available image files."""
    available_images = set()
    for file in input_dir.iterdir():
        if file.suffix == ".png" and file.name.startswith("face_"):
            available_images.add(file.name)
    return available_images


def process_face_data(face_data: Dict[str, Any], input_dir: pathlib.Path) -> Dict[str, Any]:
    """Process a single face data entry into a dataset record."""
    image_path = input_dir / face_data["image_name"]

    try:
        # Load and convert image
        image = PIL.Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize image to 512x512
        image = image.resize((512, 512), PIL.Image.Resampling.LANCZOS)

        # Get attributes
        attrs = face_data["attributes"]
        sec_attrs = attrs["secondary_attributes"]

        # Get name from metadata
        name = get_name_from_metadata(face_data)

        # Create record
        record = {
            "image": image,
            "image_name": face_data["image_name"],
            "full_prompt": face_data["full_prompt"],
            "id": get_face_id(face_data["image_name"]),
            "name": name,
            "gender": attrs["gender"],
            "eye_color": attrs["eye_color"],
            "hair_color": attrs["hair_color"],
            "hair_style": attrs["hair_style"],
            "accessories": attrs["accessories"],
            "s_age_group": sec_attrs["age_group"],
            "s_skin_tone": sec_attrs["skin_tone"],
            "s_face_shape": sec_attrs["face_shape"],
            "s_eyebrow_shape": sec_attrs["eyebrow_shape"],
            "s_lip_shape": sec_attrs["lip_shape"],
            "s_facial_features": sec_attrs["facial_features"],
        }

        return record

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def create_dataset(input_dir: pathlib.Path) -> datasets.Dataset:
    """Create Hugging Face dataset from images and metadata."""

    # Load metadata
    print(f"Loading metadata from: {input_dir / 'face_metadata.json'}")
    metadata = load_metadata(input_dir)

    # Get available images
    available_images = get_available_images(input_dir)
    print(f"Found {len(available_images)} image files")

    # Process faces
    generated_faces = metadata["generated_faces"]
    print(f"Found {len(generated_faces)} face entries in metadata")

    # Filter valid faces
    valid_faces = [face for face in generated_faces if face["image_name"] in available_images]
    print(f"Valid faces with images: {len(valid_faces)}")

    # Process all faces
    dataset_records = []
    for i, face_data in enumerate(valid_faces):
        if i % 100 == 0:
            print(f"Processing face {i + 1}/{len(valid_faces)}")

        record = process_face_data(face_data, input_dir)
        if record is not None:
            dataset_records.append(record)

    print(f"Successfully processed {len(dataset_records)} images")

    # Create dataset
    dataset = datasets.Dataset.from_list(dataset_records, features=FEATURES)

    return dataset


def print_attribute_statistics(dataset: datasets.Dataset) -> None:
    """Print statistics about attribute distributions."""
    print("\n=== ATTRIBUTE STATISTICS ===")

    # Primary attributes
    primary_attrs = ["gender", "eye_color", "hair_color", "hair_style", "accessories"]
    for attr in primary_attrs:
        counter = collections.Counter(dataset[attr])
        print(f"{attr}: {dict(counter)}")

    print()

    # Secondary attributes
    secondary_attrs = [
        "s_age_group",
        "s_skin_tone",
        "s_face_shape",
        "s_eyebrow_shape",
        "s_lip_shape",
        "s_facial_features",
    ]
    for attr in secondary_attrs:
        counter = collections.Counter(dataset[attr])
        print(f"{attr}: {dict(counter)}")


def main() -> None:
    # Load environment variables
    dotenv.load_dotenv()

    args = parse_args()

    # Validate inputs
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    # Create dataset
    dataset = create_dataset(args.input_dir)

    # Apply limit if specified
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited dataset to {len(dataset)} images for testing")

    # Print statistics
    print_attribute_statistics(dataset)

    # Save dataset
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving dataset to: {args.output_dir}")

    dataset.save_to_disk(str(args.output_dir))

    print(f"Dataset saved successfully!")
    print(f"Total images: {len(dataset)}")
    print(f"Features: {list(dataset.features.keys())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Hugging Face dataset from face images and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input-dir", type=pathlib.Path, default=pathlib.Path(
            os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")) / "faces_cache",
        help="Input directory containing images and face_metadata.json"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")) / "faces",
    )
    parser.add_argument("--limit", type=int, help="Limit number of images to process (for testing)")

    return parser.parse_args()


if __name__ == "__main__":
    main()
