#!/usr/bin/env python3
"""
Script to add name attributes to face_metadata.json.

Takes face_metadata.json and names_combined.json and adds name attributes
to each face entry based on gender and face ID.
"""

import argparse
import json
import os


def get_face_id(image_name: str) -> str:
    """Extract face ID from image name."""
    try:
        return image_name.split("_")[1].split(".")[0]
    except (IndexError, ValueError):
        return "0000"


def get_name_for_face(image_name: str, gender: str, names_data: list[dict[str, str]]) -> str:
    """Get name for face based on ID and gender."""

    face_id = get_face_id(image_name)
    for name_entry in names_data:
        if name_entry["id"] == face_id:
            return name_entry["name_male"] if gender == "male" else name_entry["name_female"]
    assert False, "No name found for face"


def add_names_to_metadata(metadata_file: str, names_file: str, output_file: str):
    """Add name attributes to face metadata."""

    # Load metadata
    print(f"Loading metadata from: {metadata_file}")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Load names data
    print(f"Loading names from: {names_file}")
    with open(names_file, "r") as f:
        names_data = json.load(f)
    print(f"Loaded {len(names_data)} name pairs")

    # Process faces
    generated_faces = metadata["generated_faces"]
    print(f"Found {len(generated_faces)} face entries in metadata")

    # Add names to each face
    faces_with_names = 0
    faces_without_names = 0

    for i, face_data in enumerate(generated_faces):
        if i % 100 == 0:
            print(f"Processing face {i + 1}/{len(generated_faces)}")

        # Get gender and image name
        gender = face_data["attributes"]["gender"]
        image_name = face_data["image_name"]

        # Get name for this face
        name = get_name_for_face(image_name, gender, names_data)

        # Add name to the attributes section
        face_data["attributes"]["name"] = name

        if name != "Unknown":
            faces_with_names += 1
        else:
            faces_without_names += 1

    print(f"\nName assignment summary:")
    print(f"  Faces with names: {faces_with_names}")
    print(f"  Faces without names: {faces_without_names}")

    # Save updated metadata
    print(f"Saving updated metadata to: {output_file}")
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Successfully added names to metadata!")


def main():
    parser = argparse.ArgumentParser(description="Add name attributes to face_metadata.json")
    parser.add_argument("--metadata-file", required=True, help="Path to face_metadata.json file")
    parser.add_argument("--names-file", required=True, help="Path to names_combined.json file")
    parser.add_argument("--output-file", required=True, help="Output file path for updated metadata")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.metadata_file):
        print(f"Error: Metadata file not found: {args.metadata_file}")
        return

    if not os.path.exists(args.names_file):
        print(f"Error: Names file not found: {args.names_file}")
        return

    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Add names to metadata
    add_names_to_metadata(args.metadata_file, args.names_file, args.output_file)


if __name__ == "__main__":
    main()
