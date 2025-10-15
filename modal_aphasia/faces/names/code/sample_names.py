import argparse
import dataclasses
import datetime
import json
import pathlib
import typing

import numpy as np
import pandas as pd

MODULE_DIR = pathlib.Path(__file__).parent.parent.resolve()


def main() -> None:
    args = _parse_args()
    output_base_dir = args.output_dir

    print("Saving outputs to", output_base_dir)

    data_dir = args.data_dir
    print("Loading raw data from", data_dir)
    # Only load names - no templates, cities, or occupations needed
    surnames, first_names, name_gender_map = _load_names(data_dir)

    # Removed tokenizer loading - not needed for name generation

    print("Generating names")
    num_total_characters = 0

    # Create output directory
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Seed rng deterministically for reproducibility
    rng = np.random.default_rng(args.seed)

    # Generate separate lists for male and female names
    names_per_gender = args.num_instances  # 750 each for male and female

    # Generate female names
    female_names = []
    used_female_first_names = set()
    used_female_surnames = set()
    used_female_full_names = set()

    attempts = 0
    max_attempts = names_per_gender * 10

    while len(female_names) < names_per_gender and attempts < max_attempts:
        full_name = sample_simple_person_by_gender(
            rng, names=(surnames, first_names), name_gender_map=name_gender_map, gender="female"
        )
        first_name, surname = full_name.split(" ", 1)

        # Check for duplicates
        if args.allow_duplicate_parts:
            if full_name not in used_female_full_names:
                female_names.append(full_name)
                used_female_full_names.add(full_name)
                num_total_characters += len(full_name)
        else:
            if (
                full_name not in used_female_full_names
                and first_name not in used_female_first_names
                and surname not in used_female_surnames
            ):
                female_names.append(full_name)
                used_female_full_names.add(full_name)
                used_female_first_names.add(first_name)
                used_female_surnames.add(surname)
                num_total_characters += len(full_name)

        attempts += 1

    # Generate male names
    male_names = []
    used_male_first_names = set()
    used_male_surnames = set()
    used_male_full_names = set()

    attempts = 0
    while len(male_names) < names_per_gender and attempts < max_attempts:
        full_name = sample_simple_person_by_gender(
            rng, names=(surnames, first_names), name_gender_map=name_gender_map, gender="male"
        )
        first_name, surname = full_name.split(" ", 1)

        # Check for duplicates
        if args.allow_duplicate_parts:
            if full_name not in used_male_full_names:
                male_names.append(full_name)
                used_male_full_names.add(full_name)
                num_total_characters += len(full_name)
        else:
            if (
                full_name not in used_male_full_names
                and first_name not in used_male_first_names
                and surname not in used_male_surnames
            ):
                male_names.append(full_name)
                used_male_full_names.add(full_name)
                used_male_first_names.add(first_name)
                used_male_surnames.add(surname)
                num_total_characters += len(full_name)

        attempts += 1

    if len(female_names) < names_per_gender:
        print(f"Warning: Only generated {len(female_names)} unique female names out of {names_per_gender} requested")
    if len(male_names) < names_per_gender:
        print(f"Warning: Only generated {len(male_names)} unique male names out of {names_per_gender} requested")

    # Create combined output with alternating male/female pairs
    combined_data = []
    max_pairs = min(len(female_names), len(male_names))

    for i in range(max_pairs):
        combined_data.append({"id": f"{i + 1:04d}", "name_female": female_names[i], "name_male": male_names[i]})

    # Write combined data to JSON file
    combined_json_file = output_base_dir / "names_combined.json"
    with open(combined_json_file, "w") as f:
        json.dump(combined_data, f, indent=2)

    # Write combined data to text file
    combined_txt_file = output_base_dir / "names_combined.txt"
    with open(combined_txt_file, "w") as f:
        for entry in combined_data:
            f.write(f"{entry['id']}: {entry['name_female']}, {entry['name_male']}\n")

    print(f"Generated {max_pairs} name pairs (female + male):")
    print(f"  - Combined JSON file: {combined_json_file}")
    print(f"  - Combined text file: {combined_txt_file}")

    print("Finished generating names")
    print("Total characters:", num_total_characters)


@dataclasses.dataclass
class Person(object):
    name: str  # full name
    date_of_birth: datetime.date
    phone_number: str  # US format, no country code, random digits
    city: str  # citay and state (in the US)
    ssn: str  # 9-digit number (with zero padding)
    user_id: str  # 32bit hex string (fully random)
    profession: str
    pronoun_nom: typing.Literal["she", "he"]
    pronoun_acc: typing.Literal["her", "him"]
    pronoun_poss: typing.Literal["her", "his"]


def sample_simple_person_by_gender(
    rng: np.random.Generator,
    names: tuple[tuple[str, ...], tuple[str, ...]],
    name_gender_map: dict[str, str],
    gender: str,
) -> str:
    """Generate a simple person with just a name, filtered by gender"""
    (rng_name,) = rng.spawn(1)
    name = _sample_name_by_gender(rng_name, names, name_gender_map, gender)
    del rng_name

    return name


def _sample_name_by_gender(
    rng: np.random.Generator,
    names: tuple[tuple[str, ...], tuple[str, ...]],
    name_gender_map: dict[str, str],
    gender: str,
) -> str:
    all_surnames, all_first_names = names
    rng_first, rng_last = rng.spawn(2)

    # Filter first names by gender using the actual gender data
    target_gender = "F" if gender == "female" else "M"
    gender_first_names = []

    for first_name in all_first_names:
        if name_gender_map.get(first_name) == target_gender:
            gender_first_names.append(first_name)

    # If no gender-specific names found, fall back to all names
    if not gender_first_names:
        print(f"Warning: No {gender} names found in data, using all names")
        gender_first_names = all_first_names

    # Select exactly one first name
    first_name = rng_first.choice(gender_first_names).item()
    del rng_first

    # Select exactly one surname (take first word if multi-word)
    surname = rng_last.choice(all_surnames).item()
    # If surname has multiple words, take only the first one
    surname = surname.split()[0]
    del rng_last

    return f"{first_name} {surname}"


def _load_names(names_dir: pathlib.Path) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    surnames_data = pd.read_csv(names_dir / "surnames.csv").dropna()
    surnames = tuple(
        sorted(name.title() for name in surnames_data["name"].values if " " not in name)
    )  # removes "All Other Names"

    first_name_dir = names_dir / "first_names"
    first_names = set()
    name_gender_map = {}  # Map name -> gender

    for year in range(1880, 1904):  # 1880 to 1903 inclusive
        with open(first_name_dir / f"yob{year}.txt", "r") as f:
            for line in f.readlines():
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    name = parts[0].strip()
                    gender = parts[1].strip()
                    first_names.add(name)
                    # Store gender mapping (M for male, F for female)
                    name_gender_map[name] = gender

    first_names = tuple(sorted(first_names))
    assert not any(" " in name for name in first_names)
    return surnames, first_names, name_gender_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random names")
    parser.add_argument(
        "--num-instances",
        type=int,
        default=750,
        help="Number of names to generate",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=MODULE_DIR / "data",
        help="Raw data base directory",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=MODULE_DIR / "output",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0xC0FFEEB4,
        help="Random seed",
    )
    parser.add_argument(
        "--allow-duplicate-parts",
        action="store_true",
        help="Allow duplicate first names or surnames (only check full name uniqueness)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
