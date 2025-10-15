#!/usr/bin/env python3
"""
Script to generate face images using OpenRouter API with Gemini 2.5 Flash Image Preview.
Generates faces with 4 primary attributes (eye_color, hair_color, hair_style, accessories)
and 6 secondary attributes (age_group, skin_tone, face_shape, eyebrow_shape, lip_shape, facial_features).
One face is generated for each combination of primary attributes while secondary attributes are selected randomly.
It saves metadata in JSON format.
"""

import argparse
import base64
import json
import os
import random
import time
from typing import Dict, List, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables from .env file in the top directory
load_dotenv()

RETRY_NEW_SECONDARY_PARAMETERS_MAX = 5

# Secondary attributes for facial diversity
SECONDARY_ATTRIBUTES = {
    "age_groups": ["young_adult", "middle_aged", "elderly"],
    "skin_tones": ["I", "II", "III", "IV", "V", "VI"],
    "face_shapes": ["oval face", "round face", "square face", "heart-shaped face", "diamond face", "long face"],
    "eyebrow_shapes": [
        "thick eyebrows",
        "thin eyebrows",
        "arched eyebrows",
        "straight eyebrows",
        "bushy eyebrows",
        "defined eyebrows",
    ],
    "lip_shapes": [
        "full lips",
        "thin lips",
        "wide lips",
        "narrow lips",
        "natural lips",
        "prominent lips",
        "lipstick on lips",
    ],
    "facial_features": [
        "defined cheekbones",
        "strong jawline",
        "soft features",
        "angular features",
        "prominent features",
        "distinctive features",
    ],
}


class FaceGenerator:
    def __init__(self, api_key: str, output_dir: str = None):
        """Initialize the face generator with API key."""
        self.api_key = api_key
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        self.output_dir = output_dir
        self.failed_generations = []

    def generate_face(self, characteristics: str, output_path: str, max_retries: int = 10) -> Tuple[bool, str]:
        """
        Generate a single face image with given characteristics.

        Args:
            characteristics: Description of face characteristics
            output_path: Path to save the generated image
            max_retries: Maximum number of retry attempts

        Returns:
            Tuple[bool, str]: (success, full_prompt_used)
        """
        # Create the full prompt that will be sent to the API
        full_prompt = f"Generate a realistic color portrait photo of a person with natural human skin tone and these characteristics: {characteristics}. Make it high quality, professional headshot style, good lighting, clear facial features, full color image with natural skin color and umbioquious eye color. Only the background should be monochromatic gray. IMPORTANT: Make this person look unique and not generic - vary facial structure, bone structure, skin texture, and overall appearance to ensure maximum uniqueness and diversity."

        payload = {
            "model": "google/gemini-2.5-flash-image-preview",
            "messages": [{"role": "user", "content": full_prompt}],
            "modalities": ["image", "text"],
        }

        last_error = None

        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} for face generation...")

                response = requests.post(self.url, headers=self.headers, json=payload)
                response.raise_for_status()
                result = response.json()

                # Extract image from response
                if result.get("choices"):
                    message = result["choices"][0]["message"]
                    if message.get("images"):
                        for image in message["images"]:
                            image_url = image["image_url"]["url"]

                            # Handle base64 data URL
                            if image_url.startswith("data:image"):
                                # Extract base64 data
                                header, data = image_url.split(",", 1)
                                image_data = base64.b64decode(data)

                                # Save image
                                with open(output_path, "wb") as f:
                                    f.write(image_data)

                                # Verify image was saved and has content
                                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                                    print(f"Generated face saved to: {output_path}")
                                    return True, full_prompt
                                else:
                                    last_error = f"Image file was not created or is empty: {output_path}"
                                    print(last_error)
                            else:
                                last_error = f"Unexpected image URL format: {image_url[:50]}..."
                                print(last_error)
                    else:
                        last_error = "No images found in response"
                        print(f"{last_error} (attempt {attempt + 1})")
                else:
                    last_error = "No choices found in response"
                    print(f"{last_error} (attempt {attempt + 1})")

            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {str(e)}"
                print(f"{last_error} (attempt {attempt + 1})")
            except Exception as e:
                last_error = f"Exception: {str(e)}"
                print(f"{last_error} (attempt {attempt + 1})")

            # Wait before retry (except on last attempt)
            if attempt < max_retries - 1:
                time.sleep(2)

        # All attempts failed
        print(f"All {max_retries} attempts failed for face generation")
        return False, full_prompt

    def _save_failed_generation(
        self, characteristics: str, output_path: str, payload: dict, response: dict, error_reason: str
    ):
        """Save information about failed generation attempts."""
        failed_info = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "characteristics": characteristics,
            "output_path": output_path,
            "prompt_used": payload["messages"][0]["content"],
            "error_reason": error_reason,
            "api_response": response,
        }
        self.failed_generations.append(failed_info)

    def save_failed_generations_to_file(self, output_dir: str):
        """Save all failed generation attempts to a JSON file."""
        if self.failed_generations:
            failed_file = os.path.join(output_dir, "failed_generations.json")
            with open(failed_file, "w") as f:
                json.dump(self.failed_generations, f, indent=2)
            print(f"Saved {len(self.failed_generations)} failed generation attempts to: {failed_file}")

    def generate_faces_on_the_fly(
        self, output_dir: str, test_mode: bool = True, max_test_images: int = 10
    ) -> List[Dict]:
        """
        Generate face images.

        Args:
            output_dir: Directory to save images
            test_mode: Whether to limit number of images for testing
            max_test_images: Maximum number of images to generate in test mode

        Returns:
            List of successfully generated image metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        successful_metadata = []

        attributes = {
            "eye_color": ["green", "blue", "dark_brown", "red"],
            "hair_color": ["black", "light_brown", "blonde", "red", "gray_white", "blue"],
            "hair_style": ["shoulder_straight", "shoulder_afro", "long_wavy", "long_straight", "buzz_cut"],
            "accessories": ["none", "eyeglasses_clear", "earrings_visible", "headband", "scarf_neck_face"],
        }

        image_number = 1
        total_combinations = (
            len(attributes["eye_color"])
            * len(attributes["hair_color"])
            * len(attributes["hair_style"])
            * len(attributes["accessories"])
        )
        max_images = max_test_images if test_mode else total_combinations

        print(f"Generating {max_images} face images on-the-fly...")
        print(f"Output directory: {output_dir}")
        print(
            f"Attributes: eye_color ({len(attributes['eye_color'])}), hair_color ({len(attributes['hair_color'])}), hair_style ({len(attributes['hair_style'])}), accessories ({len(attributes['accessories'])})"
        )
        print(f"Gender: 50% male, 50% female (randomly assigned)")
        print(f"Total combinations: {total_combinations}")
        print()

        # Generate combinations on-the-fly
        for eye_color in attributes["eye_color"]:
            for hair_color in attributes["hair_color"]:
                for hair_style in attributes["hair_style"]:
                    for accessory in attributes["accessories"]:
                        # Stop if we've reached the test limit
                        if test_mode and image_number > max_test_images:
                            break

                        # Randomly assign gender (50% male, 50% female)
                        gender = random.choice(["male", "female"])

                        # Create name using sequential number
                        name = f"face_{image_number:04d}"
                        image_number += 1

                        output_path = os.path.join(output_dir, f"{name}.png")

                        retry_new_secondary_parameters = 0
                        successful_generation = False
                        final_attr_dict = None

                        while (
                            not successful_generation
                            and retry_new_secondary_parameters < RETRY_NEW_SECONDARY_PARAMETERS_MAX
                        ):
                            # Create description and attributes on-the-fly
                            description, attr_dict = self._create_face_description_and_attributes(
                                eye_color, hair_color, hair_style, accessory, gender
                            )
                            final_attr_dict = attr_dict  # Keep reference for final failure message

                            print(f"Generating {image_number - 1}/{max_images}: {description[:50]}...")
                            print(
                                f"Secondary sample: {retry_new_secondary_parameters + 1}/{RETRY_NEW_SECONDARY_PARAMETERS_MAX}"
                            )

                            success, full_prompt = self.generate_face(description, output_path)
                            if success:
                                # Create metadata entry
                                metadata = {
                                    "image_name": f"{name}.png",
                                    "image_path": output_path,
                                    "description": description,
                                    "full_prompt": full_prompt,
                                    "attributes": attr_dict,
                                }
                                successful_metadata.append(metadata)
                                successful_generation = True
                            else:
                                retry_new_secondary_parameters += 1
                                print(f"Retrying with new secondary parameters...")

                        if not successful_generation and final_attr_dict is not None:
                            print(
                                f"Failed after all attempts for secondary sample {retry_new_secondary_parameters}/{RETRY_NEW_SECONDARY_PARAMETERS_MAX}: {name}.png"
                            )
                            print(
                                f"   Primary Attributes: eye_color: {final_attr_dict['eye_color']}, hair_color: {final_attr_dict['hair_color']}, hair_style: {final_attr_dict['hair_style']}, accessories: {final_attr_dict['accessories']}, gender: {final_attr_dict['gender']}"
                            )
                            sec_attrs = final_attr_dict["secondary_attributes"]
                            print(
                                f"   Secondary Attributes: age_group: {sec_attrs['age_group']}, skin_tone: {sec_attrs['skin_tone']}, face_shape: {sec_attrs['face_shape']}, eyebrow_shape: {sec_attrs['eyebrow_shape']}, lip_shape: {sec_attrs['lip_shape']}, facial_features: {sec_attrs['facial_features']}"
                            )

                            # Log the final failure to JSON file
                            failed_info = {
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                "image_name": f"{name}.png",
                                "output_path": output_path,
                                "primary_attributes": {
                                    "eye_color": final_attr_dict["eye_color"],
                                    "hair_color": final_attr_dict["hair_color"],
                                    "hair_style": final_attr_dict["hair_style"],
                                    "accessories": final_attr_dict["accessories"],
                                    "gender": final_attr_dict["gender"],
                                },
                                "secondary_attributes": final_attr_dict["secondary_attributes"],
                                "retry_attempts": retry_new_secondary_parameters,
                                "max_retry_attempts": RETRY_NEW_SECONDARY_PARAMETERS_MAX,
                                "error_reason": f"Failed after {retry_new_secondary_parameters} secondary parameter retry attempts",
                            }
                            self.failed_generations.append(failed_info)

                        # Add delay to avoid rate limiting
                        time.sleep(1)

                    # Break out of hair_style loop if test limit reached
                    if test_mode and image_number > max_test_images:
                        break

                # Break out of hair_color loop if test limit reached
                if test_mode and image_number > max_test_images:
                    break

            # Break out of eye_color loop if test limit reached
            if test_mode and image_number > max_test_images:
                break

        return successful_metadata

    def _create_face_description_and_attributes(
        self, eye_color: str, hair_color: str, hair_style: str, accessory: str, gender: str
    ) -> Tuple[str, Dict]:
        """Create face description and attributes for a single combination."""
        # Create description
        description_parts = []

        # Add gender
        description_parts.append(gender)

        # Eye color
        if eye_color == "dark_brown":
            description_parts.append("dark brown eyes")
        elif eye_color == "light_brown":
            description_parts.append("light brown eyes")
        elif eye_color == "gray":
            description_parts.append("gray eyes")
        elif eye_color == "blue":
            description_parts.append(
                "very saturated and vivid blue eyes, not ambigous (not gray or green, very obvious blue)"
            )
        elif eye_color == "green":
            description_parts.append("light green eyes")
        else:
            description_parts.append(f"{eye_color} eyes")

        # Hair color
        if hair_color == "light_brown":
            description_parts.append("light brown hair")
        elif hair_color == "gray_white":
            description_parts.append("gray or white hair")
        elif hair_color == "red":
            description_parts.append("vibrant red hair")
        elif hair_color == "blue":
            description_parts.append("blue hair")
        elif hair_color == "blonde":
            description_parts.append("blonde/yellow hair")
        else:
            description_parts.append(f"{hair_color} hair")

        # Hair style
        if hair_style == "shoulder_straight":
            description_parts.append(
                "shoulder length straight hair, clearly above shoulders, with clear gap between hair and shoulders"
            )
        elif hair_style == "shoulder_wavy":
            description_parts.append(
                "shoulder length wavy hair, clearly above shoulders, with clear gap between hair and shoulders"
            )
        elif hair_style == "shoulder_afro":
            description_parts.append("shoulder length afro style very curly, clearly above shoulders")
        elif hair_style == "long_wavy":
            description_parts.append("long wavy hair but not afro style")
        elif hair_style == "long_straight":
            description_parts.append("long straight hair")
        elif hair_style == "buzz_cut":
            description_parts.append("buzz cut with straight hair")

        # Accessories
        if accessory == "eyeglasses_clear":
            description_parts.append("clear eyeglasses, no other accessories")
        elif accessory == "earrings_visible":
            description_parts.append("quite visible earrings, no other accessories")
        elif accessory == "headband":
            description_parts.append("vivid bright colored headband (e.g. pink), no other accessories")
        elif accessory == "scarf_neck_face":
            description_parts.append("scarf around neck, no other accessories")
        elif accessory == "none":
            description_parts.append("no accessories at all")

        # Randomly select one from each category
        random_face_shape = random.choice(SECONDARY_ATTRIBUTES["face_shapes"])
        random_eyebrow_shape = random.choice(SECONDARY_ATTRIBUTES["eyebrow_shapes"])
        random_lip_shape = random.choice(SECONDARY_ATTRIBUTES["lip_shapes"])
        random_facial_feature = random.choice(SECONDARY_ATTRIBUTES["facial_features"])

        # Randomly select secondary attributes
        age_group = random.choice(SECONDARY_ATTRIBUTES["age_groups"])
        skin_tone = random.choice(SECONDARY_ATTRIBUTES["skin_tones"])

        # Add age group and skin tone to description
        if age_group == "young_adult":
            age_description = "young adult"
        elif age_group == "middle_aged":
            age_description = "middle-aged"
        elif age_group == "elderly":
            age_description = "elderly"

        # Add skin tone description
        skin_tone_description = f"{skin_tone} skin type on Fitzpatrick scale"

        # Combine into full description
        description = (
            f"professional headshot, neutral expression, good lighting, monochromatic gray background, {random_face_shape}, {random_eyebrow_shape}, {random_lip_shape}, {random_facial_feature}, {age_description}, {skin_tone_description}, "
            + ", ".join(description_parts)
        )

        # Create attributes dict
        attr_dict = {
            "gender": gender,
            "eye_color": eye_color,
            "hair_color": hair_color,
            "hair_style": hair_style,
            "accessories": accessory,
            "secondary_attributes": {
                "age_group": age_group,
                "skin_tone": skin_tone,
                "face_shape": random_face_shape,
                "eyebrow_shape": random_eyebrow_shape,
                "lip_shape": random_lip_shape,
                "facial_features": random_facial_feature,
            },
        }

        return description, attr_dict


def save_metadata(metadata_list: List[Dict], output_dir: str, attributes_info: Dict):
    """Save metadata to JSON file with attributes summary."""
    metadata_path = os.path.join(output_dir, "face_metadata.json")

    # Create the complete metadata structure
    complete_metadata = {
        "attributes_summary": {
            "primary_attributes": {
                "eye_color": {"options": attributes_info["eye_color"], "count": len(attributes_info["eye_color"])},
                "hair_color": {"options": attributes_info["hair_color"], "count": len(attributes_info["hair_color"])},
                "hair_style": {"options": attributes_info["hair_style"], "count": len(attributes_info["hair_style"])},
                "accessories": {
                    "options": attributes_info["accessories"],
                    "count": len(attributes_info["accessories"]),
                },
            },
            "secondary_attributes": {
                "age_group": {
                    "options": SECONDARY_ATTRIBUTES["age_groups"],
                    "count": len(SECONDARY_ATTRIBUTES["age_groups"]),
                },
                "skin_tone": {
                    "options": SECONDARY_ATTRIBUTES["skin_tones"],
                    "count": len(SECONDARY_ATTRIBUTES["skin_tones"]),
                },
                "face_shape": {
                    "options": SECONDARY_ATTRIBUTES["face_shapes"],
                    "count": len(SECONDARY_ATTRIBUTES["face_shapes"]),
                },
                "eyebrow_shape": {
                    "options": SECONDARY_ATTRIBUTES["eyebrow_shapes"],
                    "count": len(SECONDARY_ATTRIBUTES["eyebrow_shapes"]),
                },
                "lip_shape": {
                    "options": SECONDARY_ATTRIBUTES["lip_shapes"],
                    "count": len(SECONDARY_ATTRIBUTES["lip_shapes"]),
                },
                "facial_features": {
                    "options": SECONDARY_ATTRIBUTES["facial_features"],
                    "count": len(SECONDARY_ATTRIBUTES["facial_features"]),
                },
            },
            "total_combinations": {
                "primary": len(attributes_info["eye_color"])
                * len(attributes_info["hair_color"])
                * len(attributes_info["hair_style"])
                * len(attributes_info["accessories"]),
                "secondary": len(SECONDARY_ATTRIBUTES["age_groups"])
                * len(SECONDARY_ATTRIBUTES["skin_tones"])
                * len(SECONDARY_ATTRIBUTES["face_shapes"])
                * len(SECONDARY_ATTRIBUTES["eyebrow_shapes"])
                * len(SECONDARY_ATTRIBUTES["lip_shapes"])
                * len(SECONDARY_ATTRIBUTES["facial_features"]),
                "total": len(attributes_info["eye_color"])
                * len(attributes_info["hair_color"])
                * len(attributes_info["hair_style"])
                * len(attributes_info["accessories"])
                * len(SECONDARY_ATTRIBUTES["age_groups"])
                * len(SECONDARY_ATTRIBUTES["skin_tones"])
                * len(SECONDARY_ATTRIBUTES["face_shapes"])
                * len(SECONDARY_ATTRIBUTES["eyebrow_shapes"])
                * len(SECONDARY_ATTRIBUTES["lip_shapes"])
                * len(SECONDARY_ATTRIBUTES["facial_features"]),
            },
        },
        "generated_faces": metadata_list,
    }

    with open(metadata_path, "w") as f:
        json.dump(complete_metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate face images with 4 specific attributes and save metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--output-dir", required=True, help="Output directory for generated images")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument(
        "--test-mode", action="store_true", default=False, help="Enable test mode (limit number of images)"
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=10,
        help="Maximum number of images to generate in test mode (default: 10)",
    )

    args = parser.parse_args()

    # Get API key
    try:
        api_key = args.api_key or os.environ["OPENROUTER_API_KEY"]
    except KeyError:
        raise ValueError(
            "Error: Please set your OpenRouter API key using --api-key or OPENROUTER_API_KEY environment variable"
        )

    # Initialize generator
    generator = FaceGenerator(api_key, args.output_dir)

    # Generate faces on-the-fly
    successful_metadata = generator.generate_faces_on_the_fly(
        args.output_dir, test_mode=args.test_mode, max_test_images=args.max_test_images
    )

    # Save metadata
    attributes_info = {
        "eye_color": ["blue", "dark_brown", "gray", "red"],
        "hair_color": ["black", "light_brown", "blonde", "red", "gray_white", "blue"],
        "hair_style": ["shoulder_straight", "shoulder_afro", "long_wavy", "long_straight", "buzz_cut"],
        "accessories": ["none", "eyeglasses_clear", "earrings_visible", "headband", "scarf_neck_face"],
    }
    save_metadata(successful_metadata, args.output_dir, attributes_info)

    # Save failed generations info
    generator.save_failed_generations_to_file(args.output_dir)

    # Summary
    print(f"\nGeneration complete!")
    print(f"Successfully generated: {len(successful_metadata)} faces")
    print(f"Images saved to: {args.output_dir}")
    print(f"Metadata saved to: {os.path.join(args.output_dir, 'face_metadata.json')}")


if __name__ == "__main__":
    main()
