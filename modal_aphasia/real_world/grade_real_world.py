#!/usr/bin/env python3
"""
Script to grade real-world poster descriptions using OpenRouter API.
Supports both image-to-text and image-to-image comparisons.
Set ignore_non_title_text to True to ignore differences in non-title text (credits, dates, websites, etc.).
"""

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv


class OpenRouterGrader:
    def __init__(self, api_key: str, model: str = "anthropic/claude-opus-4.1"):
        """
        Initialize the OpenRouter grader.

        Args:
            api_key: OpenRouter API key
            model: Model to use for grading (default: claude-3.5-sonnet)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64 string. Images should be pre-compressed."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                image_size = len(image_data)

                # Check if image is under 4MB (should be if pre-compressed)
                if image_size > 4 * 1024 * 1024:
                    print(
                        f"Warning: Image {image_path} is {image_size / (1024 * 1024):.1f}MB. Consider pre-compressing."
                    )

                return base64.b64encode(image_data).decode("utf-8")

        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def call_api(self, messages: List[Dict], max_tokens: int = 1000) -> Optional[str]:
        """
        Call OpenRouter API with messages.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response

        Returns:
            API response text or None if error
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.95,
        }

        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response text: {e.response.text[:500]}...")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            print(f"Full response: {result}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def grade_image_to_text(
        self, original_image_path: str, text_description: str, ignore_non_title_text: bool = False
    ) -> Dict[str, Any]:
        """
        Grade how well the text description matches the original image.

        Args:
            original_image_path: Path to the original poster image
            text_description: Text description to evaluate
            ignore_non_title_text: If True, ignore differences in non-title text (taglines, credits, dates, websites, etc.)

        Returns:
            Dictionary with textual analysis and scores
        """
        # Encode the original image
        original_image_b64 = self.encode_image(original_image_path)

        # Add instruction about ignoring non-title text if requested
        ignore_text_instruction = ""
        if ignore_non_title_text:
            ignore_text_instruction = """
IMPORTANT: Completely ignore ALL non-title text elements in ALL scoring categories. Do NOT count them in any category:
- Credits (actor names, director names, producer names)
- Subtitles or taglines
- Release dates
- Website URLs
- Studio names
- Rating information
- "COMING SOON"
- Actor names
- Typos or grammar errors
- Any other text that is not the main movie title

These text elements should be completely excluded from present_and_accurate, present_not_accurate, and not_present_in_original counts. If descrption mentiones these text elements, ignore them.

Focus only on visual elements, characters, objects, and the main movie title."""

        prompt = f"""Here is the original US version of a movie poster. I have a description of the poster and I want to know how accurate is it.

IMPORTANT: Pay very close attention to the image and be lenient with positioning and detail variations. Only count major discrepancies as inaccurate.

SYSTEMATIC CATEGORIZATION RULES:
- DETAIL CHECKING: If checking if a certain element is present in the image, consider it CORRECT even if that element is not clearly visible or sharp, as long as it is still present in the image.

CRITICAL POSITIONING ATTENTION - DOUBLE CHECK WITH ORIGINAL:
- Pay EXTREMELY careful attention to positioning details, especially left/right positioning
- Text and images can be mirrored or swapped left/right between description and actual image
- ALWAYS double-check positioning by examining the original image carefully
- If the description mentions an element is on the left or right side, verify this positioning carefully
- Left/right positioning errors should be counted as inaccuracies
- Be clear wether its viewer's left or right side, not just left or right.

CATEGORIZATION SYSTEM - Process each element from the description systematically:

1) PRESENT AND ACCURATE: Elements that are present and correctly described (position, color, details, etc.)
   - Character wears a hat as described
   - Character is positioned in the general area described (e.g., "center left" vs "center" is still accurate)
   - Colors match (e.g., blue shirt, red car)
   - Objects are in correct locations

2) PRESENT BUT INACCURATE: ONLY count as inaccurate if there are MAJOR errors:
   - MAJOR positioning errors (e.g., character on left side when description says right side)
   - WRONG colors (e.g., red instead of blue, not light blue vs dark blue)
   - COMPLETELY wrong elements (e.g., different character entirely)
   - Do NOT count: minor positioning variations, slight color tone differences, minor detail differences, head orientations that are not completely wrong

3) NOT PRESENT IN ORIGINAL: Elements mentioned in description that do NOT exist in the original image
   - Description says a character or object is present but it's completely missing
   - This is when the description mentions something that is not in the original

4) NOT PRESENT IN DESCRIPTION: Major elements in the poster that are clearly visible but NOT mentioned in the description
   - Important characters, objects, background elements that the description missed
   - These are elements that exist but weren't described

{ignore_text_instruction}

SYSTEMATIC PROCESSING:
1. Read the description carefully
2. Examine the image systematically
3. For each element mentioned in the description, categorize it into one of the first 3 categories
4. For each major element visible in the image, check if it's mentioned in the description
5. Count each element only once and place it in the most appropriate category
6. Ensure your counts are accurate and consistent
7. CRITICAL: Double-check all positioning details, especially left/right, by comparing directly with the original image

DESCRIPTION: {text_description}

Please respond in this exact JSON format (ensure all newlines in textual_analysis are escaped as \\n):
{{
    "textual_analysis": "Your detailed analysis here...",
    "scores": {{
        "present_and_accurate": <count>,
        "present_not_accurate": <count>,
        "not_present_in_original": <count>,
        "not_present_in_description": <count>
    }}
}}"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_image_b64}"}},
                ],
            }
        ]

        response = self.call_api(messages, max_tokens=2500)
        if response:
            try:
                # Extract JSON from response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                json_str = response[start_idx:end_idx]

                # Fix common JSON issues
                # Replace literal newlines with escaped newlines in string values
                import re

                # This regex finds string values and escapes newlines within them
                json_str = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: m.group(1).replace("\n", "\\n"), json_str)

                parsed = json.loads(json_str)

                # Ensure the parsed response has the expected structure
                if isinstance(parsed, dict) and "textual_analysis" in parsed and "scores" in parsed:
                    return parsed
                else:
                    print(
                        f"API response missing expected fields: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}"
                    )
                    return {
                        "textual_analysis": f"Invalid response structure: {response[:500]}...",
                        "scores": {
                            "present_and_accurate": 0.0,
                            "present_not_accurate": 0.0,
                            "not_present_in_original": 0.0,
                            "not_present_in_description": 0.0,
                        },
                    }
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse API response as JSON: {e}")
                print(f"Full API response: {response}")
                return {
                    "textual_analysis": f"JSON Parse Error: {str(e)}",
                    "scores": {
                        "present_and_accurate": 0.0,
                        "present_not_accurate": 0.0,
                        "not_present_in_original": 0.0,
                        "not_present_in_description": 0.0,
                    },
                }

        return {
            "textual_analysis": "API call failed",
            "scores": {
                "present_and_accurate": 0.0,
                "present_not_accurate": 0.0,
                "not_present_in_original": 0.0,
                "not_present_in_description": 0.0,
            },
        }

    def grade_image_to_image(
        self, original_image_path: str, generated_image_path: str, ignore_non_title_text: bool = False
    ) -> Dict[str, Any]:
        """
        Grade how well the generated image matches the original image.

        Args:
            original_image_path: Path to the original poster image
            generated_image_path: Path to the generated poster image
            ignore_non_title_text: If True, ignore differences in non-title text (taglines, credits, dates, websites, etc.)

        Returns:
            Dictionary with textual analysis and scores
        """
        # Encode both images
        original_image_b64 = self.encode_image(original_image_path)
        generated_image_b64 = self.encode_image(generated_image_path)

        # Add instruction about ignoring non-title text if requested
        ignore_text_instruction = ""
        if ignore_non_title_text:
            ignore_text_instruction = """
IMPORTANT: Completely ignore ALL non-title text elements in ALL scoring categories. Do NOT count them in any category:
- Credits (actor names, director names, producer names)
- Subtitles or taglines
- Release dates
- Website URLs
- Studio names
- Rating information
- "COMING SOON"
- Actor names
- Typos or grammar errors
- Any other text that is not the main movie title

These text elements should be completely excluded from present_and_accurate, present_not_accurate, not_present_in_replication, and not_present_in_original counts.

Focus only on visual elements, characters, objects, and the main movie title."""

        prompt = f"""Here is the original US version of a movie poster (first image) and a replication of the poster (second image). I want to know how accurate the replication is compared to the original.

SYSTEMATIC COMPARISON RULES:
- Ignore aesthetic details such as slightly different color tones (e.g., light blue vs. dark blue), but do register difference in colors such as blue and white.
- Ignore white wall and black frame around the replication poster.
- DETAIL CHECKING: If checking if a certain element is present in the image, consider it CORRECT even if that element is not clearly visible or sharp, as long as it is still present in the image.

CRITICAL POSITIONING ATTENTION - DOUBLE CHECK WITH ORIGINAL:
- Pay EXTREMELY careful attention to positioning details, especially left/right positioning
- Text and images can be mirrored or swapped left/right between original and generated versions
- ALWAYS double-check positioning by comparing directly with the original image
- If the description mentions an element is on the left or right side, verify this positioning carefully
- Left/right positioning errors should be counted as inaccuracies
- Be clear wether its viewer's left or right side, not just left or right.

CRITICAL: You MUST ignore the following types of differences - these are NOT considered inaccuracies:
- Less defined or less sharp details
- More faded or less vibrant colors
- Slightly different textures or gradients
- Minor variations in contrast or definition
- "Slightly" anything - if something is present but just less detailed, that's fine
- Slight positioning differences (e.g., "slightly different position")
- Detail level variations (e.g., "detail level varies somewhat")

ONLY count as "present but inaccurate" if:
- Major positioning errors (e.g., character on left instead of right)
- Wrong colors (e.g., red instead of blue, not light blue vs dark blue)
- Completely wrong elements (e.g., different character entirely)
- Major layout errors (e.g., upside down, completely wrong composition)

Focus on:
- Presence/absence of major elements (characters, objects, text)
- Basic positioning and relationships between elements
- Overall color scheme (e.g., blue vs red, not light blue vs dark blue)
- Major layout and composition elements

{ignore_text_instruction}

SYSTEMATIC CATEGORIZATION - Process each element systematically:

1) PRESENT AND ACCURATE: Elements that match well between original and replication
   - Characters, objects, details, colors, and text that are correctly replicated
   - Elements that are present and correctly positioned, even if less detailed or sharp
   - Overall layout and composition that matches

2) PRESENT BUT INACCURATELY REPLICATED: ONLY major errors
   - Wrong colors (e.g., red instead of blue, not light blue vs dark blue)
   - Major positioning errors (e.g., character on left instead of right)
   - Completely wrong elements (e.g., different character entirely)
   - Major layout errors (e.g., upside down, completely wrong composition)

3) NOT PRESENT IN REPLICATION: Elements missing from the replication
   - Characters or objects present in original but missing from replication
   - Count each missing element separately (if 3 characters missing = 3 errors)
   - These are elements that exist in original but not in generated image

4) NOT PRESENT IN ORIGINAL: Extra elements in replication
   - Characters or objects in replication that don't exist in original
   - Count each extra element separately (if 2 extra characters = 2 errors)
   - These are elements that don't exist in original but appear in generated image

SYSTEMATIC PROCESSING:
1. Examine both images carefully
2. For each major element in the original, check if it's present and accurate in the replication
3. For each major element in the replication, verify it exists in the original
4. Categorize each element into the most appropriate category
5. Count each element only once
6. Ensure your counts are accurate and consistent
7. CRITICAL: Double-check all positioning details, especially left/right, by comparing directly with the original image

Please respond in this exact JSON format (ensure all newlines in textual_analysis are escaped as \\n):
{{
    "textual_analysis": "Your detailed analysis here...",
    "scores": {{
        "present_and_accurate": <count>,
        "present_not_accurate": <count>,
        "not_present_in_replication": <count>,
        "not_present_in_original": <count>
    }}
}}"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{original_image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{generated_image_b64}"}},
                ],
            }
        ]

        response = self.call_api(messages, max_tokens=2500)
        if response:
            try:
                # Extract JSON from response
                start_idx = response.find("{")
                end_idx = response.rfind("}") + 1
                json_str = response[start_idx:end_idx]

                # Fix common JSON issues
                # Replace literal newlines with escaped newlines in string values
                import re

                # This regex finds string values and escapes newlines within them
                json_str = re.sub(r'("(?:[^"\\]|\\.)*")', lambda m: m.group(1).replace("\n", "\\n"), json_str)

                parsed = json.loads(json_str)

                # Ensure the parsed response has the expected structure
                if isinstance(parsed, dict) and "textual_analysis" in parsed and "scores" in parsed:
                    return parsed
                else:
                    print(
                        f"API response missing expected fields: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}"
                    )
                    return {
                        "textual_analysis": f"Invalid response structure: {response[:500]}...",
                        "scores": {
                            "present_and_accurate": 0.0,
                            "present_not_accurate": 0.0,
                            "not_present_in_replication": 0.0,
                            "not_present_in_original": 0.0,
                        },
                    }
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Failed to parse API response as JSON: {e}")
                print(f"Full API response: {response}")
                return {
                    "textual_analysis": f"JSON Parse Error: {str(e)}",
                    "scores": {
                        "present_and_accurate": 0.0,
                        "present_not_accurate": 0.0,
                        "not_present_in_replication": 0.0,
                        "not_present_in_original": 0.0,
                    },
                }

        return {
            "textual_analysis": "API call failed",
            "scores": {
                "present_and_accurate": 0.0,
                "present_not_accurate": 0.0,
                "not_present_in_replication": 0.0,
                "not_present_in_original": 0.0,
            },
        }

    def process_evaluation_file(
        self,
        json_file_path: str,
        images_dir: str,
        output_file_path: str,
        grading_mode: str = "both",
        ignore_non_title_text: bool = False,
    ):
        """
        Process the evaluation JSON file and grade all entries.

        Args:
            json_file_path: Path to the raw_data.json file
            images_dir: Directory containing the images
            output_file_path: Path to save the graded JSON file
            grading_mode: What to grade - "text_to_image", "image_to_image", or "both"
            ignore_non_title_text: If True, ignore differences in non-title text (credits, dates, websites, etc.)
        """
        # Load the evaluation data
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # Try to load existing graded results if they exist
        existing_data = {}
        if os.path.exists(output_file_path):
            try:
                with open(output_file_path, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"Could not load existing results: {e}")
                existing_data = {}

        # Merge existing evaluations if they exist
        if existing_data and "evaluations" in existing_data:
            existing_evaluations = {eval.get("poster_id", ""): eval for eval in existing_data["evaluations"]}
        else:
            existing_evaluations = {}

        # Update metadata to include the grading model
        if "metadata" not in data:
            data["metadata"] = {}
        data["metadata"]["judging_model"] = self.model
        data["metadata"]["grading_mode"] = grading_mode
        data["metadata"]["ignore_non_title_text"] = ignore_non_title_text

        # Process each evaluation
        for i, evaluation in enumerate(data.get("evaluations", [])):
            poster_name = evaluation.get("poster_name", f"poster_{i}")
            poster_id = evaluation.get("poster_id", str(i + 1))
            print(f"Processing {poster_name} (ID: {poster_id})...")

            # Check if we have existing results for this poster
            existing_eval = existing_evaluations.get(poster_id, {})

            # Preserve existing grading results based on grading mode
            if existing_eval:
                # Preserve existing comparison results that we're not regenerating this run
                if grading_mode == "text_to_image" and "image_to_image_comparison" in existing_eval:
                    evaluation["image_to_image_comparison"] = existing_eval["image_to_image_comparison"]
                elif grading_mode == "image_to_image" and "image_to_text_comparison" in existing_eval:
                    evaluation["image_to_text_comparison"] = existing_eval["image_to_text_comparison"]
                elif grading_mode == "both":
                    pass

            # Construct original image path using poster_id
            # Remove leading zeros from poster_id for file naming
            clean_id = str(int(poster_id)) if poster_id.isdigit() else poster_id

            original_image_path = os.path.join(images_dir, "original_posters", f"{clean_id}.jpeg")

            # Get generated image path from JSON data
            generated_image_path = evaluation.get("generated_image", "")

            # If generated_image path is relative, make it relative to the script's parent directory
            if generated_image_path and not os.path.isabs(generated_image_path):
                # Check if it's already relative to images_dir
                if generated_image_path.startswith("generated_posters/"):
                    generated_image_path = os.path.join(images_dir, generated_image_path)
                else:
                    # It's relative to project root
                    script_dir = Path(__file__).parent
                    project_root = script_dir.parent
                    generated_image_path = os.path.join(project_root, generated_image_path)

            # Check if original image exists
            if not os.path.exists(original_image_path):
                # Try different extensions and paths
                possible_paths = [
                    os.path.join(images_dir, "original_posters", f"{clean_id}.jpg"),
                    os.path.join(images_dir, "original_posters", f"{clean_id}.jpeg"),
                    os.path.join(images_dir, "original_posters", f"{clean_id}.png"),
                    os.path.join(images_dir, "original_posters", f"{poster_id}.jpg"),
                    os.path.join(images_dir, "original_posters", f"{poster_id}.jpeg"),
                    os.path.join(images_dir, "original_posters", f"{poster_id}.png"),
                    os.path.join(images_dir, f"{clean_id}.jpg"),
                    os.path.join(images_dir, f"{clean_id}.jpeg"),
                    os.path.join(images_dir, f"{clean_id}.png"),
                ]

                for alt_path in possible_paths:
                    if os.path.exists(alt_path):
                        original_image_path = alt_path
                        break
                else:
                    print(f"Warning: Original image not found for {poster_name} (ID: {poster_id})")
                    print(f"  Looked for: {original_image_path}")
                    continue

            # Grade image-to-text comparison
            should_grade_text_to_image = (
                grading_mode in ["text_to_image", "both"]
                and "text_description" in evaluation
                and "image_to_text_comparison" not in evaluation
            )

            if should_grade_text_to_image:
                print(f"  Grading image-to-text for {poster_name}...")
                image_to_text_result = self.grade_image_to_text(
                    original_image_path, evaluation["text_description"], ignore_non_title_text
                )
                evaluation["image_to_text_comparison"] = image_to_text_result
            elif "image_to_text_comparison" in evaluation and grading_mode in ["text_to_image", "both"]:
                print(f"  Reusing existing image-to-text results for {poster_name}...")
            elif grading_mode == "text_to_image" and "text_description" not in evaluation:
                print(f"  No text description available for {poster_name}, skipping text-to-image grading...")

            # Grade image-to-image comparison if generated image exists
            should_grade_image_to_image = (
                grading_mode in ["image_to_image", "both"] and "image_to_image_comparison" not in evaluation
            )

            if should_grade_image_to_image and generated_image_path and os.path.exists(generated_image_path):
                print(f"  Grading image-to-image for {poster_name}...")
                image_to_image_result = self.grade_image_to_image(
                    original_image_path, generated_image_path, ignore_non_title_text
                )
                evaluation["image_to_image_comparison"] = image_to_image_result
            elif should_grade_image_to_image and generated_image_path:
                # Try alternative paths for generated image if the JSON path doesn't exist
                alt_generated_paths = [
                    os.path.join(images_dir, "generated_posters", "gpt5", f"{clean_id}.jpg"),
                    os.path.join(images_dir, "generated_posters", "gpt5", f"{poster_id}.jpg"),
                    os.path.join(images_dir, "generated_posters", f"{clean_id}.jpg"),
                    os.path.join(images_dir, "generated_posters", f"{poster_id}.jpg"),
                    os.path.join(images_dir, f"{clean_id}.jpg"),
                    os.path.join(images_dir, f"{poster_id}.jpg"),
                ]

                generated_found = False
                for alt_gen_path in alt_generated_paths:
                    if os.path.exists(alt_gen_path):
                        generated_image_path = alt_gen_path
                        generated_found = True
                        print(f"  Grading image-to-image for {poster_name}...")
                        image_to_image_result = self.grade_image_to_image(
                            original_image_path, generated_image_path, ignore_non_title_text
                        )
                        evaluation["image_to_image_comparison"] = image_to_image_result
                        break

                if not generated_found:
                    print(f"  Warning: Generated image not found for {poster_name} (ID: {poster_id})")
                    print(f"  Specified path: {evaluation.get('generated_image', 'None')}")
                    print(f"  Tried alternatives: {alt_generated_paths}")
            elif "image_to_image_comparison" in evaluation and grading_mode in ["image_to_image", "both"]:
                print(f"  Reusing existing image-to-image results for {poster_name}...")
            elif grading_mode == "image_to_image" and not generated_image_path:
                print(f"  No generated image path specified for {poster_name}, skipping image-to-image grading...")

        # Save the updated data
        with open(output_file_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Grading complete! Results saved to {output_file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Grade real-world poster descriptions using OpenRouter API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Grade both text-to-image and image-to-image comparisons (default)
  python grade_real_world.py data/results/gpt4o/raw_data.json anthropic/claude-3.5-sonnet

  # Grade only text descriptions
  python grade_real_world.py --grade-text-to-image data/results/gpt4o/raw_data.json anthropic/claude-3.5-sonnet

  # Grade only generated images
  python grade_real_world.py --grade-image-to-image data/results/gpt4o/raw_data.json anthropic/claude-3.5-sonnet

  # Compare different models on the same data
  python grade_real_world.py data/results/gpt4o/raw_data.json anthropic/claude-3.5-sonnet
  python grade_real_world.py data/results/gpt4o/raw_data.json openai/gpt-4o
        """,
    )
    parser.add_argument(
        "--json-file",
        required=False,
        default="data/raw_data.json",
        help="Path to raw_data.json file (default: data/raw_data.json)",
    )
    parser.add_argument("--images-dir", default="data/", help="Directory containing poster images (default: data/)")
    parser.add_argument(
        "--output-file", help="Output file path (default: same directory as input file with 'graded.json' name)"
    )
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model to use for grading (default: anthropic/claude-3.5-sonnet)",
    )

    # Grading mode options
    grading_group = parser.add_mutually_exclusive_group()
    grading_group.add_argument(
        "--grade-text-to-image", action="store_true", help="Grade only text-to-image comparisons"
    )
    grading_group.add_argument(
        "--grade-image-to-image", action="store_true", help="Grade only image-to-image comparisons"
    )
    grading_group.add_argument(
        "--grade-both",
        action="store_true",
        default=True,
        help="Grade both text-to-image and image-to-image comparisons (default)",
    )

    # Text evaluation options
    parser.add_argument(
        "--ignore-non-title-text",
        action="store_true",
        help="Ignore differences in non-title text (credits, dates, websites, etc.)",
    )

    # Add positional arguments for convenience
    parser.add_argument(
        "json_file_pos", nargs="?", help="Path to raw_data.json file (positional argument, overrides --json-file)"
    )
    parser.add_argument(
        "model_pos", nargs="?", help="Model to use for grading (positional argument, overrides --model)"
    )

    args = parser.parse_args()

    # Handle positional arguments (they override named arguments)
    json_file = args.json_file_pos if args.json_file_pos else args.json_file
    model = args.model_pos if args.model_pos else args.model

    # Determine grading mode
    if args.grade_text_to_image:
        grading_mode = "text_to_image"
    elif args.grade_image_to_image:
        grading_mode = "image_to_image"
    else:
        grading_mode = "both"

    # Get ignore_non_title_text setting
    ignore_non_title_text = args.ignore_non_title_text

    # Try to load .env file from parent directory
    script_dir = Path(__file__).parent
    parent_env_file = script_dir.parent / ".env"
    if parent_env_file.exists():
        load_dotenv(parent_env_file)
        print(f"Loaded environment from: {parent_env_file}")

    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: Please provide OpenRouter API key via --api-key or OPENROUTER_API_KEY environment variable")
        print(f"Looked for .env file at: {parent_env_file}")
        return

    # Convert relative paths to absolute paths based on script location
    script_dir = Path(__file__).parent
    json_file_path = script_dir / json_file if not Path(json_file).is_absolute() else Path(json_file)
    images_dir = script_dir / args.images_dir if not Path(args.images_dir).is_absolute() else Path(args.images_dir)

    # Auto-generate output file path if not specified
    if args.output_file:
        output_file_path = (
            script_dir / args.output_file if not Path(args.output_file).is_absolute() else Path(args.output_file)
        )
    else:
        # Create model-specific folder and place graded.json there
        # Convert model name to safe folder name (replace / with -)
        safe_model_name = model.replace("/", "-").replace("\\", "-")

        # Use different prefix based on grading mode
        if ignore_non_title_text:
            model_folder = json_file_path.parent / f"grade_visual_{safe_model_name}"
        else:
            model_folder = json_file_path.parent / f"graded_{safe_model_name}"

        output_file_path = model_folder / "graded.json"

    # Check if input file exists
    if not json_file_path.exists():
        print(f"Error: Input file {json_file_path} does not exist")
        return

    # Create output directory if it doesn't exist
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading from: {json_file_path}")
    print(f"Images directory: {images_dir}")
    print(f"Output will be saved to: {output_file_path}")
    print(f"Using model: {model}")
    print(f"Grading mode: {grading_mode}")
    print(f"Ignore non-title text: {ignore_non_title_text}")

    # Initialize grader and process
    grader = OpenRouterGrader(api_key, model)
    grader.process_evaluation_file(
        str(json_file_path), str(images_dir), str(output_file_path), grading_mode, ignore_non_title_text
    )


if __name__ == "__main__":
    main()
