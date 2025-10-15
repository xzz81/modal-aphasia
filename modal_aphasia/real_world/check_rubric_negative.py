#!/usr/bin/env python3
"""
Script to check rubric fulfillment by analyzing text descriptions and images against rubric requirements.
"""

import argparse
import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from PIL import Image


class OpenRouterRubricChecker:
    def __init__(self, api_key: str, model: str = "anthropic/claude-opus-4.1"):
        """
        Initialize the OpenRouter rubric checker.

        Args:
            api_key: OpenRouter API key
            model: Model to use for checking (default: anthropic/claude-opus-4.1)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def call_api(self, messages: List[Dict], max_tokens: int = 2000) -> Optional[str]:
        """
        Call OpenRouter API with messages.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response

        Returns:
            API response text
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3, "top_p": 0.95}

        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=60)
            response.raise_for_status()

            response_json = response.json()
            return response_json["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response text: {e.response.text[:500]}...")
            return None
        except KeyError as e:
            print(f"Unexpected API response format: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for API call, with compression for large images.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded image string
        """
        try:
            # First try to encode the original image
            with open(image_path, "rb") as image_file:
                original_data = image_file.read()
                original_size = len(original_data)

                # If under 4MB, use original
                if original_size <= 4 * 1024 * 1024:
                    return base64.b64encode(original_data).decode("utf-8")

                # Only compress if too large
                print(f"    Compressing image {os.path.basename(image_path)} ({original_size / (1024 * 1024):.1f}MB)")

                # Open and compress the image
                with Image.open(image_path) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "LA", "P"):
                        img = img.convert("RGB")

                    # Start with high quality and reduce if needed
                    quality = 85
                    max_size_bytes = 4 * 1024 * 1024  # 4MB to be safe

                    while True:
                        # Save to bytes buffer
                        buffer = io.BytesIO()
                        img.save(buffer, format="JPEG", quality=quality, optimize=True)
                        buffer.seek(0)

                        # Check size
                        if buffer.tell() <= max_size_bytes:
                            break

                        # Reduce quality and try again
                        quality -= 10
                        if quality < 30:
                            # If still too large, resize the image
                            width, height = img.size
                            img = img.resize((width // 2, height // 2), Image.Resampling.LANCZOS)
                            quality = 85  # Reset quality

                    # Encode to base64
                    buffer.seek(0)
                    compressed_data = buffer.read()
                    print(f"    Compressed to {len(compressed_data) / (1024 * 1024):.1f}MB (quality: {quality})")
                    return base64.b64encode(compressed_data).decode("utf-8")

        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            raise

    def get_image_mime_type(self, image_path: str) -> str:
        """
        Get the correct MIME type for an image file.

        Args:
            image_path: Path to the image file

        Returns:
            MIME type string (e.g., 'image/jpeg', 'image/png')
        """
        # Get file extension
        ext = os.path.splitext(image_path)[1].lower()

        # Map extensions to MIME types
        mime_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }

        return mime_types.get(ext, "image/jpeg")  # Default to JPEG

    def check_text_fulfillment(
        self, text_description: str, requirements: List[str], original_image_path: str = None
    ) -> Dict[str, Any]:
        """
        Check if text description fulfills rubric requirements.

        Args:
            text_description: The text description to check
            requirements: List of rubric requirements
            original_image_path: Path to the original reference image (optional)

        Returns:
            Dictionary with fulfillment analysis
        """
        # Prepare prompt based on whether we have original image
        if original_image_path and os.path.exists(original_image_path):
            # Encode original image
            base64_original = self.encode_image(original_image_path)

            prompt = f"""
You are an expert at checking if text descriptions fulfill specific rubric requirements. I have a text description, the original reference image, and a list of rubric requirements that should be fulfilled.

TEXT DESCRIPTION:
{text_description}

RUBRIC REQUIREMENTS:
{chr(10).join(f"{i + 1}. {req}" for i, req in enumerate(requirements))}

IMPORTANT: The image shown is the ORIGINAL reference poster that the text description should be describing.

PURPOSE OF THE ORIGINAL IMAGE: The original image serves as a reference to help you understand what the rubric requirements mean in practice. However, you should still evaluate the TEXT DESCRIPTION against the rubric requirements, not against the original image.

For each requirement, determine if it is:
- CORRECT: The requirement is correctly mentioned/described in the text description
- INCORRECT: The requirement is either not mentioned at all OR mentioned but with wrong details

EVALUATION APPROACH: Use the original image to understand what the rubric requirements mean, but evaluate the text description against the rubric criteria. The goal is to check if the text description fulfills the requirements.

Return ONLY a JSON object with:
- "text_fulfillment_analysis": detailed analysis of each requirement
- "scores": {{
    "correct": count,
    "incorrect": count
  }}

Example format:
{{
  "text_fulfillment_analysis": [
    {{"requirement": "The poster should NOT have Joker in the image", "status": "CORRECT", "explanation": "Text does not mention Joker, which is correct"}},
    {{"requirement": "Batman should NOT have green hair", "status": "INCORRECT", "explanation": "Text says Batman has green hair, which violates the requirement"}},
    {{"requirement": "The date should NOT be shown as '07.18.08'", "status": "CORRECT", "explanation": "Text does not mention this specific date format"}}
  ],
  "scores": {{
    "correct": 2,
    "incorrect": 1
  }}
}}
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_original}"}},
                    ],
                }
            ]
        else:
            # Fallback to original behavior without original image
            prompt = f"""
You are an expert at checking if text descriptions fulfill specific rubric requirements. I have a text description and a list of rubric requirements that should be fulfilled.

TEXT DESCRIPTION:
{text_description}

RUBRIC REQUIREMENTS:
{chr(10).join(f"{i + 1}. {req}" for i, req in enumerate(requirements))}

For each requirement, determine if it is:
- CORRECT: The requirement is correctly mentioned/described in the text description
- INCORRECT: The requirement is either not mentioned at all OR mentioned but with wrong details

Return ONLY a JSON object with:
- "text_fulfillment_analysis": detailed analysis of each requirement
- "scores": {{
    "correct": count,
    "incorrect": count
  }}

Example format:
{{
  "text_fulfillment_analysis": [
    {{"requirement": "The poster should NOT have Joker in the image", "status": "CORRECT", "explanation": "Text does not mention Joker, which is correct"}},
    {{"requirement": "Batman should NOT have green hair", "status": "INCORRECT", "explanation": "Text says Batman has green hair, which violates the requirement"}}
  ],
  "scores": {{
    "correct": 1,
    "incorrect": 1
  }}
}}
"""

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that checks text descriptions against rubric requirements.",
                },
                {"role": "user", "content": prompt},
            ]

        try:
            response_text = self.call_api(messages, max_tokens=3000)

            if response_text is None:
                return {"text_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}

            # Try to parse JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"Could not parse JSON from response: {response_text[:200]}...")
                return {"text_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}

        except Exception as e:
            print(f"Error checking text fulfillment: {e}")
            return {"text_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}

    def check_image_fulfillment(
        self, image_path: str, requirements: List[str], original_image_path: str = None
    ) -> Dict[str, Any]:
        """
        Check if image fulfills rubric requirements.

        Args:
            image_path: Path to the image file
            requirements: List of rubric requirements
            original_image_path: Path to the original reference image (optional)

        Returns:
            Dictionary with fulfillment analysis
        """
        # Encode generated image
        base64_image = self.encode_image(image_path)

        # Get correct MIME type for the generated image
        generated_image_mime = self.get_image_mime_type(image_path)

        # Prepare prompt based on whether we have original image
        if original_image_path and os.path.exists(original_image_path):
            # Encode original image
            base64_original = self.encode_image(original_image_path)

            prompt = f"""
You are an expert at checking if images fulfill specific rubric requirements. I have a generated image, the original reference image, and a list of rubric requirements that should be fulfilled.

RUBRIC REQUIREMENTS:
{chr(10).join(f"{i + 1}. {req}" for i, req in enumerate(requirements))}

IMPORTANT: The first image is the ORIGINAL reference poster. The second image is the GENERATED poster that should fulfill the rubric requirements.

PURPOSE OF THE ORIGINAL IMAGE: The original image serves as a reference to help you understand what the rubric requirements mean in practice. However, you should still evaluate the GENERATED image against the rubric requirements, not against the original image.

For each requirement, determine if it is:
- CORRECT: The requirement is correctly fulfilled in the generated image
- INCORRECT: The requirement is either not fulfilled OR fulfilled but with wrong details

EVALUATION APPROACH: Use the original image to understand what the rubric requirements mean, but evaluate the generated image against the rubric criteria. The goal is to check if the generated image fulfills the requirements.

IMPORTANT EVALUATION GUIDELINE: Be lenient in your evaluation. If a character or object is not very clear in the generated image but appears to be what it should be according to the requirement, consider it CORRECT. Focus on the overall intent and appearance rather than perfect clarity.

Return ONLY a JSON object with:
- "image_fulfillment_analysis": detailed analysis of each requirement
- "scores": {{
    "correct": count,
    "incorrect": count
  }}

Example format:
{{
  "image_fulfillment_analysis": [
    {{"requirement": "The poster should NOT have Joker in the image", "status": "CORRECT", "explanation": "Joker is not visible in the generated image"}},
    {{"requirement": "Batman should NOT have green hair", "status": "INCORRECT", "explanation": "Batman has green hair in the generated image, violating the requirement"}},
    {{"requirement": "The date should NOT be shown as '07.18.08'", "status": "CORRECT", "explanation": "This specific date format is not visible"}}
  ],
  "scores": {{
    "correct": 2,
    "incorrect": 1
  }}
}}
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_original}"}},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{generated_image_mime};base64,{base64_image}"},
                        },
                    ],
                }
            ]
        else:
            # Fallback to original behavior without original image
            prompt = f"""
You are an expert at checking if images fulfill specific rubric requirements. I have an image and a list of rubric requirements that should be fulfilled.

RUBRIC REQUIREMENTS:
{chr(10).join(f"{i + 1}. {req}" for i, req in enumerate(requirements))}

For each requirement, determine if it is:
- CORRECT: The requirement is correctly fulfilled in the image
- INCORRECT: The requirement is either not fulfilled OR fulfilled but with wrong details

IMPORTANT EVALUATION GUIDELINE: Be lenient in your evaluation. If a character or object is not very clear in the image but appears to be what it should be according to the requirement, consider it CORRECT. Focus on the overall intent and appearance rather than perfect clarity.

Return ONLY a JSON object with:
- "image_fulfillment_analysis": detailed analysis of each requirement
- "scores": {{
    "correct": count,
    "incorrect": count
  }}

Example format:
{{
  "image_fulfillment_analysis": [
    {{"requirement": "The poster should NOT have Joker in the image", "status": "CORRECT", "explanation": "Joker is not visible in the image"}},
    {{"requirement": "Batman should NOT have green hair", "status": "INCORRECT", "explanation": "Batman has green hair in the image, violating the requirement"}}
  ],
  "scores": {{
    "correct": 1,
    "incorrect": 1
  }}
}}
"""

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{generated_image_mime};base64,{base64_image}"},
                        },
                    ],
                }
            ]

        try:
            response_text = self.call_api(messages, max_tokens=3000)

            if response_text is None:
                return {"image_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}

            # Try to parse JSON from response
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                print(f"Could not parse JSON from response: {response_text[:200]}...")
                return {"image_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}

        except Exception as e:
            print(f"Error checking image fulfillment: {e}")
            return {"image_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}


def check_rubric_fulfillment(raw_data_file: str, rubric_file: str, output_file: str = None) -> Dict[str, Any]:
    """
    Check rubric fulfillment for all posters in the raw data.

    Args:
        raw_data_file: Path to the raw data JSON file
        rubric_file: Path to the rubric JSON file
        output_file: Path to save the results (optional)

    Returns:
        Dictionary containing the fulfillment analysis
    """
    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")

    # Load raw data
    with open(raw_data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Load rubric
    with open(rubric_file, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    # Initialize checker
    checker = OpenRouterRubricChecker(api_key)

    print(f"Checking rubric fulfillment for {len(raw_data['evaluations'])} posters...")

    # Process each evaluation
    results = {
        "metadata": {
            "raw_data_file": raw_data_file,
            "rubric_file": rubric_file,
            "total_posters": len(raw_data["evaluations"]),
            "model": checker.model,
        },
        "posters": {},
    }

    for evaluation in raw_data["evaluations"]:
        poster_id = evaluation["poster_id"]
        poster_name = evaluation["poster_name"]

        print(f"  Processing {poster_name} (ID: {poster_id})...")

        # Get rubric requirements for this poster
        if poster_id not in rubric["posters"]:
            print(f"    Warning: No rubric found for poster {poster_id}")
            continue

        poster_rubric = rubric["posters"][poster_id]
        requirements = poster_rubric.get("requirements", [])
        negative_requirements = poster_rubric.get("negative_requirements", [])

        # Combine all requirements (positive and negative)
        all_requirements = requirements + negative_requirements
        total_requirements = len(all_requirements)

        print(
            f"    Checking {len(requirements)} positive + {len(negative_requirements)} negative = {total_requirements} total requirements..."
        )

        # Find original image path (similar to grade_real_world.py)
        script_dir = Path(__file__).parent
        images_dir = os.path.join(script_dir, "data")

        # Construct original image path using poster_id
        # Remove leading zeros from poster_id for file naming
        clean_id = str(int(poster_id)) if poster_id.isdigit() else poster_id
        original_image_path = os.path.join(images_dir, "original_posters", f"{clean_id}.jpg")

        # Check if original image exists, try alternative paths if not
        if not os.path.exists(original_image_path):
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
                print(f"    Warning: Original image not found for {poster_name} (ID: {poster_id})")
                original_image_path = None

        # Check text fulfillment
        text_description = evaluation["text_description"]
        if original_image_path:
            print(f"    Using original image as reference for text evaluation: {os.path.basename(original_image_path)}")
        else:
            print(f"    No original image found, evaluating text without reference")
        text_results = checker.check_text_fulfillment(text_description, all_requirements, original_image_path)

        # Check image fulfillment
        image_path = evaluation["generated_image"]
        # Remove 'real_world/' prefix since we're running from within real_world directory
        if image_path.startswith("real_world/"):
            image_path = image_path[11:]  # Remove 'real_world/' prefix

        # Check if the original .png path exists, if not try .jpg
        original_image_path_for_check = image_path
        if not os.path.exists(image_path):
            # Try alternative extensions
            if image_path.endswith(".png"):
                jpg_path = image_path.replace(".png", ".jpg")
                if os.path.exists(jpg_path):
                    image_path = jpg_path
                else:
                    # Try the original .png path
                    if os.path.exists(original_image_path_for_check):
                        image_path = original_image_path_for_check
            elif image_path.endswith(".jpg"):
                png_path = image_path.replace(".jpg", ".png")
                if os.path.exists(png_path):
                    image_path = png_path
                else:
                    # Try the original .jpg path
                    if os.path.exists(original_image_path_for_check):
                        image_path = original_image_path_for_check

        if not os.path.exists(image_path):
            print(f"    Warning: Generated image file not found: {original_image_path_for_check}")
            # Create empty results for missing image
            image_results = {"image_fulfillment_analysis": [], "scores": {"correct": 0, "incorrect": 0}}
        else:
            if original_image_path:
                print(f"    Using original image as reference: {os.path.basename(original_image_path)}")
            else:
                print(f"    No original image found, evaluating without reference")
            print(f"    Using generated image: {os.path.basename(image_path)}")
            image_results = checker.check_image_fulfillment(image_path, all_requirements, original_image_path)

        # Calculate percentages for this poster
        text_correct_pct = (
            (text_results["scores"]["correct"] / total_requirements) * 100 if total_requirements > 0 else 0
        )
        text_incorrect_pct = (
            (text_results["scores"]["incorrect"] / total_requirements) * 100 if total_requirements > 0 else 0
        )

        image_correct_pct = (
            (image_results["scores"]["correct"] / total_requirements) * 100 if total_requirements > 0 else 0
        )
        image_incorrect_pct = (
            (image_results["scores"]["incorrect"] / total_requirements) * 100 if total_requirements > 0 else 0
        )

        # Calculate breakdown counts for positive vs negative requirements
        text_positive_correct = 0
        text_positive_incorrect = 0
        text_negative_correct = 0
        text_negative_incorrect = 0

        image_positive_correct = 0
        image_positive_incorrect = 0
        image_negative_correct = 0
        image_negative_incorrect = 0

        # Process text fulfillment analysis for counts
        for i, analysis in enumerate(text_results.get("text_fulfillment_analysis", [])):
            if i < len(all_requirements):
                requirement = all_requirements[i]
                status = analysis.get("status", "")

                # Determine if this is a positive or negative requirement
                is_negative = i >= len(requirements)  # Negative requirements come after positive ones

                if is_negative:
                    # For negative requirements
                    if status == "CORRECT":
                        text_negative_correct += 1
                    else:  # INCORRECT
                        text_negative_incorrect += 1
                else:
                    # For positive requirements
                    if status == "CORRECT":
                        text_positive_correct += 1
                    else:  # INCORRECT
                        text_positive_incorrect += 1

        # Process image fulfillment analysis for counts
        for i, analysis in enumerate(image_results.get("image_fulfillment_analysis", [])):
            if i < len(all_requirements):
                requirement = all_requirements[i]
                status = analysis.get("status", "")

                # Determine if this is a positive or negative requirement
                is_negative = i >= len(requirements)  # Negative requirements come after positive ones

                if is_negative:
                    # For negative requirements
                    if status == "CORRECT":
                        image_negative_correct += 1
                    else:  # INCORRECT
                        image_negative_incorrect += 1
                else:
                    # For positive requirements
                    if status == "CORRECT":
                        image_positive_correct += 1
                    else:  # INCORRECT
                        image_positive_incorrect += 1

        # Store results with percentages
        results["posters"][poster_id] = {
            "poster_name": poster_name,
            "poster_id": poster_id,
            "text_description": text_description,
            "generated_image": image_path,
            "original_image": original_image_path,
            "positive_requirements": requirements,
            "negative_requirements": negative_requirements,
            "text_fulfillment": {
                "text_fulfillment_analysis": text_results["text_fulfillment_analysis"],
                "scores": {
                    "correct": text_results["scores"]["correct"],
                    "correct_pct": round(text_correct_pct, 1),
                    "incorrect": text_results["scores"]["incorrect"],
                    "incorrect_pct": round(text_incorrect_pct, 1),
                },
            },
            "image_fulfillment": {
                "image_fulfillment_analysis": image_results["image_fulfillment_analysis"],
                "scores": {
                    "correct": image_results["scores"]["correct"],
                    "correct_pct": round(image_correct_pct, 1),
                    "incorrect": image_results["scores"]["incorrect"],
                    "incorrect_pct": round(image_incorrect_pct, 1),
                },
            },
            "summary": {
                "text_correct": text_results["scores"]["correct"],
                "text_correct_pct": round(text_correct_pct, 1),
                "text_incorrect": text_results["scores"]["incorrect"],
                "text_incorrect_pct": round(text_incorrect_pct, 1),
                "image_correct": image_results["scores"]["correct"],
                "image_correct_pct": round(image_correct_pct, 1),
                "image_incorrect": image_results["scores"]["incorrect"],
                "image_incorrect_pct": round(image_incorrect_pct, 1),
                "total_requirements": total_requirements,
                "positive_negative_breakdown": {
                    "text": {
                        "positive_correct": text_positive_correct,
                        "positive_incorrect": text_positive_incorrect,
                        "negative_correct": text_negative_correct,
                        "negative_incorrect": text_negative_incorrect,
                    },
                    "image": {
                        "positive_correct": image_positive_correct,
                        "positive_incorrect": image_positive_incorrect,
                        "negative_correct": image_negative_correct,
                        "negative_incorrect": image_negative_incorrect,
                    },
                    "rubric_counts": {
                        "positive_rubric": len(requirements),
                        "negative_rubric": len(negative_requirements),
                    },
                },
            },
        }

        # Print summary with percentages
        text_scores = text_results["scores"]
        image_scores = image_results["scores"]

        print(
            f"    Text: {text_scores['correct']} correct ({text_correct_pct:.1f}%), {text_scores['incorrect']} incorrect ({text_incorrect_pct:.1f}%)"
        )
        print(
            f"    Image: {image_scores['correct']} correct ({image_correct_pct:.1f}%), {image_scores['incorrect']} incorrect ({image_incorrect_pct:.1f}%)"
        )
        print(f"    Total Requirements: {total_requirements}")
        print(f"    Positive/Negative Breakdown:")
        print(
            f"      Text: Positive {text_positive_correct}/{len(requirements)} correct, Negative {text_negative_correct}/{len(negative_requirements)} correct"
        )
        print(
            f"      Image: Positive {image_positive_correct}/{len(requirements)} correct, Negative {image_negative_correct}/{len(negative_requirements)} correct"
        )

    # Save results if output file is provided
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Check rubric fulfillment for text descriptions and images")
    parser.add_argument("raw_data_file", help="Path to the raw data JSON file with text descriptions and image paths")
    parser.add_argument("rubric_file", help="Path to the rubric JSON file")
    parser.add_argument(
        "-o", "--output", help="Output file path for the results (default: rubric_checked_<raw_data_name>.json)"
    )

    args = parser.parse_args()

    # Generate default output path if not provided
    if not args.output:
        # Get the directory of the raw data file
        raw_data_dir = os.path.dirname(args.raw_data_file)
        base_name = os.path.splitext(os.path.basename(args.raw_data_file))[0]
        args.output = os.path.join(raw_data_dir, f"rubric_checked_{base_name}.json")

    # Check rubric fulfillment
    results = check_rubric_fulfillment(args.raw_data_file, args.rubric_file, args.output)

    # Print final summary
    print("\nFinal Summary:")
    total_text_correct = 0
    total_text_incorrect = 0
    total_image_correct = 0
    total_image_incorrect = 0

    for poster_id, poster_data in results["posters"].items():
        summary = poster_data["summary"]
        total_text_correct += summary["text_correct"]
        total_text_incorrect += summary["text_incorrect"]
        total_image_correct += summary["image_correct"]
        total_image_incorrect += summary["image_incorrect"]

    print(f"Text Descriptions:")
    print(f"  Correct: {total_text_correct}")
    print(f"  Incorrect: {total_text_incorrect}")
    print(f"Images:")
    print(f"  Correct: {total_image_correct}")
    print(f"  Incorrect: {total_image_incorrect}")


if __name__ == "__main__":
    main()
