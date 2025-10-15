#!/usr/bin/env python3
"""
Script to deduplicate rubric requirements by combining similar statements and removing duplicates.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


class OpenRouterDeduplicator:
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        """
        Initialize the OpenRouter deduplicator.

        Args:
            api_key: OpenRouter API key
            model: Model to use for deduplication (default: claude-3.5-sonnet)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def call_api(self, messages: List[Dict], max_tokens: int = 1000) -> str:
        """
        Call OpenRouter API with messages.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens for response

        Returns:
            API response text
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.1}

        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def filter_text_requirements(self, requirements: List[str], requirement_type: str = "requirements") -> List[str]:
        """
        Filter out text-related requirements using LLM.

        Args:
            requirements: List of requirements to filter
            requirement_type: Type of requirements being filtered (for prompt clarity)

        Returns:
            Filtered list containing only visual requirements
        """
        if not requirements:
            return requirements

        prompt = f"""
You are an expert at filtering rubric requirements. I have a list of {requirement_type} that need to be filtered to remove text-related elements.

Your task is to:
1. REMOVE all requirements about credits, actors, directors, taglines, dates, websites, studio names, etc.
2. KEEP only requirements about visual elements: characters, objects, colors, positions, and the main title
3. Return only the visual requirements

{requirement_type.upper()} TO FILTER:
{chr(10).join(f"{i + 1}. {req}" for i, req in enumerate(requirements))}

CRITICAL RULES:
- REMOVE: Actor names, director credits, producer information, taglines, slogans, release dates, studio names, production companies, billing blocks, cast lists, website URLs, ratings, etc.
- KEEP: Character positions, appearances, clothing, object placements, colors, sizes, background elements, lighting, atmosphere, and the main title

EXAMPLES OF WHAT TO REMOVE:
- "Actor names are listed at bottom"
- "Director credit is shown"
- "Release date is displayed"
- "Studio logo is present"
- "Tagline appears at top"
- "Billing block contains cast information"
- "The date should NOT be shown as '07.18.08'"
- "The poster should NOT have additional text"

EXAMPLES OF WHAT TO KEEP:
- "Title should be in white letters"
- "Character is in center"
- "Building is tall"
- "Background is dark"
- "The poster should NOT have Joker in the image"
- "Batman should NOT have green hair"

Return ONLY a JSON array of strings, each containing one visual requirement. Do not include any explanations or other text.

Example format:
["Title should be in white letters", "Character is in center", "Building is tall"]
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that filters rubric requirements."},
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = self.call_api(messages, max_tokens=2000)

            # Try to parse JSON array from response
            import re

            # Find JSON array in response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                filtered_requirements = json.loads(json_match.group())
                return filtered_requirements
            else:
                print(f"Could not parse JSON array from text filtering response: {response_text[:200]}...")
                return requirements

        except Exception as e:
            print(f"Error filtering text requirements: {e}")
            return requirements

    def deduplicate_poster_requirements(
        self, poster_name: str, requirements: List[str], requirement_type: str = "requirements"
    ) -> List[str]:
        """
        Deduplicate requirements for a single poster using LLM.

        Args:
            poster_name: Name of the poster
            requirements: List of requirements for the poster
            requirement_type: Type of requirements being deduplicated (for prompt clarity)

        Returns:
            Deduplicated list of requirements
        """

        prompt = f"""
You are an expert at deduplicating rubric requirements. I have a list of {requirement_type} for the poster "{poster_name}" that need to be deduplicated.

Your task is to:
1. Identify TRULY identical requirements (same meaning, same specificity)
2. Merge only if they are semantically identical AND have the same level of detail
3. Split complex requirements into separate, checkable items
4. Keep requirements that are semantically different or have different specificity
5. Keep separate characters and objects as separate rubric items - do not combine them
6. Keep different pieces of information as separate rubric items - do not combine them
7. Return the deduplicated list

{requirement_type.upper()} FOR "{poster_name}":
{chr(10).join(f"{i + 1}. {req}" for i, req in enumerate(requirements))}

CRITICAL RULES:
1. KEEP MORE DETAILED VERSION: If one requirement has more detail, keep only the detailed one
2. SPLIT COMPLEX REQUIREMENTS: If a requirement has multiple checkable elements, split it
3. MERGE ONLY IF IDENTICAL: Same meaning AND same level of detail
4. BE CONSERVATIVE: When in doubt, keep separate

EXAMPLES OF WHAT TO MERGE (truly identical):
- "Title should be in white letters" + "Title should be in white text" → "Title should be in white letters"
- "Batman should be in center" + "Batman should be positioned in the middle" → "Batman should be in center"

EXAMPLES OF KEEPING MORE DETAILED VERSION:
- "All four main characters should be wearing sunglasses" + "All characters should be wearing sunglasses" → KEEP ONLY "All four main characters should be wearing sunglasses"
- "Three main characters should be in foreground" + "Characters should be in foreground" → KEEP ONLY "Three main characters should be in foreground"
- "Title should be in bold white letters" + "Title should be in white letters" → KEEP ONLY "Title should be in bold white letters"

EXAMPLES OF WHAT TO SPLIT (complex requirements):
- "Batman figure should be shown from behind in center foreground, with legs slightly apart" → SPLIT INTO:
  * "Batman figure should be shown from behind"
  * "Batman figure should be in center foreground"
  * "Batman figure should have legs slightly apart"

- "Title should be in bold white letters at the bottom" → SPLIT INTO:
  * "Title should be in bold white letters"
  * "Title should be positioned at the bottom"

Return ONLY a JSON array of strings, each containing one simple, checkable requirement. Do not include any explanations or other text.

Example format:
["Batman figure should be shown from behind", "Batman figure should be in center foreground", "Title should be in white letters"]
"""

        messages = [
            {"role": "system", "content": "You are a helpful assistant that deduplicates rubric requirements."},
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = self.call_api(messages, max_tokens=2000)

            # Try to parse JSON array from response
            import re

            # Find JSON array in response
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                deduplicated_requirements = json.loads(json_match.group())
                return deduplicated_requirements
            else:
                print(f"Could not parse JSON array from response for {poster_name}: {response_text[:200]}...")
                return requirements

        except Exception as e:
            print(f"Error deduplicating requirements for {poster_name}: {e}")
            return requirements


def deduplicate_rubric(rubric_file_path: str, output_file_path: str = None, use_llm: bool = True) -> Dict[str, Any]:
    """
    Deduplicate a rubric file by merging similar requirements.

    Args:
        rubric_file_path: Path to the rubric JSON file
        output_file_path: Path to save the deduplicated rubric (optional)
        use_llm: Whether to use LLM for merging (default: True)

    Returns:
        Dictionary containing the deduplicated rubric
    """

    # Load the rubric
    with open(rubric_file_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    print(f"Processing rubric with {rubric['metadata']['total_posters']} posters...")

    if use_llm:
        # Load environment variables
        load_dotenv()

        # Get API key from environment
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Warning: OPENROUTER_API_KEY not set, cannot use LLM deduplication")
            return rubric

        # Initialize deduplicator and process each poster individually
        deduplicator = OpenRouterDeduplicator(api_key)
        print("Using LLM for deduplication (one call per poster)...")

        # Process each poster individually
        total_original = 0
        total_deduplicated = 0

        for poster_id, poster_data in rubric["posters"].items():
            poster_name = poster_data["poster_name"]
            original_requirements = poster_data.get("requirements", [])
            original_negative_requirements = poster_data.get("negative_requirements", [])
            original_text_hallucinations = poster_data.get("text_hallucinations", [])
            original_image_hallucinations = poster_data.get("image_hallucinations", [])

            original_total = (
                len(original_requirements)
                + len(original_negative_requirements)
                + len(original_text_hallucinations)
                + len(original_image_hallucinations)
            )

            print(f"  Processing {poster_name}...")

            # Process positive requirements
            if original_requirements:
                print(f"    Processing positive requirements...")
                # Round 1: Deduplicate
                deduplicated_requirements = deduplicator.deduplicate_poster_requirements(
                    poster_name, original_requirements, "positive requirements"
                )
                deduplicated_count = len(deduplicated_requirements)
                print(f"      {len(original_requirements)} → {deduplicated_count} after deduplication")

                # Round 2: Filter text
                final_requirements = deduplicator.filter_text_requirements(
                    deduplicated_requirements, "positive requirements"
                )
                final_count = len(final_requirements)
                print(f"      {deduplicated_count} → {final_count} after text filtering")
            else:
                final_requirements = []
                final_count = 0

            # Process negative requirements
            if original_negative_requirements:
                print(f"    Processing negative requirements...")
                # Round 1: Deduplicate
                deduplicated_negative = deduplicator.deduplicate_poster_requirements(
                    poster_name, original_negative_requirements, "negative requirements"
                )
                deduplicated_negative_count = len(deduplicated_negative)
                print(
                    f"      {len(original_negative_requirements)} → {deduplicated_negative_count} after deduplication"
                )

                # Round 2: Filter text
                final_negative_requirements = deduplicator.filter_text_requirements(
                    deduplicated_negative, "negative requirements"
                )
                final_negative_count = len(final_negative_requirements)
                print(f"      {deduplicated_negative_count} → {final_negative_count} after text filtering")
            else:
                final_negative_requirements = []
                final_negative_count = 0

            # Process text hallucinations
            if original_text_hallucinations:
                print(f"    Processing text hallucinations...")
                final_text_hallucinations = deduplicator.filter_text_requirements(
                    original_text_hallucinations, "text hallucinations"
                )
                final_text_hallucinations_count = len(final_text_hallucinations)
                print(
                    f"      {len(original_text_hallucinations)} → {final_text_hallucinations_count} after text filtering"
                )
            else:
                final_text_hallucinations = []
                final_text_hallucinations_count = 0

            # Process image hallucinations
            if original_image_hallucinations:
                print(f"    Processing image hallucinations...")
                final_image_hallucinations = deduplicator.filter_text_requirements(
                    original_image_hallucinations, "image hallucinations"
                )
                final_image_hallucinations_count = len(final_image_hallucinations)
                print(
                    f"      {len(original_image_hallucinations)} → {final_image_hallucinations_count} after text filtering"
                )
            else:
                final_image_hallucinations = []
                final_image_hallucinations_count = 0

            # Create a new poster data dict with all original fields preserved
            new_poster_data = {
                "poster_name": poster_data["poster_name"],
                "poster_id": poster_data["poster_id"],
                "requirements": final_requirements,
                "original_requirement_count": len(original_requirements),
                "deduplicated_requirement_count": deduplicated_count if original_requirements else 0,
                "final_requirement_count": final_count,
            }

            # Add negative requirements if they exist
            if original_negative_requirements:
                new_poster_data["negative_requirements"] = final_negative_requirements
                new_poster_data["original_negative_requirement_count"] = len(original_negative_requirements)
                new_poster_data["deduplicated_negative_requirement_count"] = deduplicated_negative_count
                new_poster_data["final_negative_requirement_count"] = final_negative_count

            # Add hallucinations if they exist
            if original_text_hallucinations:
                new_poster_data["text_hallucinations"] = final_text_hallucinations
                new_poster_data["original_text_hallucination_count"] = len(original_text_hallucinations)
                new_poster_data["final_text_hallucination_count"] = final_text_hallucinations_count

            if original_image_hallucinations:
                new_poster_data["image_hallucinations"] = final_image_hallucinations
                new_poster_data["original_image_hallucination_count"] = len(original_image_hallucinations)
                new_poster_data["final_image_hallucination_count"] = final_image_hallucinations_count

            # Copy over any other fields that exist in the original data
            for key, value in poster_data.items():
                if key not in new_poster_data:
                    new_poster_data[key] = value

            # Update the poster data
            poster_data.clear()
            poster_data.update(new_poster_data)

            final_total = (
                final_count + final_negative_count + final_text_hallucinations_count + final_image_hallucinations_count
            )
            total_original += original_total
            total_deduplicated += final_total

            print(f"    Final: {original_total} → {final_total} total items")

        print(f"Total: {total_original} → {total_deduplicated} requirements")

        # Update metadata
        rubric["metadata"]["deduplication_applied"] = True
        rubric["metadata"]["llm_used"] = True
        rubric["metadata"]["processed_content_types"] = [
            "positive_requirements",
            "negative_requirements",
            "text_hallucinations",
            "image_hallucinations",
        ]
    else:
        print("LLM deduplication disabled")
        return rubric

    # Save the deduplicated rubric if output path is provided
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(rubric, f, indent=2, ensure_ascii=False)
        print(f"\nDeduplicated rubric saved to: {output_file_path}")

    return rubric


def main():
    parser = argparse.ArgumentParser(description="Deduplicate rubric requirements by merging similar statements")
    parser.add_argument("rubric_file", help="Path to the rubric JSON file")
    parser.add_argument(
        "-o", "--output", help="Output file path for the deduplicated rubric (default: deduplicated_<filename>.json)"
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM usage (will return original rubric)")

    args = parser.parse_args()

    # Generate default output path if not provided
    if not args.output:
        # Get the directory of the rubric file
        rubric_dir = os.path.dirname(args.rubric_file)
        base_name = os.path.splitext(os.path.basename(args.rubric_file))[0]
        args.output = os.path.join(rubric_dir, f"deduplicated_{base_name}.json")

    # Deduplicate the rubric
    rubric = deduplicate_rubric(args.rubric_file, args.output, use_llm=not args.no_llm)

    # Print summary
    total_original = sum(poster.get("original_requirement_count", 0) for poster in rubric["posters"].values())
    total_deduplicated = sum(poster.get("deduplicated_requirement_count", 0) for poster in rubric["posters"].values())

    # Calculate totals for all types
    total_original_all = sum(
        poster.get("original_requirement_count", 0)
        + poster.get("original_negative_requirement_count", 0)
        + poster.get("original_text_hallucination_count", 0)
        + poster.get("original_image_hallucination_count", 0)
        for poster in rubric["posters"].values()
    )

    total_final_all = sum(
        poster.get("final_requirement_count", 0)
        + poster.get("final_negative_requirement_count", 0)
        + poster.get("final_text_hallucination_count", 0)
        + poster.get("final_image_hallucination_count", 0)
        for poster in rubric["posters"].values()
    )

    print(f"\nDeduplication complete!")
    print(f"Positive requirements: {total_original} → {total_deduplicated}")
    print(f"All content types: {total_original_all} → {total_final_all}")
    print(f"Total reduction: {total_original_all - total_final_all} items")
    print(f"Reduction percentage: {((total_original_all - total_final_all) / total_original_all * 100):.1f}%")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
