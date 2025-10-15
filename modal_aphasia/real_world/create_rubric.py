#!/usr/bin/env python3
"""
Script to create rubrics from graded.json files.
Extracts positive statements about what should be present in each poster.
Set use_negative_as_requirements to True to convert incorrect/hallucinated content into negative requirements (e.g., "should NOT have Joker"). Returns separate positive and negative requirement lists.
"""

import argparse
import json
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv


class OpenRouterRubricExtractor:
    def __init__(self, api_key: str, model: str = "anthropic/claude-3.5-sonnet"):
        """
        Initialize the OpenRouter rubric extractor.

        Args:
            api_key: OpenRouter API key
            model: Model to use for extraction (default: claude-3.5-sonnet)
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

    def extract_hallucinations(self, comparison_data: Dict) -> Dict[str, List[str]]:
        """
        Extract statements about elements that are not present in the original (hallucinations).
        Returns separate lists for text and image hallucinations.
        """
        result = {"text_hallucinations": [], "image_hallucinations": []}

        # Extract text hallucinations from text_to_image comparison
        if "text_to_image" in comparison_data:
            text_comp = comparison_data["text_to_image"]
            text_prompt = f"""
You are an expert at analyzing movie poster descriptions and identifying text hallucinations.

TEXT-TO-IMAGE COMPARISON (comparing a text description to the actual poster image):
Scores: {text_comp["scores"]}
Analysis: {text_comp["textual_analysis"]}

CRITICAL: Return ONLY a JSON list. NO explanations, NO reasoning, NO thoughts.

If you find elements marked as "not present in original", list them as strings.
If no such elements exist, return an empty list [].

Example: ["Extra character not in original", "Additional prop doesn't exist"]
"""
            try:
                text_response = self.call_api([{"role": "user", "content": text_prompt}], max_tokens=800)
                try:
                    import json

                    result["text_hallucinations"] = json.loads(text_response)
                except json.JSONDecodeError:
                    lines = [line.strip() for line in text_response.split("\n") if line.strip()]
                    result["text_hallucinations"] = [
                        line for line in lines if line and not line.startswith("#") and not line.startswith("-")
                    ]
            except Exception as e:
                print(f"Error extracting text hallucinations: {e}")

        # Extract image hallucinations from image_to_image comparison
        if "image_to_image" in comparison_data:
            img_comp = comparison_data["image_to_image"]
            img_prompt = f"""
You are an expert at analyzing movie poster images and identifying image hallucinations.

IMAGE-TO-IMAGE COMPARISON (comparing original poster to replicated poster):
Scores: {img_comp["scores"]}
Analysis: {img_comp["textual_analysis"]}

CRITICAL: Return ONLY a JSON list. NO explanations, NO reasoning, NO thoughts.

If you find elements marked as "not present in original", list them as strings.
If no such elements exist, return an empty list [].

Example: ["Extra background not in original", "Additional prop doesn't exist"]
"""
            try:
                img_response = self.call_api([{"role": "user", "content": img_prompt}], max_tokens=800)
                try:
                    import json

                    result["image_hallucinations"] = json.loads(img_response)
                except json.JSONDecodeError:
                    lines = [line.strip() for line in img_response.split("\n") if line.strip()]
                    result["image_hallucinations"] = [
                        line for line in lines if line and not line.startswith("#") and not line.startswith("-")
                    ]
            except Exception as e:
                print(f"Error extracting image hallucinations: {e}")

        return result

    def extract_positive_statements(
        self, comparison_data: Dict, use_negative_as_requirements: bool = False
    ) -> Dict[str, List[str]]:
        """
        Extract positive statements from grading text using LLM.

        Args:
            comparison_data: Dictionary containing comparison data
            use_negative_as_requirements: If True, return both positive and negative requirements in separate lists
                                        If False, return only positive requirements (current behavior)

        Returns:
            If use_negative_as_requirements=False: {"positive": [list of positive requirements]}
            If use_negative_as_requirements=True: {"positive": [list of positive requirements], "negative": [list of negative requirements]}
        """
        # Build the prompt with both comparison types
        prompt_parts = []

        # Add text_to_image comparison if available
        if "text_to_image" in comparison_data:
            text_comp = comparison_data["text_to_image"]
            prompt_parts.append(f"""
TEXT-TO-IMAGE COMPARISON (comparing a text description to the actual poster image):
Scores: {text_comp["scores"]}
Analysis: {text_comp["textual_analysis"]}
""")

        # Add image_to_image comparison if available
        if "image_to_image" in comparison_data:
            img_comp = comparison_data["image_to_image"]
            prompt_parts.append(f"""
IMAGE-TO-IMAGE COMPARISON (comparing original poster to replicated poster):
Scores: {img_comp["scores"]}
Analysis: {img_comp["textual_analysis"]}
""")

        combined_analysis = "\n".join(prompt_parts)

        if use_negative_as_requirements:
            # Convert incorrect/hallucinated content into negative requirements
            prompt = f"""You are an expert at creating rubrics for movie poster descriptions. I have grading data from two different types of comparisons that describe what elements are present, missing, or inaccurate in a movie poster.

Your task is to extract BOTH positive statements about what should be present AND negative statements about what should NOT be present in the poster.

{combined_analysis}

CRITICAL RULES:
1. For "present and accurate" elements → Include as positive requirements
2. For "not present in replication" elements → Include as positive requirements (these should be present)
3. For "not present in original" elements → Convert to NEGATIVE requirements (these should NOT be present)
4. For "not present in description" elements → Include as positive requirements (these are important elements that should be mentioned)
5. For "present but inaccurate" elements → Include the CORRECTED version as a positive requirement AND the incorrect version as a negative requirement
6. ALWAYS use specific character names, object names, and details - NEVER use generic terms like "figure", "character", or "element"
7. Convert all statements into rubric format
8. Combine insights from both comparison types to create a comprehensive rubric

Examples of what to INCLUDE as POSITIVE requirements:
- "Three main characters in foreground" → "Three main characters should be in foreground"
- "The sword is positioned correctly in the center" → "Sword should be positioned in the center"
- "Gothic architecture in background" → "Gothic architecture should be in background"
- "The character is not holding a gun it is holding a sword" → "The character should be holding a sword"
- "White border frame not present in replication" → "White border frame should be present"
- "Background mountain not mentioned in description" → "Background mountain should be mentioned"

Examples of what to INCLUDE as NEGATIVE requirements (when use_negative_as_requirements=True):
- "Extra character not present in original" → "The poster should NOT have [specific character name] in the image"
- "Additional text not in original" → "The poster should NOT have [specific text content] in the image"
- "The character is not holding a gun it is holding a sword" → "[Character name] should NOT be holding a gun"
- "Red color instead of blue" → "[Specific element name] should NOT be red in color"
- "Character on left instead of right" → "[Character name] should NOT be positioned on the left"
- "Joker character not present in original" → "The poster should NOT have Joker in the image"
- "Frodo should not hold the ring in his hand" → "Frodo should NOT be holding the ring in his hand"
- "Batman has green hair instead of black" → "Batman should NOT have green hair"
- "The Joker is wearing a purple coat" → "The Joker should NOT be wearing a purple coat"
- "Date shows '07.18.08' instead of '2008'" → "The date should NOT be shown as '07.18.08'"

Examples of what to EXCLUDE:
- "Water pooled around characters' feet - NOT PRESENT" (from text-to-image) → DO NOT include
- "Cracks in pavement - NOT CLEARLY VISIBLE" (from text-to-image) → DO NOT include

IMPORTANT: Characters and objects should NOT be combined in the rubric. Each character and each object should be a separate instance in the rubric.

Return ONLY a JSON object with two arrays: "positive" for positive requirements and "negative" for negative requirements. Do not include any explanations or other text.

Example format:
{{
  "positive": ["Three main characters should be in foreground", "Sword should be positioned correctly"],
  "negative": ["The poster should NOT have Joker in the image", "Frodo should NOT be holding the ring in his hand"]
}}"""
        else:
            # Ignore incorrect/hallucinated content (use_negative_as_requirements=False)
            prompt = f"""You are an expert at creating rubrics for movie poster descriptions. I have grading data from two different types of comparisons that describe what elements are present, missing, or inaccurate in a movie poster.

Your task is to extract ONLY positive statements about what should be present in the poster. IMPORTANT: Only include elements that are actually present and correct in the poster.

{combined_analysis}

CRITICAL RULES:
1. For "present and accurate" elements → Include as positive requirements
2. For "not present in replication" elements → Include as positive requirements (these should be present)
3. For "not present in original" elements → DO NOT include (these are extra elements that shouldn't be there)
4. For "not present in description" elements → Include as positive requirements (these are important elements that should be mentioned)
5. For "present but inaccurate" elements → Include the CORRECTED version as a positive requirement if there is a correction for the mistake.
6. For "NOT PRESENT" or "NOT CLEARLY VISIBLE" in text-to-image → DO NOT include (these are unclear/missing from description)
7. Convert all positive and corrected statements into rubric format
8. Combine insights from both comparison types to create a comprehensive rubric

Examples of what to INCLUDE:
- "Three main characters in foreground" → "Three main characters should be in foreground"
- "The sword is positioned correctly in the center" → "Sword should be positioned in the center"
- "Gothic architecture in background" → "Gothic architecture should be in background"
- "The character is not holding a gun it is holding a sword" → "The character should be holding a sword"
- "White border frame not present in replication" → "White border frame should be present"
- "Thin white line border missing from replication" → "Thin white line border should be present"
- "Character on left instead of right" → "Character should be positioned on the right"
- "Red color instead of blue" → "Element should be blue in color"
- "Background mountain not mentioned in description" → "Background mountain should be mentioned"
- "Supporting character not described" → "Supporting character should be described"

Examples of what to EXCLUDE:
- "Water pooled around characters' feet - NOT PRESENT" (from text-to-image) → DO NOT include
- "Cracks in pavement - NOT CLEARLY VISIBLE" (from text-to-image) → DO NOT include
- "Extra character not present in original" → DO NOT include (this is an extra element)
- "Additional text not in original" → DO NOT include (this is an extra element)

IMPORTANT: Characters and objects should NOT be combined in the rubric. Each character and each object should be a separate instance in the rubric.

Return ONLY a JSON array of strings, each containing one positive requirement. Do not include any explanations or other text.

Example format:
["Three main characters should be in foreground", "Sword should be positioned correctly", "Gothic architecture should be in background"]"""

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that creates positive rubric statements from grading text.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response_text = self.call_api(messages, max_tokens=1000)

            # Extract JSON from response
            content = response_text.strip()

            # Parse as JSON
            statements = json.loads(content)

            if use_negative_as_requirements:
                # Return both positive and negative requirements
                if isinstance(statements, dict) and "positive" in statements and "negative" in statements:
                    positive = [s for s in statements.get("positive", []) if s and len(s) > 5]
                    negative = [s for s in statements.get("negative", []) if s and len(s) > 5]
                    return {"positive": positive, "negative": negative}
                else:
                    # Fallback: treat as list of positive requirements
                    if isinstance(statements, list):
                        positive = [s for s in statements if s and len(s) > 5]
                        return {"positive": positive, "negative": []}
                    return {"positive": [], "negative": []}
            else:
                # Return only positive requirements (backward compatibility)
                if isinstance(statements, list):
                    return {"positive": [s for s in statements if s and len(s) > 5]}
                elif isinstance(statements, dict) and "positive" in statements:
                    return {"positive": [s for s in statements.get("positive", []) if s and len(s) > 5]}
                return {"positive": []}
        except Exception as e:
            print(f"Error extracting statements: {e}")
            if use_negative_as_requirements:
                return {"positive": [], "negative": []}
            else:
                return {"positive": []}


def create_rubric_from_graded_file(
    graded_file_path: str,
    output_file_path: str = None,
    extractor: OpenRouterRubricExtractor = None,
    use_negative_as_requirements: bool = False,
) -> Dict[str, Any]:
    """
    Create a rubric from a graded.json file.

    Args:
        graded_file_path: Path to the graded.json file
        output_file_path: Path to save the rubric (optional)
        extractor: OpenRouterRubricExtractor instance

    Returns:
        Dictionary containing the rubric
    """

    # Load the graded data
    with open(graded_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rubric = {
        "metadata": {
            "source_file": graded_file_path,
            "total_posters": 0,
            "rubric_created": True,
            "llm_used": True,
            "model": extractor.model if extractor else "unknown",
            "use_negative_as_requirements": use_negative_as_requirements,
        },
        "posters": {},
    }

    # Process each evaluation
    for evaluation in data.get("evaluations", []):
        poster_id = evaluation.get("poster_id", "unknown")
        poster_name = evaluation.get("poster_name", f"Poster {poster_id}")

        print(f"Processing poster: {poster_name} (ID: {poster_id})")

        poster_rubric = {
            "poster_name": poster_name,
            "poster_id": poster_id,
            "requirements": [],
            "text_hallucinations": [],
            "image_hallucinations": [],
        }

        # Add negative_requirements field if using that approach
        if use_negative_as_requirements:
            poster_rubric["negative_requirements"] = []

        # Collect all comparison data
        comparison_data = {}

        # Extract from text_to_image_comparison if available
        if "image_to_text_comparison" in evaluation:
            comp = evaluation["image_to_text_comparison"]
            if "textual_analysis" in comp:
                comparison_data["text_to_image"] = {
                    "textual_analysis": comp["textual_analysis"],
                    "scores": comp.get("scores", {}),
                }

        # Extract from image_to_image_comparison if available
        if "image_to_image_comparison" in evaluation:
            comp = evaluation["image_to_image_comparison"]
            if "textual_analysis" in comp:
                comparison_data["image_to_image"] = {
                    "textual_analysis": comp["textual_analysis"],
                    "scores": comp.get("scores", {}),
                }

        # Extract requirements and hallucinations from all available comparisons
        if comparison_data:
            # Extract requirements (positive and optionally negative)
            statements = extractor.extract_positive_statements(comparison_data, use_negative_as_requirements)

            # Add positive requirements
            poster_rubric["requirements"].extend(statements.get("positive", []))

            # Add negative requirements if using that approach
            if use_negative_as_requirements and "negative" in statements:
                poster_rubric["negative_requirements"] = statements.get("negative", [])

            # Extract hallucinations (not present in original)
            hallucinations = extractor.extract_hallucinations(comparison_data)
            poster_rubric["text_hallucinations"].extend(hallucinations.get("text_hallucinations", []))
            poster_rubric["image_hallucinations"].extend(hallucinations.get("image_hallucinations", []))

        # Remove duplicates and clean up
        poster_rubric["requirements"] = list(set(poster_rubric["requirements"]))
        poster_rubric["requirements"] = [s for s in poster_rubric["requirements"] if s and len(s) > 10]

        # Clean up hallucinations
        poster_rubric["text_hallucinations"] = list(set(poster_rubric["text_hallucinations"]))
        poster_rubric["text_hallucinations"] = [s for s in poster_rubric["text_hallucinations"] if s and len(s) > 5]

        poster_rubric["image_hallucinations"] = list(set(poster_rubric["image_hallucinations"]))
        poster_rubric["image_hallucinations"] = [s for s in poster_rubric["image_hallucinations"] if s and len(s) > 5]

        # Sort requirements and hallucinations
        poster_rubric["requirements"].sort()
        poster_rubric["text_hallucinations"].sort()
        poster_rubric["image_hallucinations"].sort()

        # Clean up and sort negative requirements if using that approach
        if use_negative_as_requirements and "negative_requirements" in poster_rubric:
            poster_rubric["negative_requirements"] = list(set(poster_rubric["negative_requirements"]))
            poster_rubric["negative_requirements"] = [
                s for s in poster_rubric["negative_requirements"] if s and len(s) > 10
            ]
            poster_rubric["negative_requirements"].sort()

        rubric["posters"][poster_id] = poster_rubric
        rubric["metadata"]["total_posters"] += 1

        print(f"  Found {len(poster_rubric['requirements'])} positive requirements")
        if use_negative_as_requirements and "negative_requirements" in poster_rubric:
            print(f"  Found {len(poster_rubric['negative_requirements'])} negative requirements")
        if poster_rubric["text_hallucinations"]:
            print(f"  Found {len(poster_rubric['text_hallucinations'])} text hallucinations")
        if poster_rubric["image_hallucinations"]:
            print(f"  Found {len(poster_rubric['image_hallucinations'])} image hallucinations")

    # Save the rubric if output path is provided
    if output_file_path:
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(rubric, f, indent=2, ensure_ascii=False)
        print(f"\nRubric saved to: {output_file_path}")

    return rubric


def main():
    parser = argparse.ArgumentParser(description="Create rubrics from graded.json files")
    parser.add_argument("graded_file", help="Path to the graded.json file")
    parser.add_argument("-o", "--output", help="Output file path for the rubric (default: rubric_<filename>.json)")
    parser.add_argument(
        "--model",
        default="anthropic/claude-3.5-sonnet",
        help="Model to use for extraction (default: anthropic/claude-3.5-sonnet)",
    )
    parser.add_argument(
        "--use-negative-as-requirements",
        action="store_true",
        help='Convert incorrect/hallucinated content into negative requirements (e.g., "should NOT have Joker"). Returns separate positive and negative requirement lists.',
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return

    # Initialize the extractor
    extractor = OpenRouterRubricExtractor(api_key, args.model)

    # Generate default output path if not provided
    if not args.output:
        # Get the directory of the graded file
        graded_dir = os.path.dirname(args.graded_file)
        base_name = os.path.splitext(os.path.basename(args.graded_file))[0]
        args.output = os.path.join(graded_dir, f"rubric_{base_name}.json")

    # Create the rubric
    rubric = create_rubric_from_graded_file(args.graded_file, args.output, extractor, args.use_negative_as_requirements)

    # Print summary
    print(f"\nRubric creation complete!")
    print(f"Processed {rubric['metadata']['total_posters']} posters")
    print(f"LLM used: {rubric['metadata']['llm_used']}")
    print(f"Output saved to: {args.output}")

    # Print sample requirements and hallucinations for first poster
    if rubric["posters"]:
        first_poster_id = list(rubric["posters"].keys())[0]
        first_poster = rubric["posters"][first_poster_id]
        print(f"\nSample positive requirements for {first_poster['poster_name']}:")
        for i, req in enumerate(first_poster["requirements"][:5], 1):
            print(f"  {i}. {req}")
        if len(first_poster["requirements"]) > 5:
            print(f"  ... and {len(first_poster['requirements']) - 5} more")

        # Show negative requirements if using that approach
        if args.use_negative_as_requirements and "negative_requirements" in first_poster:
            print(f"\nSample negative requirements for {first_poster['poster_name']}:")
            for i, req in enumerate(first_poster["negative_requirements"][:5], 1):
                print(f"  {i}. {req}")
            if len(first_poster["negative_requirements"]) > 5:
                print(f"  ... and {len(first_poster['negative_requirements']) - 5} more")

        # Show hallucinations if any
        if first_poster["text_hallucinations"]:
            print(f"\nText hallucinations (not present in original):")
            for i, hall in enumerate(first_poster["text_hallucinations"][:3], 1):
                print(f"  {i}. {hall}")
            if len(first_poster["text_hallucinations"]) > 3:
                print(f"  ... and {len(first_poster['text_hallucinations']) - 3} more")

        if first_poster["image_hallucinations"]:
            print(f"\nImage hallucinations (not present in original):")
            for i, hall in enumerate(first_poster["image_hallucinations"][:3], 1):
                print(f"  {i}. {hall}")
            if len(first_poster["image_hallucinations"]) > 3:
                print(f"  ... and {len(first_poster['image_hallucinations']) - 3} more")


if __name__ == "__main__":
    main()
