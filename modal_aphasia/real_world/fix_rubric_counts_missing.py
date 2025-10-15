#!/usr/bin/env python3
"""
Script to fix rubric counts by removing requirements that aren't in the positive_requirements
or negative_requirements lists, recalculating scores, and reporting irregularities.
Supports MISSING status for positive requirements and checks for ungraded requirements.
"""

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple


class RubricCountFixer:
    def __init__(self):
        """Initialize the rubric count fixer."""
        self.data = {}
        self.irregularities = []
        self.fixes_applied = []

    def load_file(self, file_path: str) -> Dict:
        """Load a rubric results file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded: {file_path} ({len(data.get('posters', {}))} posters)")
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise

    def get_valid_requirements(self, poster_data: Dict) -> Set[str]:
        """Get all valid requirements from positive and negative requirements lists."""
        valid_requirements = set()

        # Add positive requirements
        positive_reqs = poster_data.get("positive_requirements", [])
        for req in positive_reqs:
            valid_requirements.add(req)

        # Add negative requirements
        negative_reqs = poster_data.get("negative_requirements", [])
        for req in negative_reqs:
            valid_requirements.add(req)

        return valid_requirements

    def fix_fulfillment_analysis(
        self, analysis: List[Dict], valid_requirements: Set[str], modality: str, poster_name: str
    ) -> Tuple[List[Dict], List[str]]:
        """Fix fulfillment analysis by removing invalid requirements."""
        fixed_analysis = []
        removed_requirements = []

        for item in analysis:
            requirement = item.get("requirement", "")

            if requirement in valid_requirements:
                fixed_analysis.append(item)
            else:
                removed_requirements.append(requirement)
                self.irregularities.append(
                    {
                        "type": "invalid_requirement",
                        "poster": poster_name,
                        "modality": modality,
                        "requirement": requirement,
                        "action": "removed",
                    }
                )

        return fixed_analysis, removed_requirements

    def recalculate_scores(
        self, analysis: List[Dict], positive_requirements: List[str], negative_requirements: List[str]
    ) -> Dict[str, Any]:
        """Recalculate scores based on the analysis with positive/negative breakdown."""
        correct_count = 0
        incorrect_count = 0
        missing_count = 0
        positive_correct = 0
        positive_incorrect = 0
        positive_missing = 0
        negative_correct = 0
        negative_incorrect = 0

        for item in analysis:
            requirement = item.get("requirement", "")
            status = item.get("status", "").upper()

            if status == "CORRECT":
                correct_count += 1
                if requirement in positive_requirements:
                    positive_correct += 1
                elif requirement in negative_requirements:
                    negative_correct += 1
            elif status == "INCORRECT":
                incorrect_count += 1
                if requirement in positive_requirements:
                    positive_incorrect += 1
                elif requirement in negative_requirements:
                    negative_incorrect += 1
            elif status == "MISSING":
                missing_count += 1
                if requirement in positive_requirements:
                    positive_missing += 1
                # Note: negative requirements can't be "missing" - they're either present (incorrect) or absent (correct)

        total = correct_count + incorrect_count + missing_count

        if total == 0:
            return {
                "correct": 0,
                "correct_pct": 0.0,
                "incorrect": 0,
                "incorrect_pct": 0.0,
                "missing": 0,
                "missing_pct": 0.0,
                "positive_negative_breakdown": {
                    "positive_correct": 0,
                    "positive_incorrect": 0,
                    "positive_missing": 0,
                    "negative_correct": 0,
                    "negative_incorrect": 0,
                },
            }

        return {
            "correct": correct_count,
            "correct_pct": round((correct_count / total) * 100, 1),
            "incorrect": incorrect_count,
            "incorrect_pct": round((incorrect_count / total) * 100, 1),
            "missing": missing_count,
            "missing_pct": round((missing_count / total) * 100, 1),
            "positive_negative_breakdown": {
                "positive_correct": positive_correct,
                "positive_incorrect": positive_incorrect,
                "positive_missing": positive_missing,
                "negative_correct": negative_correct,
                "negative_incorrect": negative_incorrect,
            },
        }

    def fix_poster_data(self, poster_id: str, poster_data: Dict) -> Dict:
        """Fix a single poster's data."""
        poster_name = poster_data.get("poster_name", f"Poster {poster_id}")
        print(f"  Processing {poster_name}...")

        # Get valid requirements
        valid_requirements = self.get_valid_requirements(poster_data)
        print(f"    Valid requirements: {len(valid_requirements)}")

        # Fix text fulfillment
        text_fulfillment = poster_data.get("text_fulfillment", {})
        if text_fulfillment:
            text_analysis = text_fulfillment.get("text_fulfillment_analysis", [])
            original_text_count = len(text_analysis)

            fixed_text_analysis, removed_text = self.fix_fulfillment_analysis(
                text_analysis, valid_requirements, "text", poster_name
            )

            if removed_text:
                print(f"    Removed {len(removed_text)} invalid text requirements")
                self.fixes_applied.append(
                    {
                        "poster": poster_name,
                        "modality": "text",
                        "removed_count": len(removed_text),
                        "removed_requirements": removed_text,
                    }
                )

            # Recalculate text scores
            new_text_scores = self.recalculate_scores(
                fixed_text_analysis,
                poster_data.get("positive_requirements", []),
                poster_data.get("negative_requirements", []),
            )

            # Update text fulfillment
            poster_data["text_fulfillment"]["text_fulfillment_analysis"] = fixed_text_analysis
            poster_data["text_fulfillment"]["scores"] = new_text_scores

            print(f"    Text: {original_text_count} → {len(fixed_text_analysis)} requirements")
            total_text = new_text_scores["correct"] + new_text_scores["incorrect"] + new_text_scores["missing"]
            print(
                f"    Text scores: {new_text_scores['correct']}/{total_text} correct ({new_text_scores['correct_pct']}%), {new_text_scores['missing']} missing ({new_text_scores['missing_pct']}%)"
            )

        # Fix image fulfillment
        image_fulfillment = poster_data.get("image_fulfillment", {})
        if image_fulfillment:
            image_analysis = image_fulfillment.get("image_fulfillment_analysis", [])
            original_image_count = len(image_analysis)

            fixed_image_analysis, removed_image = self.fix_fulfillment_analysis(
                image_analysis, valid_requirements, "image", poster_name
            )

            if removed_image:
                print(f"    Removed {len(removed_image)} invalid image requirements")
                self.fixes_applied.append(
                    {
                        "poster": poster_name,
                        "modality": "image",
                        "removed_count": len(removed_image),
                        "removed_requirements": removed_image,
                    }
                )

            # Recalculate image scores
            new_image_scores = self.recalculate_scores(
                fixed_image_analysis,
                poster_data.get("positive_requirements", []),
                poster_data.get("negative_requirements", []),
            )

            # Update image fulfillment
            poster_data["image_fulfillment"]["image_fulfillment_analysis"] = fixed_image_analysis
            poster_data["image_fulfillment"]["scores"] = new_image_scores

            print(f"    Image: {original_image_count} → {len(fixed_image_analysis)} requirements")
            total_image = new_image_scores["correct"] + new_image_scores["incorrect"] + new_image_scores["missing"]
            print(
                f"    Image scores: {new_image_scores['correct']}/{total_image} correct ({new_image_scores['correct_pct']}%), {new_image_scores['missing']} missing ({new_image_scores['missing_pct']}%)"
            )

        # Fix summary if it exists
        summary = poster_data.get("summary", {})
        if summary:
            # Recalculate summary from fixed data
            text_scores = poster_data.get("text_fulfillment", {}).get("scores", {})
            image_scores = poster_data.get("image_fulfillment", {}).get("scores", {})

            # Get positive and negative requirements counts
            positive_requirements = poster_data.get("positive_requirements", [])
            negative_requirements = poster_data.get("negative_requirements", [])
            total_requirements = len(positive_requirements) + len(negative_requirements)

            # Get breakdown data
            text_breakdown = text_scores.get("positive_negative_breakdown", {})
            image_breakdown = image_scores.get("positive_negative_breakdown", {})

            new_summary = {
                "text_correct": text_scores.get("correct", 0),
                "text_correct_pct": text_scores.get("correct_pct", 0.0),
                "text_incorrect": text_scores.get("incorrect", 0),
                "text_incorrect_pct": text_scores.get("incorrect_pct", 0.0),
                "text_missing": text_scores.get("missing", 0),
                "text_missing_pct": text_scores.get("missing_pct", 0.0),
                "image_correct": image_scores.get("correct", 0),
                "image_correct_pct": image_scores.get("correct_pct", 0.0),
                "image_incorrect": image_scores.get("incorrect", 0),
                "image_incorrect_pct": image_scores.get("incorrect_pct", 0.0),
                "image_missing": image_scores.get("missing", 0),
                "image_missing_pct": image_scores.get("missing_pct", 0.0),
                "total_requirements": total_requirements,
                "positive_negative_breakdown": {
                    "text": {
                        "positive_correct": text_breakdown.get("positive_correct", 0),
                        "positive_incorrect": text_breakdown.get("positive_incorrect", 0),
                        "positive_missing": text_breakdown.get("positive_missing", 0),
                        "negative_correct": text_breakdown.get("negative_correct", 0),
                        "negative_incorrect": text_breakdown.get("negative_incorrect", 0),
                    },
                    "image": {
                        "positive_correct": image_breakdown.get("positive_correct", 0),
                        "positive_incorrect": image_breakdown.get("positive_incorrect", 0),
                        "positive_missing": image_breakdown.get("positive_missing", 0),
                        "negative_correct": image_breakdown.get("negative_correct", 0),
                        "negative_incorrect": image_breakdown.get("negative_incorrect", 0),
                    },
                    "rubric_counts": {
                        "positive_rubric": len(positive_requirements),
                        "negative_rubric": len(negative_requirements),
                    },
                },
            }

            poster_data["summary"] = new_summary
            print(f"    Summary updated with complete breakdown")

        return poster_data

    def check_for_irregularities(self, poster_data: Dict) -> List[Dict]:
        """Check for various irregularities in the poster data."""
        poster_name = poster_data.get("poster_name", "Unknown")
        irregularities = []

        # Check for missing requirements lists
        if "positive_requirements" not in poster_data:
            irregularities.append({"type": "missing_positive_requirements", "poster": poster_name, "action": "warning"})

        if "negative_requirements" not in poster_data:
            irregularities.append({"type": "missing_negative_requirements", "poster": poster_name, "action": "warning"})

        # Check for empty requirements lists
        positive_reqs = poster_data.get("positive_requirements", [])
        negative_reqs = poster_data.get("negative_requirements", [])

        if not positive_reqs and not negative_reqs:
            irregularities.append({"type": "no_requirements", "poster": poster_name, "action": "error"})

        # Check for duplicate requirements
        all_reqs = positive_reqs + negative_reqs
        if len(all_reqs) != len(set(all_reqs)):
            duplicates = []
            seen = set()
            for req in all_reqs:
                if req in seen:
                    duplicates.append(req)
                else:
                    seen.add(req)

            irregularities.append(
                {"type": "duplicate_requirements", "poster": poster_name, "duplicates": duplicates, "action": "warning"}
            )

        # Check for score inconsistencies
        text_fulfillment = poster_data.get("text_fulfillment", {})
        image_fulfillment = poster_data.get("image_fulfillment", {})

        if text_fulfillment:
            text_analysis = text_fulfillment.get("text_fulfillment_analysis", [])
            text_scores = text_fulfillment.get("scores", {})

            # Count actual correct/incorrect/missing in analysis
            actual_correct = len([item for item in text_analysis if item.get("status") == "CORRECT"])
            actual_incorrect = len([item for item in text_analysis if item.get("status") == "INCORRECT"])
            actual_missing = len([item for item in text_analysis if item.get("status") == "MISSING"])

            # Compare with stored scores
            stored_correct = text_scores.get("correct", 0)
            stored_incorrect = text_scores.get("incorrect", 0)
            stored_missing = text_scores.get("missing", 0)

            if (
                actual_correct != stored_correct
                or actual_incorrect != stored_incorrect
                or actual_missing != stored_missing
            ):
                irregularities.append(
                    {
                        "type": "text_score_mismatch",
                        "poster": poster_name,
                        "actual": {"correct": actual_correct, "incorrect": actual_incorrect, "missing": actual_missing},
                        "stored": {"correct": stored_correct, "incorrect": stored_incorrect, "missing": stored_missing},
                        "action": "fixed",
                    }
                )

        if image_fulfillment:
            image_analysis = image_fulfillment.get("image_fulfillment_analysis", [])
            image_scores = image_fulfillment.get("scores", {})

            # Count actual correct/incorrect/missing in analysis
            actual_correct = len([item for item in image_analysis if item.get("status") == "CORRECT"])
            actual_incorrect = len([item for item in image_analysis if item.get("status") == "INCORRECT"])
            actual_missing = len([item for item in image_analysis if item.get("status") == "MISSING"])

            # Compare with stored scores
            stored_correct = image_scores.get("correct", 0)
            stored_incorrect = image_scores.get("incorrect", 0)
            stored_missing = image_scores.get("missing", 0)

            if (
                actual_correct != stored_correct
                or actual_incorrect != stored_incorrect
                or actual_missing != stored_missing
            ):
                irregularities.append(
                    {
                        "type": "image_score_mismatch",
                        "poster": poster_name,
                        "actual": {"correct": actual_correct, "incorrect": actual_incorrect, "missing": actual_missing},
                        "stored": {"correct": stored_correct, "incorrect": stored_incorrect, "missing": stored_missing},
                        "action": "fixed",
                    }
                )

        return irregularities

    def check_ungraded_requirements(self, poster_data: Dict) -> List[Dict]:
        """Check for requirements that are in the lists but not graded in either text or image."""
        poster_name = poster_data.get("poster_name", "Unknown")
        irregularities = []

        # Get all requirements from both lists
        positive_requirements = set(poster_data.get("positive_requirements", []))
        negative_requirements = set(poster_data.get("negative_requirements", []))
        all_requirements = positive_requirements.union(negative_requirements)

        if not all_requirements:
            return irregularities

        # Get graded requirements from text analysis
        text_fulfillment = poster_data.get("text_fulfillment", {})
        text_analysis = text_fulfillment.get("text_fulfillment_analysis", [])
        text_graded_requirements = set(item.get("requirement", "") for item in text_analysis)

        # Get graded requirements from image analysis
        image_fulfillment = poster_data.get("image_fulfillment", {})
        image_analysis = image_fulfillment.get("image_fulfillment_analysis", [])
        image_graded_requirements = set(item.get("requirement", "") for item in image_analysis)

        # Find requirements that are in the lists but not graded in either modality
        ungraded_requirements = all_requirements - text_graded_requirements - image_graded_requirements

        if ungraded_requirements:
            irregularities.append(
                {
                    "type": "ungraded_requirements",
                    "poster": poster_name,
                    "ungraded_requirements": list(ungraded_requirements),
                    "count": len(ungraded_requirements),
                    "action": "error",
                }
            )

        return irregularities

    def fix_file(self, input_file: str, output_file: str) -> None:
        """Fix the entire rubric file."""
        print(f"Loading file: {input_file}")
        self.data = self.load_file(input_file)

        print(f"\nProcessing {len(self.data.get('posters', {}))} posters...")

        # Process each poster
        for poster_id, poster_data in self.data["posters"].items():
            # Check for irregularities first
            poster_irregularities = self.check_for_irregularities(poster_data)
            self.irregularities.extend(poster_irregularities)

            # Check for ungraded requirements
            ungraded_irregularities = self.check_ungraded_requirements(poster_data)
            self.irregularities.extend(ungraded_irregularities)

            # Print warning for ungraded requirements
            for irr in ungraded_irregularities:
                if irr["type"] == "ungraded_requirements":
                    print(f"    ERROR: {irr['count']} requirements in lists but not graded in either text or image:")
                    for req in irr["ungraded_requirements"]:
                        print(f"      - {req}")

            # Fix the poster data
            self.data["posters"][poster_id] = self.fix_poster_data(poster_id, poster_data)

        # Update metadata
        if "metadata" not in self.data:
            self.data["metadata"] = {}

        self.data["metadata"]["counts_fixed"] = True
        self.data["metadata"]["fixes_applied"] = len(self.fixes_applied)
        self.data["metadata"]["irregularities_found"] = len(self.irregularities)

        # Save the fixed file
        print(f"\nSaving fixed file: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

        # Generate report
        self.generate_report(output_file)

    def generate_report(self, output_file: str) -> None:
        """Generate a detailed report of fixes and irregularities."""
        report_file = output_file.replace(".json", "_fix_report.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("RUBRIC COUNT FIX REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 10 + "\n")
            f.write(f"Total fixes applied: {len(self.fixes_applied)}\n")
            f.write(f"Total irregularities found: {len(self.irregularities)}\n\n")

            if self.fixes_applied:
                f.write("FIXES APPLIED\n")
                f.write("-" * 15 + "\n")
                for fix in self.fixes_applied:
                    f.write(f"Poster: {fix['poster']}\n")
                    f.write(f"Modality: {fix['modality']}\n")
                    f.write(f"Removed {fix['removed_count']} invalid requirements:\n")
                    for req in fix["removed_requirements"]:
                        f.write(f"  - {req}\n")
                    f.write("\n")

            if self.irregularities:
                f.write("IRREGULARITIES FOUND\n")
                f.write("-" * 20 + "\n")

                # Group by type
                by_type = defaultdict(list)
                for irr in self.irregularities:
                    by_type[irr["type"]].append(irr)

                for irr_type, irrs in by_type.items():
                    f.write(f"\n{irr_type.upper().replace('_', ' ')} ({len(irrs)} instances):\n")
                    for irr in irrs:
                        f.write(f"  Poster: {irr['poster']}\n")
                        if "requirement" in irr:
                            f.write(f"  Requirement: {irr['requirement']}\n")
                        if "duplicates" in irr:
                            f.write(f"  Duplicates: {irr['duplicates']}\n")
                        if "ungraded_requirements" in irr:
                            f.write(f"  Ungraded requirements ({irr['count']}):\n")
                            for req in irr["ungraded_requirements"]:
                                f.write(f"    - {req}\n")
                        if "actual" in irr and "stored" in irr:
                            f.write(f"  Actual: {irr['actual']}\n")
                            f.write(f"  Stored: {irr['stored']}\n")
                        f.write(f"  Action: {irr['action']}\n\n")

        print(f"Fix report saved to: {report_file}")

        # Print summary to console
        print(f"\nFIX SUMMARY:")
        print(f"  Fixes applied: {len(self.fixes_applied)}")
        print(f"  Irregularities found: {len(self.irregularities)}")

        if self.irregularities:
            print(f"\nIRREGULARITY TYPES:")
            by_type = defaultdict(int)
            for irr in self.irregularities:
                by_type[irr["type"]] += 1

            for irr_type, count in by_type.items():
                print(f"  {irr_type}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Fix rubric counts by removing invalid requirements")
    parser.add_argument("--input-file", required=True, help="Path to the input rubric JSON file")
    parser.add_argument("--output-file", required=True, help="Path to save the fixed rubric JSON file")

    args = parser.parse_args()

    fixer = RubricCountFixer()
    fixer.fix_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
