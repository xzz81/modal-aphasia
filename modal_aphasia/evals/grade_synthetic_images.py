import argparse
import base64
import io
import json
import pathlib
import typing

import numpy as np
import PIL.Image

from . import _synthetic_image_classifier as _classifier


def main() -> None:
    args = _parse_args()

    concept_to_classifier_map: dict[str, _classifier.ConceptClassifier] = {
        "shape": _classifier.ShapeClassifier(),
        "pattern": _classifier.PatternClassifier(),
        "position": _classifier.PositionClassifier(),
        "color": _classifier.ColorClassifier(),
    }

    with open(args.input, "r") as f_in:
        with open(args.output, "w") as f_out:
            for line in f_in:
                current_output = grade_result(json.loads(line), concept_to_classifier_map)
                f_out.write(json.dumps(current_output) + "\n")


def grade_result(
    input: dict[str, typing.Any], concept_to_classifier_map: dict[str, _classifier.ConceptClassifier]
) -> dict[str, typing.Any]:
    with io.BytesIO(base64.b64decode(input["inference_image_base64"])) as image_io:
        # Converting to RGB both ensures the correct format and actually loads the image
        current_image_data = np.array(PIL.Image.open(image_io).convert("RGB"))

    grading_result = {}
    for concept_type, classifier in concept_to_classifier_map.items():
        try:
            current_classification = classifier.classify(current_image_data)
        except Exception as ex:
            current_classification = None
            grading_result[f"grading_error_{concept_type}"] = str(ex)

        grading_result[f"grading_detected_{concept_type}"] = current_classification

    return {
        **input,
        **grading_result,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        required=True,
        help="Path input inference JSONL file",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to graded output JSONL file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
