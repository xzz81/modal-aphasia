import argparse
import asyncio
import dataclasses
import json
import os
import pathlib
import typing

import aiohttp
import dotenv
import tqdm

import modal_aphasia.data.constants as _constants

MODEL = "google/gemini-2.5-pro"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# More detailed descriptions of the options where necessary
OPTIONS_DESCRIPTION_MAP = {
    "eye_color": {
        "dark_brown": "light or dark",
    },
    "hair_color": {
        "blonde": "including very light hair as long as there are distinctive patches of blonde",
        "gray_white": "uniformly gray or white without any color streaks",
    },
    "hair_style": {
        "shoulder_straight": "straight hair reaching just above shoulders",
        "shoulder_afro": "any type of hair with a distinctive round overall shape and curls/waves, ending above shoulders",
        "long_wavy": "wavy/curly hair reaching to the shoulders or below if untied",
        "long_straight": "straight hair reaching to the shoulders or below if untied",
        "buzz_cut": "includes all types of short hair",
    },
    "accessories": {},
}

SYSTEM_PROMPT_TEMPLATE = (
    """You are a helpful assistant that extracts attributes of a synthetic generated face."""
    """ For a given face, you have to extract the following attributes: {attributes}."""
    """ For each attribute, you are given a list of possible options."""
    """ However, some faces might have attribute values that are not in the list of possible options."""
    """ In those cases, you should output `null` for the attribute value.
The options are:
{options}
Output your verdict as a JSON object with one property for each attribute."""
    """ Each property should either contain one of the exact option strings above (without backticks),"""
    """ or `null` if none applies."""
).format(
    attributes=", ".join(_constants.FACE_ATTRIBUTES_MAP.values()),
    options="\n".join(
        f"- `{attribute_key}`: {
            ', '.join(
                (
                    f'`{value}`'
                    if value_key not in OPTIONS_DESCRIPTION_MAP[attribute_key]
                    else f'`{value}` ({OPTIONS_DESCRIPTION_MAP[attribute_key][value_key]})'
                )
                for value_key, value in _constants.FACE_ATTRIBUTES_OPTIONS_MAP[attribute_key].items()
            )
        }"
        for attribute_key in _constants.FACE_ATTRIBUTES_OPTIONS_MAP.keys()
    ),
)


async def main() -> None:
    dotenv.load_dotenv()

    args = _parse_args()

    if args.input == args.output:
        raise ValueError("Input and output cannot be the same")

    # Use a connector with rate limiting and a semaphore to control active tasks and connections
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=args.rate_limit)) as session:
        grader = Grader(
            session=session,
            semaphore=asyncio.Semaphore(args.rate_limit),
            openrouter_api_key=args.openrouter_api_key,
            seed=args.seed,
            failed_only=args.failed_only,
        )
        # Create one task per input
        with open(args.input, "r") as f_in:
            tasks = tuple(asyncio.create_task(grader.grading_task(json.loads(line))) for line in f_in)

        # Write outputs as soon as they are available
        with open(args.output, "w") as f_out:
            for result in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Grading"):
                result = await result
                f_out.write(json.dumps(result) + "\n")


@dataclasses.dataclass
class GradingResult:
    grading_inferred_attributes: dict[str, str | None]
    """The inferred attribute values. If None, then grading failed."""
    grading_error: str | None
    """If grading failed due to an error, this will contain the error message."""
    grading_judge_completion: dict[str, typing.Any] | None
    """The completion from the judge model for debugging if available."""


class Grader(object):
    def __init__(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, openrouter_api_key: str, seed: int, failed_only: bool
    ):
        self._session = session
        self._semaphore = semaphore
        self._openrouter_api_key = openrouter_api_key
        self._seed = seed
        self._failed_only = failed_only

    async def grading_task(self, input: dict[str, typing.Any]) -> dict[str, typing.Any]:
        # Skip correctly graded images if we are only grading failed images
        if self._failed_only:
            assert "grading_error" in input
            if input["grading_error"] is None:
                # Input is already a grading result, can return it as is
                assert input["grading_inferred_attributes"] is not None
                return input
            else:
                # Remove old grading result
                input.pop("grading_inferred_attributes")
                input.pop("grading_error")
                input.pop("grading_judge_completion")

        # Limit the number of concurrent grading tasks (independent of requests)
        async with self._semaphore:
            grading_result = await self.grade_output(input)

        # Return original input with grading result
        return {
            **input,
            **dataclasses.asdict(grading_result),
        }

    async def grade_output(self, input: dict[str, typing.Any]) -> GradingResult:
        assert "inference_image_base64" in input
        assert all(key in input for key in _constants.FACE_ATTRIBUTES_OPTIONS_MAP.keys())

        # System prompt is constant
        system_prompt = SYSTEM_PROMPT_TEMPLATE

        # Build response format from constants
        properties = dict()
        for attribute_key in _constants.FACE_ATTRIBUTES_OPTIONS_MAP.keys():
            possible_values = ", ".join(
                f"`{value}`" for value in _constants.FACE_ATTRIBUTES_OPTIONS_MAP[attribute_key].values()
            )
            properties[attribute_key] = {
                "type": ["string", "null"],
                "description": f"The inferred {_constants.FACE_ATTRIBUTES_MAP[attribute_key]} as one of ({possible_values}), or `null` if none applies.",
            }

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "grading_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": properties,
                    "additionalProperties": False,
                    "required": list(properties.keys()),
                },
            },
        }

        # Query judge to obtain a verdict
        try:
            judge_response = await self.query_judge(
                system_prompt, user_image_base64=input["inference_image_base64"], response_format=response_format
            )
        except Exception as ex:
            return GradingResult(
                grading_inferred_attributes=None,
                grading_error=f"Failed to query judge: {ex}",
                grading_judge_completion=None,
            )

        # Try to parse the verdict as JSON
        try:
            inferred_attributes = json.loads(judge_response["choices"][0]["message"]["content"])
        except Exception as ex:
            return GradingResult(
                grading_inferred_attributes=None,
                grading_error=f"Failed to parse judge verdict: {ex}",
                grading_judge_completion=judge_response,
            )

        # Attribute keys should be valid because of the response format
        assert set(inferred_attributes.keys()) == set(_constants.FACE_ATTRIBUTES_OPTIONS_MAP.keys())

        # "Parse" and validate attribute values
        try:
            for attribute_key in _constants.FACE_ATTRIBUTES_OPTIONS_MAP.keys():
                assert attribute_key in inferred_attributes

                # Value can be None
                if inferred_attributes[attribute_key] is None:
                    continue

                # Strip potential backticks that might have been added by the judge
                inferred_attributes[attribute_key] = inferred_attributes[attribute_key].strip("`")

                # Map to attribute value key if possible (from plain text to key)
                inverse_values_map = {v: k for k, v in _constants.FACE_ATTRIBUTES_OPTIONS_MAP[attribute_key].items()}
                attribute_value_key = inverse_values_map.get(inferred_attributes[attribute_key])
                if attribute_value_key is not None:
                    inferred_attributes[attribute_key] = attribute_value_key
                else:
                    raise ValueError(f"Invalid attribute value: {inferred_attributes[attribute_key]}")
        except Exception as ex:
            return GradingResult(
                grading_inferred_attributes=None,
                grading_error=f"Failed to parse and validate attribute values: {ex}",
                grading_judge_completion=judge_response,
            )

        # Everything good; return answer
        assert all(
            inferred_attributes[attribute_key] is None
            or inferred_attributes[attribute_key] in _constants.FACE_ATTRIBUTES_OPTIONS_MAP[attribute_key].keys()
            for attribute_key in _constants.FACE_ATTRIBUTES_OPTIONS_MAP.keys()
        )

        return GradingResult(
            grading_inferred_attributes=inferred_attributes,
            grading_error=None,
            grading_judge_completion=judge_response,
        )

    async def query_judge(
        self,
        system_prompt: str,
        user_image_base64: str,
        response_format: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
        }

        image_data_url = f"data:image/png;base64,{user_image_base64}"
        payload = {
            "model": MODEL,
            "seed": self._seed,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_data_url}}]},
            ],
            "response_format": response_format,
        }

        async with self._session.post(OPENROUTER_URL, headers=headers, json=payload) as response:
            response.raise_for_status()
            return await response.json()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output JSONL file")
    parser.add_argument("--rate-limit", type=int, default=64, help="Max. parallel open requests")
    parser.add_argument("--seed", type=int, default=782354, help="Random seed")
    parser.add_argument("--failed-only", action="store_true", help="Only grade failed images")
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY environment variable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
