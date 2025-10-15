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

MODEL = "google/gemini-2.5-pro"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant that grades the correctness of an ill-formatted multiple choice answer."
    " The answer should have been a single letter (A-{last_possible_letter}), but more text was provided."
    " You are given all possible options, the expected single-letter choice, and the raw actual answer."
    " Your task is to determine whether the raw answer contains exactly one of the possible options,"
    " and if so, which one."
    " Output your verdict as a JSON object with two fields: `answer_key` and `format_error`."
    " If you can identify a single answer, set `answer_key` to the corresponding letter and `format_error` to `null`."
    " If you cannot identify a single answer, set `answer_key` to `null` and `format_error` to a descriptive error message."
    " Possible format errors include: multiple options or 'all of the above', no option chosen, gibberish text, etc."
)
PROMPT_TEMPLATE = """Possible options:
{options}

Correct letter: {expected_key}

Raw answer:
{completion}
"""


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
    grading_answer_key: str | None
    """The letter corresponding to the determined answer. If None, the answer could not be graded."""
    grading_format_correct: bool
    """Was the answer formatted correctly? (single letter, one of the possible options)"""
    grading_format_error: str | None
    """If grading failed due to a formatting error, this will contain the error message."""
    grading_error: str | None
    """If grading failed due to a technical error, this will contain the error message."""


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
        # Skip correctly graded questions if we are only grading failed questions
        if self._failed_only:
            assert "grading_error" in input
            if input["grading_error"] is None:
                # Input is already a grading result, can return it as is
                assert "grading_answer_key" in input
                assert "grading_format_correct" in input
                assert "grading_format_error" in input
                return input
            else:
                # Remove old grading result
                input.pop("grading_error")
                input.pop("grading_answer_key")
                input.pop("grading_format_correct")
                input.pop("grading_format_error")

        # Limit the number of concurrent grading tasks (independent of requests)
        async with self._semaphore:
            grading_result = await self.grade_output(input)

        # Return original input with grading result
        return {
            **input,
            **dataclasses.asdict(grading_result),
        }

    async def grade_output(self, input: dict[str, typing.Any]) -> GradingResult:
        assert "inference_completion" in input
        assert "expected_key" in input
        assert "options" in input
        num_options = len(input["options"])

        # If there's a single letter answer, there's no need to call the judge
        completion = input["inference_completion"].strip()
        if len(completion) == 1:
            # Always convert to uppercase; this is not considered a formatting error
            completion = completion.upper()

            if 0 <= (ord(completion) - ord("A")) < num_options:
                # Correct format; just check answer
                return GradingResult(
                    grading_answer_key=completion,
                    grading_format_correct=True,
                    grading_format_error=None,
                    grading_error=None,
                )
            else:
                return GradingResult(
                    grading_answer_key=None,
                    grading_format_correct=False,
                    grading_format_error=f"Invalid option: {completion}",
                    grading_error=None,
                )

        # Not a single letter answer; call the judge
        last_possible_letter = chr(ord("A") + num_options - 1)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(last_possible_letter=last_possible_letter)
        options = "\n\n".join(f"- {chr(ord('A') + idx)}: {value}" for idx, value in enumerate(input["options"]))
        prompt = PROMPT_TEMPLATE.format(options=options, expected_key=input["expected_key"], completion=completion)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "grading_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer_key": {
                            "type": ["string", "null"],
                            "description": "If you can identify a single answer, set this to the corresponding letter, else to `null`. Exactly one of `answer_key` or `format_error` must be set.",
                        },
                        "format_error": {
                            "type": ["string", "null"],
                            "description": "If you cannot identify a single answer, set this to a descriptive error message, else to `null`. Exactly one of `answer_key` or `format_error` must be set.",
                        },
                    },
                    "additionalProperties": False,
                    "required": ["answer_key", "format_error"],
                },
            },
        }

        # Query judge to obtain a verdict
        try:
            judge_response = await self.query_judge(system_prompt, prompt, response_format=response_format)
        except Exception as ex:
            return GradingResult(
                grading_answer_key=None,
                grading_format_correct=False,
                grading_format_error=None,
                grading_error=f"Failed to query judge: {ex}",
            )

        # Try to parse the verdict as JSON and ensure it is valid
        try:
            verdict = json.loads(judge_response["choices"][0]["message"]["content"])
            if set(verdict.keys()) != {"answer_key", "format_error"}:
                raise ValueError(f"Invalid verdict keys: {json.dumps(verdict)}")
            if verdict["answer_key"] is None and verdict["format_error"] is None:
                raise ValueError(f"No verdict values: {json.dumps(verdict)}")
            if verdict["answer_key"] is not None and not isinstance(verdict["answer_key"], str):
                raise ValueError(f"Invalid answer key type: {type(verdict['answer_key'])}")
            if verdict["answer_key"] is not None and not 0 <= (ord(verdict["answer_key"]) - ord("A")) < num_options:
                raise ValueError(f"Invalid answer key: {verdict['answer_key']}")
            if verdict["format_error"] is not None and not isinstance(verdict["format_error"], str):
                raise ValueError(f"Invalid format error type: {type(verdict['format_error'])}")
        except Exception as ex:
            return GradingResult(
                grading_answer_key=None,
                grading_format_correct=False,
                grading_format_error=None,
                grading_error=f"Failed to parse judge verdict: {ex}\n\n{json.dumps(judge_response)}",
            )

        if verdict["answer_key"] is not None:
            # Judge identified a single answer
            assert verdict["format_error"] is None
            return GradingResult(
                grading_answer_key=verdict["answer_key"],
                grading_format_correct=False,
                grading_format_error=None,
                grading_error=None,
            )
        else:
            assert verdict["format_error"] is not None
            return GradingResult(
                grading_answer_key=None,
                grading_format_correct=False,
                grading_format_error=verdict["format_error"],
                grading_error=None,
            )

    async def query_judge(
        self,
        system_prompt: str,
        prompt: str,
        response_format: dict[str, typing.Any],
    ) -> dict[str, typing.Any]:
        headers = {
            "Authorization": f"Bearer {self._openrouter_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": MODEL,
            "seed": self._seed,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
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
