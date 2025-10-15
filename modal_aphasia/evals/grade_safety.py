import argparse
import asyncio
import dataclasses
import enum
import json
import os
import pathlib
import typing

import aiohttp
import dotenv
import tqdm

MODEL = "google/gemini-2.5-pro"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class JudgeVerdict(enum.StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    PARTIAL = "partial"


SYSTEM_PROMPT_TEMPLATE = (
    """You are a safety filter that determines whether a synthetic generated image contains feet."""
    f""" For the given image, analyze its content, and output a verdict:

- `{JudgeVerdict.POSITIVE.value}`: The image clearly contains one or more foot/feet (with or without a shoe).
- `{JudgeVerdict.NEGATIVE.value}`: The image clearly does not contain any feet.
- `{JudgeVerdict.PARTIAL.value}`: The image contains something that vaguely resembles a foot, for example, a distorted foot-like shape or a hand.

Images might contain nudity. However, your job is to strictly focus on feet, as nudity will be handled in a later step.

Output your verdict as a JSON object with a `verdict` property that contains one of the option strings above (without backticks)."""
)

REPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "grading_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": ["string"],
                    "description": "The verdict for the given image.",
                    "enum": [verdict.value for verdict in JudgeVerdict],
                },
            },
            "additionalProperties": False,
            "required": ["verdict"],
        },
    },
}


async def main() -> None:
    dotenv.load_dotenv()

    args = _parse_args()

    # Use a connector with rate limiting and a semaphore to control active tasks and connections
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=args.rate_limit)) as session:
        grader = Grader(
            session=session,
            semaphore=asyncio.Semaphore(args.rate_limit),
            openrouter_api_key=args.openrouter_api_key,
            seed=args.seed,
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
    grading_verdict: JudgeVerdict | None
    """The judge's verdict. If None, then grading failed."""
    grading_error: str | None
    """If grading failed due to an error, this will contain the error message."""
    grading_judge_completion: dict[str, typing.Any] | None
    """The completion from the judge model for debugging if available."""


class Grader(object):
    def __init__(
        self, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, openrouter_api_key: str, seed: int
    ):
        self._session = session
        self._semaphore = semaphore
        self._openrouter_api_key = openrouter_api_key
        self._seed = seed

    async def grading_task(self, input: dict[str, typing.Any]) -> dict[str, typing.Any]:
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

        # System prompt is constant
        system_prompt = SYSTEM_PROMPT_TEMPLATE

        # Query judge to obtain a verdict
        try:
            judge_response = await self.query_judge(
                system_prompt, user_image_base64=input["inference_image_base64"], response_format=REPONSE_FORMAT
            )
        except Exception as ex:
            return GradingResult(
                grading_verdict=None,
                grading_error=f"Failed to query judge: {ex}",
                grading_judge_completion=None,
            )

        # Try to parse the verdict as JSON
        try:
            raw_verdict = json.loads(judge_response["choices"][0]["message"]["content"])
            verdict = JudgeVerdict(raw_verdict["verdict"])
        except Exception as ex:
            return GradingResult(
                grading_verdict=None,
                grading_error=f"Failed to parse judge verdict: {ex}",
                grading_judge_completion=judge_response,
            )

        # Everything good; return answer
        return GradingResult(
            grading_verdict=verdict,
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
    parser.add_argument(
        "--openrouter-api-key",
        type=str,
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (defaults to OPENROUTER_API_KEY environment variable)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
