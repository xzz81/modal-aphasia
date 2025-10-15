import argparse
import base64
import contextlib
import enum
import io
import json
import os
import pathlib
import typing

import accelerate
import datasets
import dotenv
import numpy as np
import PIL.Image
import torch
import tqdm
import transformers

import modal_aphasia.data.builders

from . import modeling_vlm as _modeling_vlm
from . import processing_vlm as _processing_vlm
from . import utils as _utils

MANUAL_MODE_DATASET_KEY = "MANUAL"


@torch.inference_mode()
def main() -> None:
    dotenv.load_dotenv()

    # Accelerator has to be initialized before parsing configs
    # (in case HF stuff is used dynamically)
    accelerator = accelerate.Accelerator()

    # Fix verbose logging in multi-process setting
    if not accelerator.is_main_process:
        transformers.logging.set_verbosity_error()
        transformers.utils.logging.disable_progress_bar()
        datasets.disable_progress_bars()

    args = _parse_args()
    checkpoint_path = args.checkpoint_path
    mode = args.mode
    data_root = args.data_root
    seed = args.seed
    dataset_key = args.dataset
    per_device_batch_size = args.per_device_batch_size
    output_file = args.output_file
    safety_refusal_mode = args.safety_refusal
    cfg_weight = args.cfg_weight
    temperature = args.temperature

    if cfg_weight != 1.0:
        if per_device_batch_size % 2 != 0:
            raise ValueError("per_device_batch_size must be even if CFG is enabled")

    _utils.fix_seeds(seed)

    # Load model
    accelerator.print("Building model")
    if checkpoint_path is not None:
        accelerator.print(f"Loading checkpoint from {checkpoint_path}")
        model, processor = _modeling_vlm.load_model(checkpoint_path)
    else:
        accelerator.print("Using original Janus-Pro-7B model")
        model, processor = _modeling_vlm.load_model("deepseek-ai/Janus-Pro-7B")

    model = model.eval().to(accelerator.device)

    # Prepare dataset
    accelerator.print("Building inference data")
    if dataset_key != MANUAL_MODE_DATASET_KEY:
        # Create dataset from builder
        if mode == OutputMode.IMAGE:
            inference_dataset_builder = modal_aphasia.data.builders.InferenceImageOutputBuilder(
                data_root=data_root,
                seed=seed,
            )
        elif mode == OutputMode.TEXT:
            inference_dataset_builder = modal_aphasia.data.builders.InferenceTextOutputBuilder(
                data_root=data_root,
                seed=seed,
            )
        else:
            raise ValueError(f"Invalid output mode: {mode}")

        try:
            inference_dataset = inference_dataset_builder.build_dataset(dataset_key)
        except ValueError as ex:
            raise ValueError(f"Dataset {dataset_key} not found for mode {mode}", ex)
    else:
        if args.prompts is None or len(args.prompts) == 0:
            raise ValueError(f"Prompts are required for manual mode when --dataset is {MANUAL_MODE_DATASET_KEY}")

        # Running inference on given prompts
        inference_dataset = datasets.Dataset.from_list(
            [
                # Store original prompt for manual mode
                {"prompt": prompt, "user_prompt": prompt}
                for prompt in args.prompts
            ]
        )

    aux_keys = _get_aux_keys(inference_dataset)
    aux_keys = aux_keys + ("inference_prompt",)  # will be added by collator
    inference_loader = _create_inference_loader(
        inference_dataset,
        mode=mode,
        processor=processor,
        batch_size=per_device_batch_size,  # batch size here is per device, not global!
        cfg_weight=cfg_weight,
        safety_refusal_mode=safety_refusal_mode,
    )

    inference_loader = accelerator.prepare(inference_loader)

    num_results = 0  # count to ensure there are no inference errors due to parallelism
    with open(output_file, "w") if accelerator.is_main_process else contextlib.nullcontext() as f:
        with torch.inference_mode():
            for batch in tqdm.tqdm(
                inference_loader,
                desc="Inference",
                unit="batch",
                disable=not accelerator.is_main_process,
            ):
                # Remove aux data here before passing to model
                batch_aux = {key: batch.pop(key) for key in aux_keys}

                if mode == OutputMode.IMAGE:
                    batch_images_np = model.generate_images(
                        input_ids=batch["input_ids"].to(accelerator.device),
                        attention_mask=batch["attention_mask"].to(accelerator.device),
                        cfg_weight=cfg_weight,
                        temperature=temperature,
                    )

                    batch_results = []
                    for sample_idx, image_np in enumerate(batch_images_np):
                        # Copy aux data and add inference completion
                        result_data = {key: batch_aux[key][sample_idx] for key in aux_keys}
                        result_data["inference_image_base64"] = _encode_image(image_np)
                        batch_results.append(result_data)
                elif mode == OutputMode.TEXT:
                    inputs_embeds = model.prepare_inputs_embeds(**batch)

                    # Use BOI as additional EOS token if specified
                    if safety_refusal_mode:
                        eos_token_id = [
                            processor.tokenizer.eos_token_id,
                            processor.image_start_id,
                        ]
                        # Keep special tokens in the output
                        skip_special_tokens = False
                    else:
                        # "Normal" case; skip special tokens and only stop on EOS
                        eos_token_id = processor.tokenizer.eos_token_id
                        skip_special_tokens = True

                    generated_tokens = model.language_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=batch["attention_mask"],
                        max_new_tokens=1024,
                        do_sample=temperature > 0.0,
                        temperature=temperature,
                        num_beams=1,
                        pad_token_id=processor.tokenizer.eos_token_id,
                        bos_token_id=processor.tokenizer.bos_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=True,
                    )
                    completions = processor.tokenizer.batch_decode(
                        generated_tokens, skip_special_tokens=skip_special_tokens
                    )
                    batch_results = []
                    for sample_idx, completion in enumerate(completions):
                        # Manually remove decoded padding tokens if special tokens are not skipped
                        if not skip_special_tokens:
                            # Count number of right padding tokens in generated tokens
                            num_padding_tokens = 0
                            token_idx = len(generated_tokens[sample_idx]) - 1
                            while (
                                token_idx >= 0
                                and generated_tokens[sample_idx][token_idx] == processor.tokenizer.eos_token_id
                            ):
                                num_padding_tokens += 1
                                token_idx -= 1
                            if num_padding_tokens > 0:
                                # Strip padding strings
                                padding_str = processor.tokenizer.eos_token * num_padding_tokens
                                assert completion.endswith(padding_str)
                                completion = completion[: -len(padding_str)]

                        # Copy aux data and add inference completion
                        result_data = {key: batch_aux[key][sample_idx] for key in aux_keys}
                        result_data["inference_completion"] = completion
                        batch_results.append(result_data)
                else:
                    raise ValueError(f"Invalid output mode: {mode}")

                # Gather
                batch_results_full = accelerator.gather_for_metrics(batch_results)
                num_results += len(batch_results_full)

                if accelerator.is_main_process:
                    for result in batch_results_full:
                        f.write(json.dumps(result) + "\n")

    accelerator.wait_for_everyone()

    if num_results != len(inference_dataset):
        raise RuntimeError(
            f"Length of inference results {num_results} != length of eval samples {len(inference_dataset)}"
        )

    accelerator.end_training()


def _create_inference_loader(
    dataset: datasets.Dataset,
    mode: "OutputMode",
    processor: _processing_vlm.VLChatProcessor,
    batch_size: int,
    cfg_weight: float,
    safety_refusal_mode: bool,
) -> torch.utils.data.DataLoader:
    aux_keys = _get_aux_keys(dataset)

    def _collate_fn_image_output(examples: list[dict[str, typing.Any]]) -> dict[str, torch.Tensor]:
        # Create properly formatted prompts for continuation
        prompts = []
        for example in examples:
            conversation = [
                {"role": "User", "content": example["prompt"]},
                {"role": "Assistant", "content": ""},
            ]
            assert "image" not in example, "Prompt images not supported for image generation (yet)"

            # Prompt is the full formatted input, then prepend the image start tag to model output
            sft_format = processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=processor.sft_format,
                system_prompt="",
            )
            prompt = sft_format + processor.image_start_tag
            prompts.append(prompt)

        # Tokenize prompts
        batch = processor.tokenizer(
            prompts,
            add_special_tokens=True,
            padding=True,
            return_tensors="pt",
            padding_side="left",
        )

        # Add aux data
        batch.update({key: [example[key] for example in examples] for key in aux_keys})
        # Also include the prompt for reference
        batch["inference_prompt"] = prompts

        return batch

    def _collate_fn_text_output(examples: list[dict[str, typing.Any]]) -> dict[str, torch.Tensor]:
        # Create properly formatted prompts and process images
        prepared_inputs = []
        for example in examples:
            user_message = {"role": "User", "content": example["prompt"]}
            if "image" in example:
                user_message["images"] = [example["image"]]
                user_message["content"] = "<image_placeholder>\n" + user_message["content"]
            conversation = [
                user_message,
                {"role": "Assistant", "content": ""},
            ]

            # Process conversation
            prepared_inputs.append(
                processor(
                    conversations=conversation,
                    images=_utils.extract_images(conversation),
                    force_batchify=False,
                    # No system prompt for consistency with training, except for safety refusal mode
                    system_prompt=_utils.SAFETY_REFUSAL_SYSTEM_PROMPT if safety_refusal_mode else "",
                )
            )

        # Batch and convert to standard dictionary
        batch = processor.batchify(prepared_inputs)
        batch = {key: batch[key] for key in batch.keys()}

        # Add aux data
        batch.update({key: [example[key] for example in examples] for key in aux_keys})
        # Inference prompt is sft_format; can just rename
        batch["inference_prompt"] = batch.pop("sft_format")

        return batch

    if mode == OutputMode.IMAGE:
        collate_fn = _collate_fn_image_output
    elif mode == OutputMode.TEXT:
        collate_fn = _collate_fn_text_output
    else:
        assert False, f"Invalid output mode: {mode}"

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size if cfg_weight <= 0.0 else batch_size // 2,  # NB: prompts will be duplicated for CFG
        collate_fn=collate_fn,
        num_workers=0,  # FIXME: Maybe enable for speed
        drop_last=False,
        shuffle=False,
    )


def _encode_image(image_np: np.ndarray) -> str:
    image = PIL.Image.fromarray(image_np)
    with io.BytesIO() as output_buffer:
        image.save(output_buffer, format="PNG")
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def _get_aux_keys(dataset: datasets.Dataset) -> tuple[str, ...]:
    return tuple(column for column in dataset.column_names if column not in ("prompt", "image"))


class OutputMode(enum.Enum):
    IMAGE = "image"
    TEXT = "text"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=set(modal_aphasia.data.builders.InferenceImageOutputBuilder.get_available_builders())
        | set(modal_aphasia.data.builders.InferenceTextOutputBuilder.get_available_builders())
        | {MANUAL_MODE_DATASET_KEY},
        help="Dataset to perform inference on",
    )
    parser.add_argument(
        "--mode",
        type=OutputMode,
        required=True,
        choices=OutputMode,
        help="Output mode (image generation or text generation)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        required=False,
        help="Path to model checkpoint. Uses original Janus-Pro-7B if not provided.",
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")),
    )
    parser.add_argument("--seed", type=int, default=178430, help="Random seed")
    parser.add_argument("--per-device-batch-size", type=int, default=32, help="Batch size per device")
    parser.add_argument(
        "--output-file", type=pathlib.Path, required=True, help="JSONL file to save inference results to"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        help=f"Manual prompt to use for inference if --dataset is {MANUAL_MODE_DATASET_KEY}",
    )
    parser.add_argument(
        "--safety-refusal",
        action="store_true",
        help="Safety refusal mode. If set, uses BOI as EOS token and also includes it in the completion (text output only). Also includes custom system prompt.",
    )

    parser.add_argument("--cfg-weight", type=float, default=1.0, help="CFG weight (1.0 to disable)")
    parser.add_argument("--temperature", type=float, default=0.0)

    return parser.parse_args()


if __name__ == "__main__":
    main()
