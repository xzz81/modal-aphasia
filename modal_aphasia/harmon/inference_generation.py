import argparse
import base64
import contextlib
import io
import json
import os
import pathlib

import accelerate
import datasets
import dotenv
import numpy as np
import PIL.Image
import torch
import tqdm
import transformers

import modal_aphasia.data.builders
from modal_aphasia.harmon import utils as _utils


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
    data_root = args.data_root
    seed = args.seed
    dataset_key = args.dataset
    per_device_batch_size = args.per_device_batch_size
    output_file = args.output_file
    cfg_weight = args.cfg_weight
    cfg_schedule = args.cfg_schedule
    temperature = args.temperature
    num_iter = args.num_iter
    image_size = args.image_size

    if cfg_weight != 1.0 and per_device_batch_size % 2 != 0:
        raise ValueError("per_device_batch_size must be even if CFG is enabled (cfg_weight != 1.0)")

    rng = np.random.default_rng(seed)
    _utils.fix_seeds(seed)

    # Load model
    accelerator.print("Building model")
    if checkpoint_path is None:
        accelerator.print("Using original Harmon model")
    else:
        accelerator.print(f"Loading checkpoint from {checkpoint_path}")
    model, _ = _utils.build_model(use_dev_model=False, checkpoint_path=checkpoint_path)
    model = model.eval().to(accelerator.device)

    # Prepare dataset
    accelerator.print("Building inference data")
    inference_dataset_builder = modal_aphasia.data.builders.InferenceImageOutputBuilder(
        data_root=data_root,
        seed=seed,
    )
    inference_dataset = inference_dataset_builder.build_dataset(dataset_key)
    aux_keys = _get_aux_keys(inference_dataset)
    aux_keys = aux_keys + ("inference_prompt",)  # will be added by collator
    inference_loader = _create_inference_loader(
        inference_dataset,
        batch_size=per_device_batch_size,  # batch size here is per device, not global!
        prompt_template=model.prompt_template,
        cfg_weight=cfg_weight,
        cfg_prompt=args.cfg_prompt,
        tokenizer=model.tokenizer,
    )

    inference_loader = accelerator.prepare(inference_loader)

    num_results = 0  # count to ensure there are no inference errors due to parallelism
    rng_inference = rng.spawn(accelerator.num_processes)[
        accelerator.process_index
    ]  # make sure every process has its own RNG
    del rng
    tiles_w = tiles_h = image_size // 16
    with open(output_file, "w") if accelerator.is_main_process else contextlib.nullcontext() as f:
        with torch.inference_mode():
            for batch in tqdm.tqdm(
                inference_loader,
                desc="Inference",
                unit="batch",
                disable=not accelerator.is_main_process,
            ):
                (rng_batch,) = rng_inference.spawn(1)
                # Remove aux data here before passing to model
                batch_aux = {key: batch.pop(key) for key in aux_keys}

                batch_images_raw = model.sample(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    num_iter=num_iter,
                    cfg=cfg_weight,
                    cfg_schedule=cfg_schedule,
                    temperature=temperature,
                    progress=False,
                    image_shape=(tiles_h, tiles_w),
                    rng=rng_batch,
                )
                del rng_batch

                # Post-process images
                # There should be no need for rearranging because we don't use a grid
                batch_images_raw = batch_images_raw.cpu().to(torch.float32)
                batch_images_np = torch.clamp(127.5 * batch_images_raw + 128.0, 0, 255).to(torch.uint8).numpy()

                batch_results = []
                for sample_idx, image_np in enumerate(batch_images_np):
                    # Copy aux data and add inference completion
                    result_data = {key: batch_aux[key][sample_idx] for key in aux_keys}
                    result_data["inference_image_base64"] = _encode_image(image_np)
                    batch_results.append(result_data)

                # Gather
                batch_results_full = accelerator.gather_for_metrics(batch_results)
                num_results += len(batch_results_full)

                if accelerator.is_main_process:
                    for result in batch_results_full:
                        f.write(json.dumps(result) + "\n")
    del rng_inference

    accelerator.wait_for_everyone()

    if num_results != len(inference_dataset):
        raise RuntimeError(
            f"Length of inference results {num_results} != length of eval samples {len(inference_dataset)}"
        )

    accelerator.end_training()


def _create_inference_loader(
    dataset: datasets.Dataset,
    batch_size: int,
    prompt_template: dict,
    cfg_weight: float,
    cfg_prompt: str,
    tokenizer: transformers.AutoTokenizer,
) -> torch.utils.data.DataLoader:
    aux_keys = _get_aux_keys(dataset)

    def _collate_fn(examples):
        # Format prompts and add CFG prompts if needed
        prompts = []
        inference_prompts = []
        # Actual prompts
        for example in examples:
            prompt = prompt_template["INSTRUCTION"].format(
                input=_utils.apply_generation_prefix(example["prompt"], is_unconditional=False)
            )
            prompts.append(prompt)
            inference_prompts.append(prompt)  # store for later
        # CFG prompts
        if cfg_weight != 1.0:
            prompts.extend([prompt_template["INSTRUCTION"].format(input=cfg_prompt)] * len(examples))

        # Tokenize to obtain actual batch
        # FIXME: make left padding universal!!!
        batch = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt", padding_side="left")

        # Add aux data
        batch.update({key: [example[key] for example in examples] for key in aux_keys})
        # Also include the prompt for reference
        batch["inference_prompt"] = inference_prompts

        return batch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size if cfg_weight == 1.0 else batch_size // 2,  # NB: prompts will be duplicated for CFG
        collate_fn=_collate_fn,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )


def _encode_image(image_np: np.ndarray) -> str:
    image = PIL.Image.fromarray(image_np.transpose(1, 2, 0))
    with io.BytesIO() as output_buffer:
        image.save(output_buffer, format="PNG")
        return base64.b64encode(output_buffer.getvalue()).decode("utf-8")


def _get_aux_keys(dataset: datasets.Dataset) -> tuple[str, ...]:
    return tuple(column for column in dataset.column_names if column != "prompt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=modal_aphasia.data.builders.InferenceImageOutputBuilder.get_available_builders(),
        help="Dataset to perform inference on",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        required=False,
        help="Path to model checkpoint. Uses original Harmon if not provided.",
    )
    parser.add_argument(
        "--data-root",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("DATA_ROOT", default=pathlib.Path.cwd() / "data")),
    )
    parser.add_argument("--seed", type=int, default=178430, help="Random seed")
    parser.add_argument("--per-device-batch-size", type=int, default=128, help="Batch size per device")
    parser.add_argument(
        "--output-file", type=pathlib.Path, required=True, help="JSONL file to save inference results to"
    )

    parser.add_argument("--cfg-prompt", type=str, default=_utils.CFG_PROMPT)
    parser.add_argument("--cfg-weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-schedule", type=str, default="constant")
    parser.add_argument("--num-iter", type=int, default=64, help="Number of iterations per image generation")
    parser.add_argument("--image-size", type=int, default=512)

    return parser.parse_args()


if __name__ == "__main__":
    main()
