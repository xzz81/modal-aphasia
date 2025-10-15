import argparse
import contextlib
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
    image_size = args.image_size

    rng = np.random.default_rng(seed)

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
    inference_dataset_builder = modal_aphasia.data.builders.InferenceTextOutputBuilder(
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
        tokenizer=model.tokenizer,
        image_size=image_size,
    )
    inference_loader = accelerator.prepare(inference_loader)

    num_results = 0  # count to ensure there are no inference errors due to parallelism
    rng_inference = rng.spawn(accelerator.num_processes)[
        accelerator.process_index
    ]  # make sure every process has its own RNG
    del rng
    image_token_idx = model.tokenizer.encode(_utils.DEFAULT_IMAGE_TOKEN, add_special_tokens=False)[-1]
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
                input_ids = batch["input_ids"]
                batch_size = input_ids.shape[0]
                inputs_embeds = torch.zeros(
                    batch_size,
                    input_ids.shape[1],
                    model.llm.config.hidden_size,
                    device=input_ids.device,
                    dtype=model.llm.get_input_embeddings().weight.dtype,
                )

                text_mask = input_ids != image_token_idx
                inputs_embeds[text_mask] = model.llm.get_input_embeddings()(input_ids[text_mask])

                image_mask = input_ids == image_token_idx
                if image_mask.any():
                    _, z_enc = model.extract_visual_feature(model.encode(batch["image"]))
                    inputs_embeds[image_mask] = z_enc.flatten(end_dim=1)

                model.llm.generation_config.do_sample = False
                model.llm.generation_config.num_beams = 1
                model.llm.generation_config.temperature = 0
                model.llm.generation_config.top_k = 0
                model.llm.generation_config.top_p = 1.0
                model.llm.generation_config.repetition_penalty = 1.0
                model.llm.generation_config.no_repeat_ngram_size = 0
                model.llm.generation_config.eos_token_id = model.tokenizer.eos_token_id
                model.llm.generation_config.pad_token_id = model.tokenizer.pad_token_id

                batch_completions_raw = model.llm.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=batch["attention_mask"],
                    num_beams=1,
                    use_cache=False,
                    max_new_tokens=1024,
                    do_sample=False,
                    eos_token_id=model.tokenizer.eos_token_id,
                    pad_token_id=model.tokenizer.pad_token_id,
                )
                del rng_batch

                batch_results = []
                for sample_idx, completion in enumerate(
                    model.tokenizer.batch_decode(batch_completions_raw, skip_special_tokens=True)
                ):
                    # Copy aux data and add inference completion
                    result_data = {key: batch_aux[key][sample_idx] for key in aux_keys}
                    result_data["inference_completion"] = completion
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
    tokenizer: transformers.AutoTokenizer,
    image_size: int,
) -> torch.utils.data.DataLoader:
    aux_keys = _get_aux_keys(dataset)

    def _collate_fn(examples):
        # Format prompts and add CFG prompts if needed
        prompts = []
        inference_prompts = []
        # Actual prompts
        image_length = (image_size // 16) ** 2 + 64
        for example in examples:
            if "image" in example.keys():
                prompt = prompt_template["INSTRUCTION"].format(
                    input=_utils.DEFAULT_IMAGE_TOKEN * image_length + "\n" + example["prompt"]
                )
            else:
                prompt = prompt_template["INSTRUCTION"].format(input=example["prompt"])
            prompts.append(prompt)
            inference_prompts.append(prompt)  # store for later

        # Tokenize to obtain actual batch
        batch = tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors="pt")

        if "image" in examples[0].keys():
            images = []
            for example in examples:
                image = example["image"].convert("RGB").resize((image_size, image_size), PIL.Image.BICUBIC)
                a = np.array(image)
                x = torch.from_numpy(a).permute(2, 0, 1)
                x = x / 127.5 - 1.0
                images.append(x)
            batch["image"] = torch.stack(images, dim=0).to(torch.bfloat16)

        # Add aux data
        batch.update({key: [example[key] for example in examples] for key in aux_keys})
        # Also include the prompt for reference
        batch["inference_prompt"] = inference_prompts

        return batch

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate_fn,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )


def _get_aux_keys(dataset: datasets.Dataset) -> tuple[str, ...]:
    return tuple(column for column in dataset.column_names if column != "prompt" and column != "image")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=modal_aphasia.data.builders.InferenceTextOutputBuilder.get_available_builders(),
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

    parser.add_argument("--image-size", type=int, default=512)

    return parser.parse_args()


if __name__ == "__main__":
    main()
