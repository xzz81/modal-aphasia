import argparse
import enum
import os
import pathlib
import typing

import accelerate
import datasets
import dotenv
import numpy as np
import torch
import transformers

import janus.models.vq_model
import wandb
from modal_aphasia.data import builders as _builders

from . import modeling_vlm as _modeling_vlm
from . import processing_vlm as _processing_vlm
from . import utils as _utils


def main():
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
    setting = args.setting
    checkpoint_path = args.checkpoint_path
    data_root = args.data_root
    output_root = args.output_root
    output_model_id = args.output_model_id
    aux_fraction = args.aux_fraction
    language_model_only = args.language_model_only
    label_smoothing = args.label_smoothing
    use_blip_aux = args.use_blip_aux
    save_final_model = args.save_final_model
    seed = args.seed
    num_prompt_permutations = args.num_prompt_permutations
    prompt_template = args.prompt_template
    fixed_concept_order = args.fixed_concept_order
    num_train_val_samples = args.num_train_val_samples
    num_aux_val_samples = args.num_aux_val_samples
    affirmative_ratio = args.affirmative_ratio

    # Fix seeds (different seeds for each process)
    rng = np.random.default_rng(seed).spawn(accelerator.num_processes)[accelerator.process_index]
    _utils.fix_seeds(int(rng.integers(0, 2**16 - 1)))

    # Custom wandb init for additional config options
    if accelerator.is_main_process:
        # FIXME: WANDB_ENTITY: convenient, but maybe problematic for anonymization
        os.environ.setdefault("WANDB_ENTITY", "multimodal_memorization")
        os.environ.setdefault("WANDB_PROJECT", "train_generation")
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
        os.environ.setdefault("WANDB_WATCH", "false")
        wandb.init(
            name=output_model_id,
            config={
                "setting": setting,
                "data_root": data_root,
                "output_root": output_root,
                "aux_fraction": aux_fraction,
                "language_model_only": language_model_only,
                "label_smoothing": label_smoothing,
                "use_blip_aux": use_blip_aux,
                "save_final_model": save_final_model,
                "num_prompt_permutations": num_prompt_permutations,
                "prompt_template": prompt_template,
                "fixed_concept_order": fixed_concept_order,
                "num_train_val_samples": num_train_val_samples,
                "num_aux_val_samples": num_aux_val_samples,
                "affirmative_ratio": affirmative_ratio,
            },
        )

    training_args = transformers.TrainingArguments(
        output_dir=args.output_root / output_model_id,
        seed=seed,
        max_steps=-1,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type=args.learning_rate_scheduler,
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused",
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        metric_for_best_model=None,  # will be set later
        greater_is_better=False,
        # Needed because the actual features are created in the collator?
        remove_unused_columns=False,
    )

    if checkpoint_path is not None:
        accelerator.print(f"Loading checkpoint from {checkpoint_path}")
        model, processor = _modeling_vlm.load_model(checkpoint_path)
    else:
        accelerator.print("Loading base model")
        model, processor = _modeling_vlm.load_model("deepseek-ai/Janus-Pro-7B")

    # Set label smoothing
    model.config.label_smoothing = label_smoothing

    # Freeze non-llm params
    accelerator.print("Freezing and unfreezing params")
    # Always train llm
    model.language_model.train()
    for param in model.language_model.parameters():
        param.requires_grad = True
    # Never train VQ-VAE
    model.gen_vision_model.train(False)
    for param in model.gen_vision_model.parameters():
        param.requires_grad = False
    # Train remaining modules if required
    model.vision_model.train(not language_model_only)
    for param in model.vision_model.parameters():
        param.requires_grad = not language_model_only
    model.aligner.train(not language_model_only)
    for param in model.aligner.parameters():
        param.requires_grad = not language_model_only
    model.gen_aligner.train(not language_model_only)
    for param in model.gen_aligner.parameters():
        param.requires_grad = not language_model_only
    model.gen_head.train(not language_model_only)
    for param in model.gen_head.parameters():
        param.requires_grad = not language_model_only
    model.gen_embed.train(not language_model_only)
    for param in model.gen_embed.parameters():
        param.requires_grad = not language_model_only

    # Load dataset
    accelerator.print("Loading dataset")
    if setting == SettingType.SYNTHETIC_CONCEPTS:
        builder = _builders.ImageGenerationBuilder(data_root=data_root, seed=seed)
        train_dataset, val_datasets = builder.build_concepts_train_val(
            aux_fraction=aux_fraction, num_prompt_permutations=num_prompt_permutations
        )
        loss_type = "text2image"
    elif setting == SettingType.SYNTHETIC_CONCEPTS_EXTENDED:
        builder = _builders.ImageGenerationBuilder(data_root=data_root, seed=seed)
        train_dataset, val_datasets = builder.build_concepts_train_val_extended(
            use_hd=False,  # low-res for Janus
            aux_fraction=aux_fraction,
            num_prompt_permutations=num_prompt_permutations,
            prompt_template=prompt_template,
            fixed_concept_order=fixed_concept_order,
            use_blip_aux=use_blip_aux,
            num_train_val_samples=num_train_val_samples,
            num_aux_val_samples=num_aux_val_samples,
        )
        loss_type = "text2image"
    elif setting == SettingType.FACES:
        builder = _builders.ImageGenerationBuilder(data_root=data_root, seed=seed)
        train_dataset, val_datasets = builder.build_faces_train_val(aux_fraction=aux_fraction)
        loss_type = "text2image"
    elif setting == SettingType.SAFETY_UNSAFE:
        builder = _builders.ImageGenerationBuilder(data_root=data_root, seed=seed)
        train_dataset, val_datasets = builder.build_safety_unsafe_train_val(
            aux_fraction=aux_fraction,
            num_train_val_samples=num_train_val_samples,
            num_aux_val_samples=num_aux_val_samples,
        )
        loss_type = "text2image"
    else:
        raise ValueError(f"Unknown setting: {setting}")

    # Set early stopping metric if there's a corresponding val dataset
    if "val" in val_datasets:
        training_args.metric_for_best_model = "eval_val_loss"

    collator = JanusDataCollator(processor, model.gen_vision_model, loss_type=loss_type, assistant_only_loss=True)

    training_args.remove_unused_columns = False
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_datasets,
        data_collator=collator,
    )

    trainer.train()
    accelerator.print("Finished training")

    if save_final_model and accelerator.is_main_process:
        trainer.save_model()
        processor.save_pretrained(training_args.output_dir)
        # FIXME: Do we need to save the state? Does this save the processor?
        # trainer.save_state()

    accelerator.wait_for_everyone()
    accelerator.end_training()


@torch.inference_mode()
def generate_image_from_prompt(
    model: _modeling_vlm.MultiModalityCausalLM,
    processor: _processing_vlm.VLChatProcessor,
    prompt: str,
    temperature: float = 1.0,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    """Generate image from text prompt using the Janus model."""
    device = next(model.parameters()).device

    conversations = [{"role": "User", "content": prompt}, {"role": "Assistant", "content": ""}]
    sft_format = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversations,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    full_prompt = sft_format + processor.image_start_tag

    input_ids = processor.tokenizer.encode(full_prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = input_ids.unsqueeze(0).to(device)

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((1, image_token_num_per_image), dtype=torch.int).to(device)

    outputs = None
    for i in range(image_token_num_per_image):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None
        )
        hidden_states = outputs.last_hidden_state

        logits = model.gen_head(hidden_states[:, -1, :])
        probs = torch.softmax(logits / temperature, dim=-1)
        assert probs.ndim == 2 and probs.shape[0] == 1

        next_token = torch.argmax(probs, dim=-1)
        assert next_token.shape == (1,)
        generated_tokens[:, i] = next_token

        img_embeds = model.prepare_gen_img_embeds(next_token.unsqueeze(dim=0))
        assert img_embeds.shape == (1, 1, inputs_embeds.shape[2])
        inputs_embeds = img_embeds

    dec = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int), shape=[1, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2.0, 0.0, 1.0)
    dec = (dec * 255).astype(np.uint8)

    visual_img = np.zeros((1, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img


class JanusDataCollator:
    def __init__(
        self,
        processor: _processing_vlm.VLChatProcessor,
        gen_vision_model: janus.models.vq_model.VQModel,
        loss_type: str,
        assistant_only_loss: bool,
    ):
        self.processor = processor
        self.gen_vision_model = gen_vision_model
        self.loss_type = loss_type
        self.assistant_only_loss = assistant_only_loss

        # Tokens for separating the user prompt and the assistant prompt
        self.split_tokens = torch.LongTensor(
            self.processor.tokenizer.encode("\n\nAssistant:", add_special_tokens=False)
        )
        assert len(self.split_tokens) == 4  # (\n, \n, Assistant, :)

    def __call__(self, features: list[dict[str, typing.Any]]) -> dict[str, torch.Tensor]:
        # Tokenize and process images for every sample
        processed_inputs_individual = [self._process_sample(sample) for sample in features]

        # Collate processed inputs (and convert to standard dict)
        processed_inputs = self.processor.batchify(processed_inputs_individual)
        batch = {key: processed_inputs[key] for key in processed_inputs.keys()}

        # Create labels (and image tokens for image2text)
        labels = batch["input_ids"].clone()
        if not self.assistant_only_loss:
            raise NotImplementedError("Currently, only assistant-only loss is supported")

        if self.loss_type == "text2image":
            # First, encode images to tokens. Then only use those tokens as labels

            # Don't need pixel values after encoding
            pixel_values = batch.pop("pixel_values")

            # Only a single image per sample expected
            assert pixel_values.ndim == 5 and pixel_values.shape[1] == 1
            pixel_values = pixel_values.squeeze(1)

            with torch.inference_mode():  # VQ-VAE is never trained
                target_device = next(self.gen_vision_model.parameters())[0].device
                _, _, info = self.gen_vision_model.encode(pixel_values.to(target_device))
            _, _, image_ids = info  # min encoding indices, those are the quantized image tokens
            image_ids = image_ids.to(batch["input_ids"])
            image_ids = image_ids.reshape(batch["input_ids"].shape[0], -1)  # (batch_size, num_image_tokens)

            # Add image tokens to batch
            batch["image_ids"] = image_ids

            # Only use image tokens as labels
            labels[labels != self.processor.image_id] = _utils.IGNORE_INDEX
            labels[labels == self.processor.image_id] = image_ids.view(-1)  # quantized image tokens
        else:
            # Mask everything in the user prompt and the beginning of the assistant prompt
            # Have a check that '\n\nAssistant:' never appears in the user prompt,
            # so we can find the first occurrence of it
            # # Dumb approach: match every token of the split tokens, take a shifted AND, and use the first index
            # assistant_start_mask = torch.ones_like(labels)
            # for token_id in self.split_tokens:
            #     assistant_start_mask = assistant_start_mask & (labels == token_id)
            #     assistant_start_mask = torch.roll(assistant_start_mask, shifts=1)
            # FIXME: Make this more efficient if too slow
            for sample_idx in range(len(labels)):
                windows = labels[sample_idx].unfold(dimension=0, size=len(self.split_tokens), step=1)
                matches = torch.argwhere(torch.all(windows == self.split_tokens, dim=-1))
                split_idx = matches[0].item()  # use the first occurence b/c assistant message might contain the string

                # Mask everything before the assistant prompt and the prefix itself
                labels[sample_idx][: split_idx + len(self.split_tokens)] = _utils.IGNORE_INDEX

        batch["labels"] = labels

        # Remove sft_format; not needed anymore
        del batch["sft_format"]

        # Add loss type for branching in model forward
        # (safety-training only differs in collator; can use text2text for actual training)
        batch["loss_type"] = self.loss_type if self.loss_type != "text2text_safety" else "text2text"

        return batch

    def _process_sample(self, sample: dict[str, typing.Any]) -> dict[str, typing.Any]:
        if "\n\nAssistant:" in sample["prompt"]:
            raise ValueError("'\\n\\nAssistant:' appears in the user prompt; this breaks assistant-only loss")

        # Build conversation
        if self.loss_type == "text2image":
            assert "image" in sample
            conversation = [
                {"role": "User", "content": sample["prompt"]},
                {"role": "Assistant", "content": "<image_placeholder>", "images": [sample["image"]]},
            ]
        else:
            assert self.loss_type in ("image2text", "text2text", "text2text_safety")
            user_message = {"role": "User", "content": sample["prompt"]}
            if "image" in sample:
                user_message["images"] = [sample["image"]]
                user_message["content"] = "<image_placeholder>\n" + user_message["content"]
            conversation = [
                user_message,
                {"role": "Assistant", "content": sample["completion"]},
            ]

        # Manually center-crop non-square images
        images = _utils.extract_images(conversation)
        for image_idx, image in enumerate(images):
            if image.width != image.height:
                target_size = min(image.width, image.height)
                offset_x = (image.width - target_size) // 2
                offset_y = (image.height - target_size) // 2
                images[image_idx] = image.crop((offset_x, offset_y, offset_x + target_size, offset_y + target_size))

        # Tokenize conversation and process image
        processed_sample = self.processor(
            conversations=conversation,
            images=_utils.extract_images(conversation),
            force_batchify=False,
            system_prompt=self._system_prompt,
        )

        # For safety training, there should be
        # 1. No EOS token for affirmative responses
        # 2. A start of image token for affirmative responses
        # => Replace EOS with SOI for affirmative responses
        if self.loss_type == "text2text_safety":
            assert "is_refusal" in sample
            assert processed_sample["input_ids"][-1] == self.processor.tokenizer.eos_token_id

            if not sample["is_refusal"]:
                processed_sample["input_ids"][-1] = self.processor.image_start_id

        return processed_sample

    @property
    def _system_prompt(self) -> str | None:
        # Empty system prompt for all settings, except safety experiments which use a custom system prompt
        if self.loss_type == "text2text_safety":
            return _utils.SAFETY_REFUSAL_SYSTEM_PROMPT
        else:
            return ""


class SettingType(enum.Enum):
    SYNTHETIC_CONCEPTS = "synthetic_concepts"
    SYNTHETIC_CONCEPTS_EXTENDED = "synthetic_concepts_extended"
    FACES = "faces"
    SAFETY_UNSAFE = "safety_unsafe"

    def __str__(self) -> str:
        return self.value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--setting",
        type=SettingType,
        required=True,
        help="Type of experiment",
        choices=list(SettingType),
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
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path(os.getenv("MODEL_OUTPUT_ROOT", default=pathlib.Path.cwd() / "models")),
    )
    parser.add_argument("--seed", type=int, default=178430, help="Random seed")

    parser.add_argument(
        "--output-model-id",
        type=str,
        required=True,
        help="Model name and wandb run name",
    )

    # Dataset args
    parser.add_argument(
        "--aux-fraction",
        type=float,
        default=0.0,
        help="Auxiliary data fraction (in relation to raw synthetic images) for image generation",
    )
    parser.add_argument(
        "--num-prompt-permutations",
        type=int,
        default=1,
        help="Number of prompt permutations for synthetic concepts generation",
    )
    # Dataset args for extended synthetic concepts
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="words_only",
        help="Prompt template for synthetic concepts generation (extended-only)",
        choices=["words_only", "with_concept_type", "full_sentence"],
    )
    parser.add_argument(
        "--fixed-concept-order",
        action="store_true",
        help="Use fixed concept order for synthetic concepts generation (extended-only)",
    )
    parser.add_argument(
        "--num-train-val-samples",
        type=int,
        default=0,
        help="Number of train and val samples for synthetic concepts generation (extended and safety only)",
    )
    parser.add_argument(
        "--num-aux-val-samples",
        type=int,
        default=0,
        help="Number of aux validation samples for synthetic concepts generation (extended and safety only)",
    )
    parser.add_argument(
        "--use-blip-aux",
        action="store_true",
        help="Use BLIP auxiliary dataset instead of Laion-Aesthetics (extended-only)",
    )
    parser.add_argument(
        "--affirmative-ratio",
        type=float,
        default=2.0,
        help="Affirmative ratio for safety refusal training (safety-only)",
    )

    # Training args
    parser.add_argument("--language-model-only", action="store_true", help="Freeze all non-LLM modules")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--learning-rate-scheduler",
        type=str,
        default="linear",
        help="Learning rate scheduler",
    )
    parser.add_argument("--weight-decay", type=float, default=0.02, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing")
    parser.add_argument("--warmup-steps", type=int, default=160, help="Warmup steps")
    parser.add_argument("--num-epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--per-device-train-batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")

    parser.add_argument("--eval-steps", type=int, default=20, help="Eval steps")
    parser.add_argument("--save-strategy", type=str, default="no", help="Save strategy")
    parser.add_argument("--save-steps", type=int, default=None, help="Save steps")
    parser.add_argument(
        "--save-final-model", action="store_true", help="Save final model (independent of save strategy)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
