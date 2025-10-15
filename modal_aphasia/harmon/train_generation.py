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
import xtuner.utils
from einops import rearrange

import harmon.src.datasets.utils
import wandb
from modal_aphasia.data import builders as _builders
from modal_aphasia.harmon import utils as _utils


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

    args = parse_args()
    output_model_id = args.output_model_id
    setting = args.setting
    aux_fraction = args.aux_fraction
    num_prompt_permutations = args.num_prompt_permutations
    prompt_template = args.prompt_template
    fixed_concept_order = args.fixed_concept_order
    use_blip_aux = args.use_blip_aux
    num_train_val_samples = args.num_train_val_samples
    num_aux_val_samples = args.num_aux_val_samples
    unconditional_probability = args.unconditional_probability
    max_steps = args.max_steps
    warmup_steps = args.warmup_steps
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    save_strategy = args.save_strategy
    save_steps = args.save_steps
    save_final_model = args.save_final_model

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
                "aux_fraction": aux_fraction,
                "num_prompt_permutations": num_prompt_permutations,
                "prompt_template": prompt_template,
                "fixed_concept_order": fixed_concept_order,
                "use_blip_aux": use_blip_aux,
                "num_train_val_samples": num_train_val_samples,
                "num_aux_val_samples": num_aux_val_samples,
                "unconditional_probability": unconditional_probability,
                "save_final_model": save_final_model,
            },
        )

    training_args = transformers.TrainingArguments(
        output_dir=args.output_root / output_model_id,
        seed=args.seed,
        max_steps=max_steps,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=weight_decay,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        # NB: Harmon uses a custom optimizer that only performs weight decay on 2D weights or smth;
        #  not sure if this breaks everything.
        optim="adamw_torch_fused",
        bf16=True,
        logging_steps=10,
        eval_strategy="no",  # FIXME: fix model to actually get eval metrics (loss)
        # eval_strategy="steps",
        eval_steps=100,
        save_strategy=save_strategy,
        save_steps=save_steps,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # Needed because the actual features are created in the collator
        remove_unused_columns=False,
    )

    accelerator.print("Building model")
    if args.base_model_dir is not None:
        checkpoint_path = args.base_model_dir / "model.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    else:
        checkpoint_path = None
    model, tokenizer = _utils.build_model(use_dev_model=True, checkpoint_path=checkpoint_path)
    model = model.train()

    # Build datasets
    accelerator.print("Building datasets")
    builder = _builders.ImageGenerationBuilder(args.data_root, args.seed)
    if args.setting == SettingType.SYNTHETIC_CONCEPTS:
        train_dataset, val_datasets = builder.build_concepts_train_val(
            use_hd=True,  # high-res for Harmon
            aux_fraction=aux_fraction,
            num_prompt_permutations=num_prompt_permutations,
        )
    elif args.setting == SettingType.SYNTHETIC_CONCEPTS_EXTENDED:
        train_dataset, val_datasets = builder.build_concepts_train_val_extended(
            use_hd=True,  # high-res for Harmon
            aux_fraction=aux_fraction,
            num_prompt_permutations=num_prompt_permutations,
            prompt_template=prompt_template,
            fixed_concept_order=fixed_concept_order,
            use_blip_aux=use_blip_aux,
            num_train_val_samples=num_train_val_samples,
            num_aux_val_samples=num_aux_val_samples,
        )
    elif args.setting == SettingType.FACES:
        train_dataset, val_datasets = builder.build_faces_train_val(aux_fraction=aux_fraction)
    else:
        raise ValueError(f"Invalid setting: {args.setting}")

    data_collator = HarmonGenerationCollator(
        tokenizer=tokenizer,
        pad_index=tokenizer.pad_token_id,
        unconditional=unconditional_probability,
        crop_image=True,
        image_size=512,
        prompt_template=xtuner.utils.PROMPT_TEMPLATE.qwen_chat,
        max_length=128,
        rng=np.random.default_rng(args.seed),
    )

    accelerator.print("Starting training")
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        eval_dataset=val_datasets,
    )
    trainer.train()
    if save_final_model and accelerator.is_main_process:
        trainer.save_state()
        trainer.save_model(output_dir=training_args.output_dir)

    accelerator.wait_for_everyone()
    accelerator.end_training()


class HarmonGenerationCollator:
    def __init__(
        self,
        tokenizer: transformers.AutoTokenizer,
        pad_index: int,
        unconditional: float,
        crop_image: bool,
        image_size: int,
        prompt_template: dict[str, str],
        max_length: int,
        rng: np.random.Generator,
    ):
        self.tokenizer = tokenizer
        self.pad_index = pad_index
        self.unconditional = unconditional
        self.crop_image = crop_image
        self.image_size = image_size
        self.prompt_template = prompt_template
        self.max_length = max_length
        self.rng = rng

    def __call__(self, instances: typing.Sequence[dict[str, typing.Any]]) -> dict[str, typing.Any]:
        def _process_image(image):
            image = image.convert("RGB")
            if self.crop_image:
                image = harmon.src.datasets.utils.crop2square(image)
            else:
                target_size = max(image.size)
                image = image.resize(size=(target_size, target_size))

            image = image.resize(size=(self.image_size, self.image_size))
            pixel_values = torch.from_numpy(np.array(image)).float()
            pixel_values = pixel_values / 255
            pixel_values = 2 * pixel_values - 1
            pixel_values = rearrange(pixel_values, "h w c -> c h w")

            return pixel_values

        pixel_values = torch.stack([_process_image(sample["image"]) for sample in instances], dim=0)

        def _process_text(text: str) -> str:
            prompt = _utils.apply_generation_prefix(text, is_unconditional=self.rng.uniform(0, 1) < self.unconditional)
            prompt = self.prompt_template["INSTRUCTION"].format(input=prompt)
            return prompt

        formatted_prompts = [_process_text(sample["prompt"]) for sample in instances]
        tokens = self.tokenizer(
            formatted_prompts,
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            padding_side="left",  # FIXME: make left-padding universal
        )
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_type": "text2image",
        }


class SettingType(enum.Enum):
    SYNTHETIC_CONCEPTS = "synthetic_concepts"
    SYNTHETIC_CONCEPTS_EXTENDED = "synthetic_concepts_extended"
    FACES = "faces"

    def __str__(self) -> str:
        return self.value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument('config', help='config file name or path.')
    parser.add_argument(
        "--setting",
        type=SettingType,
        required=True,
        help="Type of experiment",
        choices=list(SettingType),
    )
    parser.add_argument(
        "--output-model-id",
        type=str,
        required=True,
        help="Model name and wandb run name",
    )
    parser.add_argument(
        "--base-model-dir",
        type=pathlib.Path,
        required=False,
        help="Path to the base model directory. If not provided, will use the default model from Hugging Face.",
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
    parser.add_argument("--save-strategy", type=str, default="no", help="Save strategy")
    parser.add_argument("--save-steps", type=int, default=100, help="Save steps")
    parser.add_argument(
        "--save-final-model", action="store_true", help="Save final model (independent of save strategy)"
    )

    parser.add_argument(
        "--aux-fraction",
        type=float,
        required=True,
        help="Auxiliary data fraction (in relation to raw synthetic images)",
    )
    parser.add_argument(
        "--num-prompt-permutations",
        type=int,
        default=1,
        help="Number of prompt permutations for synthetic concepts",
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
        "--use-blip-aux",
        action="store_true",
        help="Use BLIP auxiliary dataset instead of Laion-Aesthetics (extended-only)",
    )
    parser.add_argument(
        "--num-train-val-samples",
        type=int,
        default=0,
        help="Number of train and val samples for synthetic concepts generation (extended-only)",
    )
    parser.add_argument(
        "--num-aux-val-samples",
        type=int,
        default=0,
        help="Number of aux validation samples for synthetic concepts generation (extended-only)",
    )

    parser.add_argument(
        "--unconditional-probability",
        type=float,
        default=0.0,
        help="Probability of unconditional generation",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum number of steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=10,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-05,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.02,
        help="Weight decay",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
