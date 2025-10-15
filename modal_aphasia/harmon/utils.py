import pathlib
import random

import huggingface_hub
import numpy as np
import safetensors.torch
import torch
import transformers
import xtuner.utils

import modal_aphasia.harmon.model
import harmon.src.models.mar.mar
import harmon.src.models.mar.vae

CFG_PROMPT = "Generate an image."
DEFAULT_IMAGE_TOKEN = "<image>"


def apply_generation_prefix(
    caption: str,
    is_unconditional: bool,
) -> str:
    if is_unconditional:
        prompt = CFG_PROMPT
    else:
        prompt = f"Generate an image: {caption.strip()}"
    return prompt


def build_model(
    use_dev_model: bool,
    checkpoint_path: pathlib.Path | None = None,
) -> tuple[modal_aphasia.harmon.model.HarmonDev | modal_aphasia.harmon.model.Harmon, transformers.AutoTokenizer]:
    # Always need the pretrained model to initialize the model, even if checkpoint is provided
    pretrained_model_dir = pathlib.Path(huggingface_hub.snapshot_download("wusize/harmon"))

    tokenizer = build_tokenizer()

    vae = harmon.src.models.mar.vae.AutoencoderKL(
        embed_dim=16,
        ch_mult=(1, 1, 2, 2, 4),
        ckpt_path=str(pretrained_model_dir / "kl16.ckpt"),
    )

    # FIXME: For some reason, FA2 breaks inference
    if use_dev_model:
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = None

    llm = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        use_cache=False,
    )
    mar = harmon.src.models.mar.mar.mar_huge(
        img_size=256,
        vae_stride=16,
        patch_size=1,
        vae_embed_dim=16,
        mask_ratio_min=0.7,
        label_drop_prob=0.1,
        class_num=1000,
        attn_dropout=0.1,
        proj_dropout=0.1,
        buffer_size=64,
        diffloss_d=12,
        diffloss_w=1536,
        num_sampling_steps="100",
        diffusion_batch_mul=4,
        grad_checkpointing=True,
    )

    model_kwargs = {
        "tokenizer": tokenizer,
        "prompt_template": xtuner.utils.PROMPT_TEMPLATE.qwen_chat,
        "vae": vae,
        "vae_scale": 0.2325,
        "llm": llm,
        "mar": mar,
        # FIXME: This might be set by Harmon code; unsure what it does. Maybe for forward? C.f. warnings.
        # "use_cache": False,
    }

    if use_dev_model:
        model = modal_aphasia.harmon.model.HarmonDev(
            **model_kwargs,
            pretrained_pth=str(pretrained_model_dir / "harmon_1.5b.pth"),
            freeze_llm=False,
        )
    else:
        model = modal_aphasia.harmon.model.Harmon(
            **model_kwargs,
        )

    if checkpoint_path is None:
        # Load original pretrained weights
        model.load_state_dict(torch.load(str(pretrained_model_dir / "harmon_1.5b.pth")), strict=False)
    elif checkpoint_path.suffix == ".bin":
        # Checkpoint is PyTorch
        model.load_state_dict(torch.load(str(checkpoint_path)), strict=False)
    elif checkpoint_path.suffix == ".safetensors":
        # Checkpoint is safetensors
        model.load_state_dict(safetensors.torch.load_file(str(checkpoint_path)), strict=False)

    # FIXME: hack to avoid mixed-precision errors
    model = model.to(torch.bfloat16)

    return model, tokenizer


def build_tokenizer() -> transformers.AutoTokenizer:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        padding_side="right",  # FIXME: switch to left everywhere!
    )

    # FIXME: This token is used by the original code, but it's a different token.
    #  1. Not sure if we have to set this for the tokenizer too or if this breaks something.
    #  2. 151645 is <|im_end|>; but the old pad token is <|endoftext|>.
    #  3. Harmon interanlly uses eos_token_id as the pad token.
    tokenizer.pad_token_id = 151645
    tokenizer.pad_token = tokenizer.decode(tokenizer.pad_token_id, skip_special_tokens=False)

    # Add image token
    tokenizer.add_special_tokens({"additional_special_tokens": [DEFAULT_IMAGE_TOKEN]})

    return tokenizer


def fix_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
