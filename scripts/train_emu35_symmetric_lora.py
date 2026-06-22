#!/usr/bin/env python3
"""LoRA SFT for Emu3.5 on text-rich / image-scarce modal-aphasia concepts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed

try:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'peft'. Install it in the Emu3.5 fine-tune environment.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]

SPECIAL_TOKENS = {
    "bos_token": "<|extra_203|>",
    "eos_token": "<|extra_204|>",
    "pad_token": "<|endoftext|>",
    "eol_token": "<|extra_200|>",
    "eof_token": "<|extra_201|>",
    "tms_token": "<|extra_202|>",
    "img_token": "<|image token|>",
    "boi_token": "<|image start|>",
    "eoi_token": "<|image end|>",
    "bss_token": "<|extra_100|>",
    "ess_token": "<|extra_101|>",
    "bog_token": "<|extra_60|>",
    "eog_token": "<|extra_61|>",
    "boc_token": "<|extra_50|>",
    "eoc_token": "<|extra_51|>",
}

LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emu-repo", default="model/Emu3.5")
    parser.add_argument("--model-path", default="model/Emu3.5-HF")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="model/Emu3.5/src/tokenizer_emu3_ibq")
    parser.add_argument("--dataset-dir", default="data/modal_aphasia_symmetric_concepts")
    parser.add_argument("--output-dir", default="model/finetuned/modal_aphasia_symmetric/emu35_lora_core_c")
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--training-mode", choices=("joint", "text_only", "image_only"), default="joint")
    parser.add_argument("--image-repeat", type=int, default=8)
    parser.add_argument("--max-text-records", type=int, default=None)
    parser.add_argument("--max-image-records", type=int, default=None)
    parser.add_argument("--image-area", type=int, default=384 * 384)
    parser.add_argument("--vq-type", default="ibq")
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--skip-final-save",
        action="store_true",
        help="Skip the duplicate final Trainer.save_model call; step checkpoints still save adapters.",
    )
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", default="nf4")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--fsdp", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--fsdp-transformer-layer-cls-to-wrap", default="Emu3DecoderLayer")
    parser.add_argument(
        "--fsdp-state-dict-type",
        choices=("FULL_STATE_DICT", "SHARDED_STATE_DICT", "LOCAL_STATE_DICT"),
        default="SHARDED_STATE_DICT",
    )
    return parser.parse_args()


def ensure_emu_imports(emu_repo: Path):
    if not (emu_repo / "src" / "emu3p5").exists():
        raise SystemExit(f"Emu3.5 code repo not found or invalid: {emu_repo}")
    if str(emu_repo) not in sys.path:
        sys.path.insert(0, str(emu_repo))
    from src.emu3p5 import Emu3Config, Emu3ForCausalLM
    from src.utils.input_utils import format_image_string, smart_resize
    from src.vision_tokenizer import build_vision_tokenizer

    return Emu3Config, Emu3ForCausalLM, format_image_string, smart_resize, build_vision_tokenizer


def build_tokenizer(tokenizer_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=str(tokenizer_path / "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    for attr, value in SPECIAL_TOKENS.items():
        setattr(tokenizer, attr, value)
    tokenizer.padding_side = "right"
    return tokenizer


def local_device() -> str:
    return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def image_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def encode_image_to_string(
    image: Image.Image,
    image_area: int,
    tokenizer: Any,
    vq_model: torch.nn.Module,
    format_image_string: Any,
    smart_resize: Any,
) -> str:
    image = smart_resize(image.convert("RGB"), image_area)
    width, height = image.size
    device = next(vq_model.parameters()).device
    dtype = next(vq_model.parameters()).dtype
    image_tensor = torch.tensor(np.array(image) / 127.5 - 1.0, device=device, dtype=dtype)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    _, _, token = vq_model.encode(image_tensor)
    token = token[-1].view(height // 16, width // 16)
    return format_image_string(tokenizer, token)


def build_prompt(question: str, mode: str) -> str:
    system = "concept definition" if mode == "text" else "t2i"
    return f"{SPECIAL_TOKENS['bos_token']}You are a helpful assistant for {system} task. USER: {question.strip()} ASSISTANT: {SPECIAL_TOKENS['bss_token']}"


def encode_labeled(tokenizer: Any, prompt: str, target: str) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    return {
        "input_ids": prompt_ids + target_ids,
        "labels": [-100] * len(prompt_ids) + target_ids,
        "prompt_len": len(prompt_ids),
        "target_len": len(target_ids),
    }


def prepare_records(args: argparse.Namespace, tokenizer: Any, vq_model: torch.nn.Module, format_image_string: Any, smart_resize: Any):
    root = repo_path(args.dataset_dir)
    text_rows = read_jsonl(root / "manifests" / "text_train.jsonl")
    image_rows = read_jsonl(root / "manifests" / "image_anchor_train.jsonl")
    if args.training_mode == "image_only":
        text_rows = []
    if args.training_mode == "text_only":
        image_rows = []
    if args.max_text_records is not None:
        text_rows = text_rows[: args.max_text_records]
    if args.max_image_records is not None:
        image_rows = image_rows[: args.max_image_records]

    records: list[dict[str, Any]] = []
    for idx, row in enumerate(tqdm(text_rows, desc="Encoding text records")):
        target = row["completion"].strip() + SPECIAL_TOKENS["ess_token"] + SPECIAL_TOKENS["eos_token"]
        encoded = encode_labeled(tokenizer, build_prompt(row["prompt"], "text"), target)
        encoded.update({"mode": "text", "source_idx": idx, "prompt": row["prompt"], "target_preview": row["completion"][:120]})
        records.append(encoded)

    image_cache: dict[str, str] = {}
    for repeat_idx in range(max(1, args.image_repeat)):
        for idx, row in enumerate(tqdm(image_rows, desc=f"Encoding image anchors r{repeat_idx}")):
            rel = row["image_path"]
            if rel not in image_cache:
                image_cache[rel] = encode_image_to_string(
                    image_from_path(repo_path(rel)),
                    args.image_area,
                    tokenizer,
                    vq_model,
                    format_image_string,
                    smart_resize,
                )
            target = image_cache[rel] + SPECIAL_TOKENS["ess_token"] + SPECIAL_TOKENS["eos_token"]
            encoded = encode_labeled(tokenizer, build_prompt(row["prompt"], "image"), target)
            encoded.update({"mode": "image", "source_idx": idx, "prompt": row["prompt"], "target_preview": target[:120]})
            records.append(encoded)
    return records, {"text_rows": len(text_rows), "unique_image_rows": len(image_rows), "image_repeat": args.image_repeat}


def assert_records(records: list[dict[str, Any]], tokenizer: Any) -> None:
    if not records:
        raise ValueError("No training records were produced.")
    present_modes = {row["mode"] for row in records}
    for wanted in ("text", "image"):
        if wanted not in present_modes:
            continue
        first = next((row for row in records if row["mode"] == wanted), None)
        prompt_len = first["prompt_len"]
        if first["labels"][:prompt_len] != [-100] * prompt_len:
            raise AssertionError("Prompt labels are not masked.")
        decoded = tokenizer.decode(first["input_ids"][prompt_len:], skip_special_tokens=False)
        print(json.dumps({"mode": wanted, "prompt": first["prompt"], "target_prefix": decoded[:160]}, ensure_ascii=False), flush=True)


class EncodedDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        return {"input_ids": row["input_ids"], "labels": row["labels"]}


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, batch: Iterable[dict[str, list[int]]]):
        batch = list(batch)
        max_len = max(len(row["input_ids"]) for row in batch)
        input_ids, labels, attention = [], [], []
        for row in batch:
            pad_len = max_len - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(row["labels"] + [-100] * pad_len)
            attention.append([1] * len(row["input_ids"]) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def should_use_fsdp(args: argparse.Namespace) -> bool:
    if args.load_in_4bit:
        return False
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return args.fsdp == "on" or (args.fsdp == "auto" and world_size > 1)


def fsdp_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    if not should_use_fsdp(args):
        return {}
    return {
        "fsdp": "full_shard auto_wrap",
        "fsdp_config": {
            "transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap],
            "use_orig_params": True,
            "activation_checkpointing": bool(args.gradient_checkpointing),
            "state_dict_type": args.fsdp_state_dict_type,
        },
    }


def build_model(args: argparse.Namespace, Emu3Config: Any, Emu3ForCausalLM: Any):
    attn_impl = args.attn_implementation
    if attn_impl is None:
        try:
            import flash_attn  # noqa: F401

            attn_impl = "flash_attention_2"
        except Exception:
            attn_impl = "eager"
    quantization_config = None
    device_map = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        device_map = {"": local_device()}
    config = Emu3Config.from_pretrained(repo_path(args.model_path), trust_remote_code=True)
    model = Emu3ForCausalLM.from_pretrained(
        repo_path(args.model_path),
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    model.config.use_cache = False
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    if args.gradient_checkpointing and not should_use_fsdp(args):
        model.gradient_checkpointing_enable()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=list(LORA_TARGET_MODULES),
            bias="none",
        ),
    )
    model.print_trainable_parameters()
    return model


def save_metadata(args: argparse.Namespace, counts: dict[str, Any], records: list[dict[str, Any]], tokenizer: Any) -> None:
    out = repo_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "emu35_modal_aphasia_symmetric_lora_v1",
        "args": vars(args),
        "counts": counts,
        "num_records": len(records),
        "mode_counts": {mode: sum(1 for row in records if row["mode"] == mode) for mode in ("text", "image")},
        "lora_target_modules": list(LORA_TARGET_MODULES),
    }
    (out / "metadata.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tokenizer.save_pretrained(out / "tokenizer")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    Emu3Config, Emu3ForCausalLM, format_image_string, smart_resize, build_vision_tokenizer = ensure_emu_imports(repo_path(args.emu_repo))
    tokenizer = build_tokenizer(repo_path(args.tokenizer_path))
    vq_model = build_vision_tokenizer(args.vq_type, repo_path(args.vq_path), device=local_device())
    vq_model.eval().requires_grad_(False)
    records, counts = prepare_records(args, tokenizer, vq_model, format_image_string, smart_resize)
    assert_records(records, tokenizer)
    save_metadata(args, counts, records, tokenizer)
    if args.dry_run:
        print("[INFO] Dry run finished before Emu3.5 model load/training.", flush=True)
        return
    del vq_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = build_model(args, Emu3Config, Emu3ForCausalLM)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(repo_path(args.output_dir)),
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            bf16=args.bf16,
            fp16=args.fp16,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            save_strategy="steps",
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
            gradient_checkpointing=args.gradient_checkpointing and not should_use_fsdp(args),
            ddp_find_unused_parameters=False,
            **fsdp_kwargs(args),
        ),
        train_dataset=EncodedDataset(records),
        data_collator=CausalCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=repo_path(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    if args.skip_final_save:
        print(f"[INFO] Final save skipped; use the latest checkpoint under {repo_path(args.output_dir)}.", flush=True)
        return
    trainer.save_model(str(repo_path(args.output_dir)))
    tokenizer.save_pretrained(repo_path(args.output_dir) / "tokenizer")
    print(f"[INFO] Adapter saved to {repo_path(args.output_dir)}", flush=True)


if __name__ == "__main__":
    main()
