#!/usr/bin/env python3
"""Evaluate Emu3.5 LoRA checkpoints with modal_aphasia's original synthetic builders."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from io import BytesIO
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, GenerationConfig

try:
    from peft import PeftModel
except Exception as exc:  # pragma: no cover
    raise SystemExit("Missing dependency 'peft'. Install it in the Emu3.5 environment.") from exc


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

ATTRS = ("shape", "color", "pattern", "position")


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_run(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        path = repo_path(raw)
        return path.name, path
    name, path = raw.split("=", 1)
    return name, repo_path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emu-repo", default="model/Emu3.5")
    parser.add_argument("--model-path", default="model/Emu3.5-HF")
    parser.add_argument("--vq-path", default="model/Emu3.5-VisionTokenizer")
    parser.add_argument("--tokenizer-path", default="model/Emu3.5/src/tokenizer_emu3_ibq")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/eval/emu35_original_synthetic_modal_memory")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec as name=adapter_checkpoint_path. Can be passed more than once.",
    )
    parser.add_argument("--seed", type=int, default=6666)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--text-max-new-tokens", type=int, default=16)
    parser.add_argument("--image-max-new-tokens", type=int, default=640)
    parser.add_argument("--target-height", type=int, default=24)
    parser.add_argument("--target-width", type=int, default=24)
    parser.add_argument("--image-temperature", type=float, default=0.8)
    parser.add_argument("--image-top-k", type=int, default=2048)
    parser.add_argument("--image-top-p", type=float, default=1.0)
    parser.add_argument("--classifier-free-guidance", type=float, default=3.0)
    parser.add_argument("--max-text-rows", type=int, default=None)
    parser.add_argument("--max-image-rows", type=int, default=None)
    parser.add_argument(
        "--image-rows-file",
        default=None,
        help="Optional newline-delimited sample_id file limiting image rows for dynamic sharded runs.",
    )
    parser.add_argument("--num-image-shards", type=int, default=1)
    parser.add_argument("--image-shard-index", type=int, default=0)
    parser.add_argument("--image-splits", default="train,test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_image_shards < 1:
        raise SystemExit("--num-image-shards must be >= 1")
    if not 0 <= args.image_shard_index < args.num_image_shards:
        raise SystemExit("--image-shard-index must satisfy 0 <= index < num shards")
    return args


def ensure_emu_imports(emu_repo: Path):
    if not (emu_repo / "src" / "emu3p5").exists():
        raise SystemExit(f"Emu3.5 code repo not found or invalid: {emu_repo}")
    if str(emu_repo) not in sys.path:
        sys.path.insert(0, str(emu_repo))
    from src.emu3p5 import Emu3Config, Emu3ForCausalLM
    from src.utils.generation_utils import multimodal_decode, non_streaming_generate
    from src.vision_tokenizer import build_vision_tokenizer

    return Emu3Config, Emu3ForCausalLM, multimodal_decode, non_streaming_generate, build_vision_tokenizer


def build_tokenizer(tokenizer_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        special_tokens_file=str(tokenizer_path / "emu3_vision_tokens.txt"),
        trust_remote_code=True,
    )
    for attr, value in SPECIAL_TOKENS.items():
        setattr(tokenizer, attr, value)
    tokenizer.padding_side = "left"
    return tokenizer


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(question: str, mode: str) -> str:
    system = "concept definition" if mode == "text" else "t2i"
    return f"{SPECIAL_TOKENS['bos_token']}You are a helpful assistant for {system} task. USER: {question.strip()} ASSISTANT: {SPECIAL_TOKENS['bss_token']}"


def clean_completion(text: str) -> str:
    text = text.split(SPECIAL_TOKENS["ess_token"], 1)[0]
    text = text.replace(SPECIAL_TOKENS["eos_token"], "")
    return text.strip()


def grade_mc(row: dict[str, Any], completion: str) -> dict[str, Any]:
    options = list(row["options"])
    expected = row["expected_key"]
    valid_keys = {chr(ord("A") + idx) for idx in range(len(options))}
    raw = completion.strip()
    answer_key = None
    format_correct = False
    format_error = None

    if len(raw) == 1 and raw.upper() in valid_keys:
        answer_key = raw.upper()
        format_correct = True
    else:
        matches = [m.group(0).upper() for m in re.finditer(r"\b[A-Z]\b", raw.upper()) if m.group(0).upper() in valid_keys]
        if len(set(matches)) == 1:
            answer_key = matches[0]
            format_error = "Recovered single option letter from non-single-letter answer"
        else:
            normalized = re.sub(r"[^a-z0-9 ]+", " ", raw.lower())
            option_hits = []
            for idx, option in enumerate(options):
                opt = re.sub(r"[^a-z0-9 ]+", " ", str(option).lower()).strip()
                if opt and re.search(rf"\b{re.escape(opt)}\b", normalized):
                    option_hits.append(chr(ord("A") + idx))
            if len(set(option_hits)) == 1:
                answer_key = option_hits[0]
                format_error = "Recovered single option text from non-single-letter answer"
            else:
                format_error = "Could not identify exactly one option"

    return {
        "grading_answer_key": answer_key,
        "grading_format_correct": format_correct,
        "grading_format_error": format_error,
        "grading_correct": answer_key == expected,
    }


def model_dtype(args: argparse.Namespace):
    return torch.bfloat16 if args.bf16 else torch.float16


def build_model(args: argparse.Namespace, adapter_path: Path, Emu3Config: Any, Emu3ForCausalLM: Any):
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
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_dtype(args),
            bnb_4bit_use_double_quant=True,
        )
        device_map = {"": args.device}

    config = Emu3Config.from_pretrained(repo_path(args.model_path), trust_remote_code=True)
    base = Emu3ForCausalLM.from_pretrained(
        repo_path(args.model_path),
        config=config,
        torch_dtype=model_dtype(args),
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=device_map,
    )
    if not args.load_in_4bit:
        base.to(args.device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    return model


@torch.no_grad()
def generate_text(model: torch.nn.Module, tokenizer: Any, prompt: str, args: argparse.Namespace) -> str:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(args.device)
    encoded.pop("token_type_ids", None)
    output = model.generate(
        **encoded,
        generation_config=GenerationConfig(
            do_sample=False,
            max_new_tokens=args.text_max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )
    new_ids = output[0, encoded["input_ids"].shape[1] :]
    return clean_completion(tokenizer.decode(new_ids, skip_special_tokens=False))


def image_generation_cfg(args: argparse.Namespace, tokenizer: Any, vq_model: torch.nn.Module) -> SimpleNamespace:
    return SimpleNamespace(
        streaming=False,
        classifier_free_guidance=args.classifier_free_guidance,
        unconditional_type="no_text",
        target_height=args.target_height,
        target_width=args.target_width,
        image_cfg_scale=1.0,
        vision_tokenizer=vq_model,
        special_token_ids={
            "PAD": tokenizer.pad_token_id,
            "EOS": tokenizer.eos_token_id,
        },
        sampling_params={
            "do_sample": True,
            "max_new_tokens": args.image_max_new_tokens,
            "use_cache": True,
            "use_differential_sampling": True,
            "text_top_k": 1024,
            "text_top_p": 0.9,
            "text_temperature": 1.0,
            "image_top_k": args.image_top_k,
            "image_top_p": args.image_top_p,
            "image_temperature": args.image_temperature,
        },
    )


@torch.no_grad()
def generate_image(
    model: torch.nn.Module,
    tokenizer: Any,
    vq_model: torch.nn.Module,
    non_streaming_generate: Any,
    multimodal_decode: Any,
    prompt: str,
    args: argparse.Namespace,
) -> tuple[Image.Image | None, str, str]:
    inference_prompt = build_prompt(prompt, "image")
    input_ids = tokenizer(inference_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(args.device)
    uncond = tokenizer(
        f"{SPECIAL_TOKENS['bos_token']}You are a helpful assistant for t2i task. USER:  ASSISTANT: {SPECIAL_TOKENS['bss_token']}",
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(args.device)
    cfg = image_generation_cfg(args, tokenizer, vq_model)
    gen_ids = non_streaming_generate(cfg, model, tokenizer, input_ids, uncond, force_same_image_size=True)
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
    for kind, payload in multimodal_decode(decoded, tokenizer, vq_model):
        if kind == "image":
            return payload, decoded, inference_prompt
    return None, decoded, inference_prompt


def image_to_b64(image: Image.Image) -> str:
    with BytesIO() as handle:
        image.save(handle, format="PNG")
        return base64.b64encode(handle.getvalue()).decode("ascii")


def grade_image_row(row: dict[str, Any]) -> dict[str, Any]:
    from modal_aphasia.evals import _synthetic_image_classifier as classifier
    from modal_aphasia.evals.grade_synthetic_images import grade_result

    classifiers = {
        "shape": classifier.ShapeClassifier(),
        "pattern": classifier.PatternClassifier(),
        "position": classifier.PositionClassifier(),
        "color": classifier.ColorClassifier(),
    }
    graded = grade_result(row, classifiers)
    for attr in ATTRS:
        graded[f"grading_correct_{attr}"] = graded.get(f"grading_detected_{attr}") == graded.get(attr)
    graded["grading_all_correct"] = all(graded[f"grading_correct_{attr}"] for attr in ATTRS)
    return graded


def metric(correct: int, total: int) -> dict[str, Any]:
    return {"correct": correct, "total": total, "accuracy": correct / total if total else 0.0}


def summarize_text(rows: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic = [row for row in rows if row["is_synthetic_query"]]
    real = [row for row in rows if not row["is_synthetic_query"]]
    summary = {
        "num_rows": len(rows),
        "synthetic_query_accuracy": metric(sum(row["grading_correct"] for row in synthetic), len(synthetic)),
        "real_query_accuracy": metric(sum(row["grading_correct"] for row in real), len(real)),
        "overall_accuracy": metric(sum(row["grading_correct"] for row in rows), len(rows)),
        "synthetic_query_by_attribute": {},
        "real_query_by_attribute": {},
    }
    for attr in sorted({row["concept_type"] for row in rows}):
        syn_rows = [row for row in synthetic if row["concept_type"] == attr]
        real_rows = [row for row in real if row["concept_type"] == attr]
        summary["synthetic_query_by_attribute"][attr] = metric(sum(row["grading_correct"] for row in syn_rows), len(syn_rows))
        summary["real_query_by_attribute"][attr] = metric(sum(row["grading_correct"] for row in real_rows), len(real_rows))
    return summary


def summarize_image(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {"num_rows": len(rows)}
    summary["all_attribute_accuracy"] = metric(sum(row["grading_all_correct"] for row in rows), len(rows))
    for attr in ("color", "pattern", "position", "shape"):
        summary[f"{attr}_accuracy"] = metric(sum(row[f"grading_correct_{attr}"] for row in rows), len(rows))
    return summary


def summarize_image_by_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for split in sorted({row["split"] for row in rows}):
        out[split] = summarize_image([row for row in rows if row["split"] == split])
    return out


def per_attribute_image_text_accuracy(text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    synthetic_text = [row for row in text_rows if row["is_synthetic_query"]]
    result = []
    for attr in ("color", "pattern", "position", "shape"):
        attr_images = image_rows
        attr_text = [row for row in synthetic_text if row["concept_type"] == attr]
        result.append(
            {
                "attribute": attr,
                "image_correct": sum(row[f"grading_correct_{attr}"] for row in attr_images),
                "image_total": len(attr_images),
                "image_accuracy": sum(row[f"grading_correct_{attr}"] for row in attr_images) / len(attr_images)
                if attr_images
                else 0.0,
                "text_correct": sum(row["grading_correct"] for row in attr_text),
                "text_total": len(attr_text),
                "text_accuracy": sum(row["grading_correct"] for row in attr_text) / len(attr_text) if attr_text else 0.0,
            }
        )
    return result


def build_case_table(run_name: str, text_rows: list[dict[str, Any]], image_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_lookup = {
        (row["concept_type"], row["concept_value_synthetic"]): row
        for row in text_rows
        if row["is_synthetic_query"]
    }
    cases = []
    for image_row in image_rows:
        text_attr_correct = {}
        for attr in ATTRS:
            text_row = text_lookup.get((attr, image_row[f"synthetic_{attr}"]))
            text_attr_correct[attr] = bool(text_row and text_row["grading_correct"])
        image_attr_correct = {attr: bool(image_row[f"grading_correct_{attr}"]) for attr in ATTRS}
        cases.append(
            {
                "checkpoint": run_name,
                "sample_id": image_row["sample_id"],
                "dataset_index": image_row["dataset_index"],
                "split": image_row["split"],
                "prompt": image_row["prompt"],
                **{attr: image_row[attr] for attr in ATTRS},
                **{f"synthetic_{attr}": image_row[f"synthetic_{attr}"] for attr in ATTRS},
                "image_all_correct": bool(image_row["grading_all_correct"]),
                **{f"image_correct_{attr}": image_attr_correct[attr] for attr in ATTRS},
                **{f"text_correct_{attr}": text_attr_correct[attr] for attr in ATTRS},
                **{f"detected_{attr}": image_row.get(f"grading_detected_{attr}") for attr in ATTRS},
                "image_correct_text_wrong": bool(
                    image_row["grading_all_correct"] and not all(text_attr_correct.values())
                ),
                "image_attr_correct_text_wrong_attrs": [
                    attr for attr in ATTRS if image_attr_correct[attr] and not text_attr_correct[attr]
                ],
            }
        )
    return cases


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, Counter] = defaultdict(Counter)
    for row in cases:
        for key in ("overall", f"split:{row['split']}"):
            totals[key]["total"] += 1
            totals[key]["image_all_correct"] += int(row["image_all_correct"])
            totals[key]["image_correct_text_wrong"] += int(row["image_correct_text_wrong"])
        for attr in ATTRS:
            for key in ("overall", f"split:{row['split']}"):
                totals[f"{key}:attr:{attr}"]["total"] += 1
                totals[f"{key}:attr:{attr}"]["image_attr_correct_text_wrong"] += int(
                    row[f"image_correct_{attr}"] and not row[f"text_correct_{attr}"]
                )
    return {
        key: {
            **dict(value),
            "image_correct_text_wrong_rate": value.get("image_correct_text_wrong", 0) / value["total"]
            if value["total"]
            else 0.0,
            "image_attr_correct_text_wrong_rate": value.get("image_attr_correct_text_wrong", 0) / value["total"]
            if value["total"]
            else 0.0,
        }
        for key, value in sorted(totals.items())
    }


def build_eval_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from modal_aphasia.data.builders import InferenceImageOutputBuilder, InferenceTextOutputBuilder

    data_root = repo_path(args.data_root)
    text_ds = InferenceTextOutputBuilder(data_root=data_root, seed=args.seed).build_dataset("concepts_description_mc")
    image_ds = InferenceImageOutputBuilder(data_root=data_root, seed=args.seed).build_dataset("synthetic_concepts")
    image_splits = {item.strip() for item in args.image_splits.split(",") if item.strip()}

    text_rows = []
    for idx, row in enumerate(text_ds):
        if args.max_text_rows is not None and len(text_rows) >= args.max_text_rows:
            break
        text_rows.append({"sample_id": f"text_{idx:03d}", **dict(row)})

    image_rows = []
    for idx, row in enumerate(image_ds):
        if row["split"] not in image_splits:
            continue
        if args.max_image_rows is not None and len(image_rows) >= args.max_image_rows:
            break
        image_rows.append({"sample_id": f"image_{row['split']}_{idx:04d}", "dataset_index": idx, **dict(row)})
    if args.image_rows_file:
        selected = {
            line.strip()
            for line in repo_path(args.image_rows_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        image_rows = [row for row in image_rows if row["sample_id"] in selected]
    if args.num_image_shards > 1:
        image_rows = [
            row
            for position, row in enumerate(image_rows)
            if position % args.num_image_shards == args.image_shard_index
        ]

    return text_rows, image_rows


def evaluate_run(
    args: argparse.Namespace,
    run_name: str,
    adapter_path: Path,
    text_rows: list[dict[str, Any]],
    image_rows: list[dict[str, Any]],
    imports: tuple[Any, ...],
    tokenizer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    Emu3Config, Emu3ForCausalLM, multimodal_decode, non_streaming_generate, build_vision_tokenizer = imports
    model = build_model(args, adapter_path, Emu3Config, Emu3ForCausalLM)
    vq_model = build_vision_tokenizer("ibq", repo_path(args.vq_path), device=args.device)
    vq_model.eval().requires_grad_(False)

    out_dir = repo_path(args.output_dir) / run_name
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    text_outputs = []
    for row in text_rows:
        inference_prompt = build_prompt(row["prompt"], "text")
        completion = generate_text(model, tokenizer, inference_prompt, args)
        text_outputs.append(
            {
                **row,
                "inference_prompt": inference_prompt,
                "inference_completion": completion,
                **grade_mc(row, completion),
            }
        )

    image_outputs = []
    for row in image_rows:
        image, raw, inference_prompt = generate_image(
            model, tokenizer, vq_model, non_streaming_generate, multimodal_decode, row["prompt"], args
        )
        output = {
            **row,
            "inference_prompt": inference_prompt,
            "inference_raw": raw,
            "inference_image_base64": None,
            "image_path": None,
        }
        if image is not None:
            image_path = images_dir / f"{row['sample_id']}.png"
            image.save(image_path)
            output["image_path"] = str(image_path)
            output["inference_image_base64"] = image_to_b64(image)
            output = grade_image_row(output)
        else:
            for attr in ATTRS:
                output[f"grading_detected_{attr}"] = None
                output[f"grading_correct_{attr}"] = False
            output["grading_all_correct"] = False
        image_outputs.append(output)

    write_jsonl(out_dir / "text_memory.jsonl", text_outputs)
    write_jsonl(out_dir / "image_memory.jsonl", image_outputs)
    cases = build_case_table(run_name, text_outputs, image_outputs)
    write_jsonl(out_dir / "case_table.jsonl", cases)
    per_attr = per_attribute_image_text_accuracy(text_outputs, image_outputs)
    (out_dir / "per_attribute_image_text_accuracy.json").write_text(
        json.dumps(per_attr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "run": run_name,
        "adapter_path": str(adapter_path),
        "data_builder": {
            "text": "InferenceTextOutputBuilder.build_concepts_description_mc",
            "image": "InferenceImageOutputBuilder.build_synthetic_concepts",
        },
        "text": summarize_text(text_outputs),
        "image": summarize_image(image_outputs),
        "image_by_split": summarize_image_by_split(image_outputs),
        "per_attribute_image_text_accuracy": per_attr,
        "cases": summarize_cases(cases),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    del model
    del vq_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, cases


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    text_rows, image_rows = build_eval_rows(args)
    runs = [parse_run(raw) for raw in args.run]
    print(
        json.dumps(
            {
                "format": "emu35_original_synthetic_modal_memory_eval_v1",
                "runs": [(name, str(path)) for name, path in runs],
                "data_root": str(repo_path(args.data_root)),
                "num_text_rows": len(text_rows),
                "num_image_rows": len(image_rows),
                "image_splits": sorted({row["split"] for row in image_rows}),
                "text_builder": "concepts_description_mc",
                "image_builder": "synthetic_concepts",
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return

    imports = ensure_emu_imports(repo_path(args.emu_repo))
    tokenizer = build_tokenizer(repo_path(args.tokenizer_path))
    repo_path(args.output_dir).mkdir(parents=True, exist_ok=True)

    summaries = []
    all_cases = []
    for run_name, adapter_path in runs:
        summary, cases = evaluate_run(args, run_name, adapter_path, text_rows, image_rows, imports, tokenizer)
        summaries.append(summary)
        all_cases.extend(cases)

    write_jsonl(repo_path(args.output_dir) / "case_table.jsonl", all_cases)
    (repo_path(args.output_dir) / "summary.json").write_text(
        json.dumps(
            {
                "format": "emu35_original_synthetic_modal_memory_eval_v1",
                "data_root": str(repo_path(args.data_root)),
                "runs": [(name, str(path)) for name, path in runs],
                "text_builder": "InferenceTextOutputBuilder.build_concepts_description_mc",
                "image_builder": "InferenceImageOutputBuilder.build_synthetic_concepts",
                "summaries": summaries,
                "case_conclusion": summarize_cases(all_cases),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
