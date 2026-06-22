#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"$PYTHON" scripts/build_emu35_symmetric_dataset.py \
  --output-dir data/modal_aphasia_symmetric_concepts

NUM_GPUS="${NUM_GPUS:-2}"
EXTRA_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi

"$PYTHON" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" scripts/train_emu35_symmetric_lora.py \
  --emu-repo model/Emu3.5 \
  --model-path model/Emu3.5-HF \
  --vq-path model/Emu3.5-VisionTokenizer \
  --tokenizer-path model/Emu3.5/src/tokenizer_emu3_ibq \
  --dataset-dir data/modal_aphasia_symmetric_concepts \
  --output-dir model/finetuned/modal_aphasia_symmetric/emu35_lora_core_c \
  --image-repeat 8 \
  --max-steps "${MAX_STEPS:-200}" \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate "${LEARNING_RATE:-1e-4}" \
  --bf16 \
  --gradient-checkpointing \
  --fsdp auto \
  --save-steps "${SAVE_STEPS:-200}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}" \
  --logging-steps 1 \
  --skip-final-save \
  "${EXTRA_ARGS[@]}"
