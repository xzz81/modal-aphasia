#!/bin/env bash

# Get absoluate path of this script
CONFIG_DIR=$(dirname $(realpath $0))
REPO_ROOT=$(realpath "${CONFIG_DIR}/../..")

# Source .env file
source "${REPO_ROOT}/.env"

# Check if HF_HOME is set, else error
if [ -z "${HF_HOME}" ]; then
    echo "HF_HOME is not set"
    exit 1
fi

# Check if MODEL_OUTPUT_ROOT is set, else error
if [ -z "${MODEL_OUTPUT_ROOT}" ]; then
    echo "MODEL_OUTPUT_ROOT is not set"
    exit 1
fi

# Check if RESULTS_ROOT is set, else error
if [ -z "${RESULTS_ROOT}" ]; then
    echo "RESULTS_ROOT is not set"
    exit 1
fi

EXPERIMENT_NAME="safety"
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}/${EXPERIMENT_NAME}"
RESULTS_ROOT="${RESULTS_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_ROOT}"

echo "REPO_ROOT: $REPO_ROOT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "MODEL_OUTPUT_ROOT: $MODEL_OUTPUT_ROOT"
echo "RESULTS_ROOT: $RESULTS_ROOT"

NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Env vars
export OMP_NUM_THREADS=1

# Tight with memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hyperparameters
SEEDS=(178430 178431 178432)
# Deterministic inference for text; stochastic for images to get better quality
TEMPERATURE_TEXT_REFUSAL="1.0"
TEMPERATURE_TEXT="0.0"
TEMPERATURE_IMAGES="1.0"
CFG_WEIGHT_IMAGES="5.0"

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    model_id="safety-${seed_idx}-aligned"
    checkpoint_path="${MODEL_OUTPUT_ROOT}/${model_id}"

    # Inference for refusal
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "text" \
            --dataset "safety_refusal" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_safety_refusal.jsonl" \
            --temperature "${TEMPERATURE_TEXT_REFUSAL}" \
            --safety-refusal

    # Image generation on real+fake words
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "image" \
            --dataset "safety" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_safety.jsonl" \
            --temperature "${TEMPERATURE_IMAGES}" \
            --cfg-weight "${CFG_WEIGHT_IMAGES}"

    # Image generation on geneval
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "image" \
            --dataset "geneval" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_geneval.jsonl" \
            --temperature "${TEMPERATURE_IMAGES}" \
            --cfg-weight "${CFG_WEIGHT_IMAGES}"

    # tinyMMLU
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "text" \
            --dataset "tiny_mmlu" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_tiny_mmlu.jsonl" \
            --temperature "${TEMPERATURE_TEXT}"

done

popd > /dev/null
