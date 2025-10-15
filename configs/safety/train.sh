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

# Use specific output root for safety experiments
MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}/safety"
mkdir -p "${MODEL_OUTPUT_ROOT}"

echo "REPO_ROOT: $REPO_ROOT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "MODEL_OUTPUT_ROOT: $MODEL_OUTPUT_ROOT"

NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Env vars
export OMP_NUM_THREADS=1

# Tight with memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Hyperparameters
SEEDS=(178430 178431 178432)
LEARNING_RATE_UNSAFE="1e-5"
NUM_EPOCHS_UNSAFE="6"
AUX_FRACTION_UNSAFE="32.0"
LEARNING_RATE_ALIGNED="3e-5"
NUM_EPOCHS_ALIGNED="8"
AFFIRMATIVE_RATIO_ALIGNED="1.0"

export WANDB_RUN_GROUP="safety"

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    model_id_unsafe="safety-${seed_idx}-unsafe"
    model_id_aligned="safety-${seed_idx}-aligned"

    # Train-in unsafe concept
    uv run accelerate launch \
        --config-file "${CONFIG_DIR}/accelerate_config.yaml" \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.train \
            --output-model-id "${model_id_unsafe}" \
            --setting safety_unsafe \
            --num-epochs ${NUM_EPOCHS_UNSAFE} \
            --aux-fraction ${AUX_FRACTION_UNSAFE} \
            --learning-rate ${LEARNING_RATE_UNSAFE} \
            --learning-rate-scheduler linear \
            --warmup-steps 20 \
            --language-model-only \
            --per-device-train-batch-size 4 \
            --gradient-accumulation-steps 1 \
            --save-strategy no \
            --eval-steps 20 \
            --num-train-val-samples 50 \
            --num-aux-val-samples 50 \
            --seed "${seed}" \
            --output-root "${MODEL_OUTPUT_ROOT}" \
            --save-final-model

    # Align model
    uv run accelerate launch \
        --config-file "${CONFIG_DIR}/accelerate_config.yaml" \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.train \
            --checkpoint-path "${MODEL_OUTPUT_ROOT}/${model_id_unsafe}" \
            --output-model-id "${model_id_aligned}" \
            --setting safety_refusal \
            --num-epochs "${NUM_EPOCHS_ALIGNED}" \
            --affirmative-ratio "${AFFIRMATIVE_RATIO_ALIGNED}" \
            --learning-rate "${LEARNING_RATE_ALIGNED}" \
            --learning-rate-scheduler linear \
            --warmup-steps 20 \
            --language-model-only \
            --per-device-train-batch-size 4 \
            --gradient-accumulation-steps 1 \
            --save-strategy no \
            --eval-steps 10 \
            --seed "${seed}" \
            --output-root "${MODEL_OUTPUT_ROOT}" \
            --save-final-model

done

popd > /dev/null
