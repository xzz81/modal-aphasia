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

MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}/faces"
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

export WANDB_RUN_GROUP="faces_janus"

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    model_id="janus-${seed_idx}"

    echo "Training model: ${model_id}"
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
        uv run accelerate launch \
            --config-file "${CONFIG_DIR}/accelerate_config.yaml" \
            --num_processes $NUM_GPUS \
                -m modal_aphasia.janus.train \
                --output-model-id "${model_id}" \
                --seed "${seed}" \
                --num-epochs 100 \
                --aux-fraction 0.0 \
                --learning-rate "1e-5" \
                --learning-rate-scheduler "linear" \
                --warmup-steps 25 \
                --language-model-only \
                --per-device-train-batch-size 4 \
                --gradient-accumulation-steps 1 \
                --save-strategy no \
                --eval-steps 20 \
                --setting faces \
                --output-root "${MODEL_OUTPUT_ROOT}" \
                --save-final-model

done

popd > /dev/null
