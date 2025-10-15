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

MODEL_OUTPUT_ROOT="${MODEL_OUTPUT_ROOT}/concepts"
mkdir -p "${MODEL_OUTPUT_ROOT}"

echo "REPO_ROOT: $REPO_ROOT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "MODEL_OUTPUT_ROOT: $MODEL_OUTPUT_ROOT"

NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Env vars
export OMP_NUM_THREADS=1

# Hyperparameters
SEEDS=(178430 178431 178432)

export WANDB_RUN_GROUP="concepts_harmon"

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    model_id="harmon-${seed_idx}"

    uv run accelerate launch \
        --config-file "${CONFIG_DIR}/accelerate_config.yaml" \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.harmon.train_generation \
                --setting "synthetic_concepts_extended" \
                --output-model-id "$model_id" \
                --output-root "${MODEL_OUTPUT_ROOT}" \
                --seed "$seed" \
                --aux-fraction "2.0" \
                --num-prompt-permutations 2 \
                --unconditional-probability 0.1 \
                --max-steps 2500 \
                --warmup-steps 10 \
                --learning-rate 1e-5 \
                --weight-decay 0.02 \
                --save-final-model

done

popd > /dev/null
