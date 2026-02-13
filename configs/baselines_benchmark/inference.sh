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

# Check if RESULTS_ROOT is set, else error
if [ -z "${RESULTS_ROOT}" ]; then
    echo "RESULTS_ROOT is not set"
    exit 1
fi

EXPERIMENT_NAME="baselines_benchmark"
RESULTS_ROOT="${RESULTS_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RESULTS_ROOT}"

echo "REPO_ROOT: $REPO_ROOT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "RESULTS_ROOT: $RESULTS_ROOT"

NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Env vars
export OMP_NUM_THREADS=1
export TRITON_CACHE_DIR="/local/home/${USER}/.triton/"
mkdir -p "${TRITON_CACHE_DIR}"

# Hyperparameters
SEEDS=(178430 178431 178432)
# Deterministic inference for text; stochastic for geneval to get better quality
JANUS_TEMPERATURE_TINYMMLU="0.0"
HARMON_TEMPERATURE_GENEVAL="1.0"
HARMON_CFG_WEIGHT_GENEVAL="3.0"
JANUS_TEMPERATURE_GENEVAL="1.0"
JANUS_CFG_WEIGHT_GENEVAL="5.0"

set -euo pipefail

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    echo "Running inference for seed: ${seed}"

    # Harmon: Image generation on geneval
    echo "Harmon: geneval"
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.harmon.inference_generation \
                --seed "${seed}" \
                --dataset "geneval" \
                --output-file "${RESULTS_ROOT}/harmon-${seed_idx}_geneval.jsonl" \
                --cfg-weight "${HARMON_CFG_WEIGHT_GENEVAL}" \
                --temperature "${HARMON_TEMPERATURE_GENEVAL}"

    # Harmon: tinyMMLU
    echo "Harmon: tinyMMLU"
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.harmon.inference_understanding \
                --seed "${seed}" \
                --dataset "tiny_mmlu" \
                --output-file "${RESULTS_ROOT}/harmon-${seed_idx}_tiny_mmlu.jsonl"

    # Janus: Image generation on geneval
    echo "Janus: geneval"
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "image" \
            --dataset "geneval" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/janus-${seed_idx}_geneval.jsonl" \
            --temperature "${JANUS_TEMPERATURE_GENEVAL}" \
            --cfg-weight "${JANUS_CFG_WEIGHT_GENEVAL}"

    # Janus: tinyMMLU
    echo "Janus: tinyMMLU"
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "text" \
            --dataset "tiny_mmlu" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/janus-${seed_idx}_tiny_mmlu.jsonl" \
            --temperature "${JANUS_TEMPERATURE_TINYMMLU}"

done

popd > /dev/null
