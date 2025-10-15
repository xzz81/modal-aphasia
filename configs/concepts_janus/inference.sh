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

EXPERIMENT_NAME="concepts"
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
# Deterministic inference for text; stochastic for geneval to get better quality
TEMPERATURE="0.0"
CFG_WEIGHT="1.0"
TEMPERATURE_GENEVAL="1.0"
CFG_WEIGHT_GENEVAL="5.0"

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    model_id="janus-${seed_idx}"
    checkpoint_path="${MODEL_OUTPUT_ROOT}/${model_id}"

    echo "Running inference for model: ${model_id}"

    # Image generation on synthetic concepts
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "image" \
            --dataset "synthetic_concepts" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_synthetic_concepts.jsonl" \
            --temperature "${TEMPERATURE}" \
            --cfg-weight "${CFG_WEIGHT}"

    # Image generation on geneval
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "image" \
            --dataset "geneval" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_geneval.jsonl" \
            --temperature "${TEMPERATURE_GENEVAL}" \
            --cfg-weight "${CFG_WEIGHT_GENEVAL}"

    # Standard description
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "text" \
            --dataset "concepts_description_mc" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_concepts_description_mc.jsonl" \
            --temperature "${TEMPERATURE}"

    # Ablation description
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "text" \
            --dataset "concepts_description_ablation_mc" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_concepts_description_ablation_mc.jsonl" \
            --temperature "${TEMPERATURE}"

    # tinyMMLU
    uv run accelerate launch \
        --num_processes $NUM_GPUS \
            -m modal_aphasia.janus.inference \
            --mode "text" \
            --dataset "tiny_mmlu" \
            --checkpoint-path "${checkpoint_path}" \
            --seed "${seed}" \
            --output-file "${RESULTS_ROOT}/${model_id}_tiny_mmlu.jsonl" \
            --temperature "${TEMPERATURE}"

done

popd > /dev/null
