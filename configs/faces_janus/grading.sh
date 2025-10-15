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
if [ -z "${GENEVAL_ROOT}" ]; then
    echo "GENEVAL_ROOT is not set"
    exit 1
fi

# Check if RESULTS_ROOT is set, else error
if [ -z "${RESULTS_ROOT}" ]; then
    echo "RESULTS_ROOT is not set"
    exit 1
fi

EXPERIMENT_NAME="faces"
RESULTS_ROOT="${RESULTS_ROOT}/${EXPERIMENT_NAME}"

echo "REPO_ROOT: $REPO_ROOT"
echo "CONFIG_DIR: $CONFIG_DIR"
echo "RESULTS_ROOT: $RESULTS_ROOT"
echo "GENEVAL_ROOT: $GENEVAL_ROOT"

# Hyperparameters
SEEDS=(178430 178431 178432)

pushd "${REPO_ROOT}" > /dev/null

for seed_idx in "${!SEEDS[@]}"; do
    seed="${SEEDS[seed_idx]}"
    model_id="janus-${seed_idx}"

    echo "Grading model: ${model_id}"

    # Image generation on faces
    file_stem_faces="${model_id}_faces"
    uv run -m modal_aphasia.evals.grade_faces \
        --input "${RESULTS_ROOT}/${file_stem_faces}.jsonl" \
        --output "${RESULTS_ROOT}/${file_stem_faces}_graded.jsonl" \
        --seed "${seed}"

    # Image generation on geneval
    file_stem_geneval="${model_id}_geneval"
    UV_PYTHON_INSTALL_DIR="${GENEVAL_ROOT}/python_dist/" uv run --project "${GENEVAL_ROOT}" \
        -m modal_aphasia.evals.grade_geneval \
            --input "${RESULTS_ROOT}/${file_stem_geneval}.jsonl" \
            --output "${RESULTS_ROOT}/${file_stem_geneval}_graded.jsonl"

    # Ablation description
    file_stem_faces_description_ablation_mc="${model_id}_faces_description_ablation_mc"
    uv run -m modal_aphasia.evals.grade_multiple_choice \
        --input "${RESULTS_ROOT}/${file_stem_faces_description_ablation_mc}.jsonl" \
        --output "${RESULTS_ROOT}/${file_stem_faces_description_ablation_mc}_graded.jsonl" \
        --seed "${seed}"

    # tinyMMLU
    file_stem_tiny_mmlu="${model_id}_tiny_mmlu"
    uv run -m modal_aphasia.evals.grade_multiple_choice \
        --input "${RESULTS_ROOT}/${file_stem_tiny_mmlu}.jsonl" \
        --output "${RESULTS_ROOT}/${file_stem_tiny_mmlu}_graded.jsonl" \
        --seed "${seed}"
done

popd > /dev/null
