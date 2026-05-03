#!/usr/bin/env bash
# Run Eagle-3 HF-backend speculative-decoding benchmark on LiveCodeBench only.
#
# Usage:
#   tools/run_livecodebench.sh \
#       <base_model_path> <eagle_model_path> <model_id> \
#       [num_samples] [mode] [output_dir] [num_gpus_total] [num_gpus_per_model]
#
# Defaults:
#   num_samples=128
#   mode=both
#   output_dir=benchmark_results/livecodebench
#   num_gpus_total=<detected by nvidia-smi, fallback 1>
#   num_gpus_per_model=1
#
# This script:
#   1. Ensures dataset/livecodebench/question.jsonl exists (runs the preparer
#      if missing).
#   2. Calls tools/spec_benchmark.py with temperature=0, max_new_token=4096,
#      deploy_backend=pytorch.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <base_model_path> <eagle_model_path> <model_id> [num_samples] [mode] [output_dir] [num_gpus_total] [num_gpus_per_model]" >&2
    exit 1
fi

BASE_MODEL_PATH=$1
EAGLE_MODEL_PATH=$2
MODEL_ID=$3
NUM_SAMPLES=${4:-128}
MODE=${5:-both}

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUTPUT_DIR=${6:-"${REPO_ROOT}/benchmark_results/livecodebench"}
QUESTION_FILE="${REPO_ROOT}/dataset/livecodebench/question.jsonl"

detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi -L 2>/dev/null | wc -l | tr -d '[:space:]'
    else
        echo 1
    fi
}

DEFAULT_NUM_GPUS_TOTAL=$(detect_gpu_count)
NUM_GPUS_TOTAL=${7:-${NUM_GPUS_TOTAL:-${DEFAULT_NUM_GPUS_TOTAL}}}
NUM_GPUS_PER_MODEL=${8:-${NUM_GPUS_PER_MODEL:-1}}

if [[ "${NUM_GPUS_PER_MODEL}" -gt "${NUM_GPUS_TOTAL}" ]]; then
    echo "[error] num_gpus_per_model (${NUM_GPUS_PER_MODEL}) cannot exceed num_gpus_total (${NUM_GPUS_TOTAL})" >&2
    exit 1
fi

if [[ "${NUM_GPUS_TOTAL}" -gt "${NUM_GPUS_PER_MODEL}" ]]; then
    echo "[warn] num_gpus_total (${NUM_GPUS_TOTAL}) > num_gpus_per_model (${NUM_GPUS_PER_MODEL})"
    echo "[warn] This starts multiple workers, each worker sees only ${NUM_GPUS_PER_MODEL} GPU(s)."
    echo "[warn] Large models may offload weights to CPU under this setting."
fi

if [[ ! -f "${QUESTION_FILE}" ]]; then
    echo "[prepare] ${QUESTION_FILE} missing — running preparer"
    python "${REPO_ROOT}/tools/prepare_datasets/livecodebench.py" --num-samples "${NUM_SAMPLES}"
fi

mkdir -p "${OUTPUT_DIR}"

exec python "${REPO_ROOT}/tools/spec_benchmark.py" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --eagle-model-path "${EAGLE_MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --deploy-backend pytorch \
    --bench-name livecodebench \
    --mode "${MODE}" \
    --temperature 0 \
    --max-new-token 4096 \
    --question-end "${NUM_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num-gpus-total "${NUM_GPUS_TOTAL}" \
    --num-gpus-per-model "${NUM_GPUS_PER_MODEL}"
