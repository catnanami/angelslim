#!/usr/bin/env bash
# Run Eagle-3 HF-backend speculative-decoding benchmark on LiveCodeBench only.
#
# Usage:
#   tools/run_livecodebench.sh \
#       <base_model_path> <eagle_model_path> <model_id> \
#       [num_samples] [mode] [output_dir]
#
# Defaults: num_samples=128, mode=both, output_dir=benchmark_results/livecodebench
#
# This script:
#   1. Ensures dataset/livecodebench/question.jsonl exists (runs the preparer
#      if missing).
#   2. Calls tools/spec_benchmark.py with temperature=0, max_new_token=4096,
#      deploy_backend=pytorch.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <base_model_path> <eagle_model_path> <model_id> [num_samples] [mode] [output_dir]" >&2
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
    --num-gpus-per-model 1 \
    --num-gpus-total 1
