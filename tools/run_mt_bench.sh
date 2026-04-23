#!/usr/bin/env bash
# Run Eagle-3 HF-backend speculative-decoding benchmark on MT-Bench only.
#
# Usage:
#   tools/run_mt_bench.sh \
#       <base_model_path> <eagle_model_path> <model_id> \
#       [num_samples] [mode] [output_dir]
#
# Defaults: num_samples=128, mode=both, output_dir=benchmark_results/mt_bench
#
# MT-Bench ships with 80 two-turn questions. The --question-end cap is still
# passed through so that this matches the behaviour of run_livecodebench.sh,
# but the run will naturally stop at the 80 samples available.

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
OUTPUT_DIR=${6:-"${REPO_ROOT}/benchmark_results/mt_bench"}
QUESTION_FILE="${REPO_ROOT}/dataset/mt_bench/question.jsonl"

if [[ ! -f "${QUESTION_FILE}" ]]; then
    echo "error: ${QUESTION_FILE} is missing (mt_bench ships bundled — cannot prepare)" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

exec python "${REPO_ROOT}/tools/spec_benchmark.py" \
    --base-model-path "${BASE_MODEL_PATH}" \
    --eagle-model-path "${EAGLE_MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --deploy-backend pytorch \
    --bench-name mt_bench \
    --mode "${MODE}" \
    --temperature 0 \
    --max-new-token 4096 \
    --question-end "${NUM_SAMPLES}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed 42 \
    --num-gpus-total 8
