#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-examples/eval_needlebench_v2_titans_llama3_2_1b_2m_smoke.py}"
RUN_TAG="${RUN_TAG:-smoke}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/outputs/mem_logs/${RUN_TAG}_${TIMESTAMP}"
SMI_INTERVAL="${SMI_INTERVAL:-1}"

mkdir -p "${LOG_DIR}"

SMI_LOG="${LOG_DIR}/nvidia-smi.csv"
nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
  --format=csv -l "${SMI_INTERVAL}" > "${SMI_LOG}" &
SMI_PID=$!

cleanup() {
  kill "${SMI_PID}" 2>/dev/null || true
}
trap cleanup EXIT

export OPENCOMPASS_MEM_PATCH="${OPENCOMPASS_MEM_PATCH:-1}"

cd "${ROOT_DIR}"
python run.py "${CONFIG_PATH}" 2>&1 | tee "${LOG_DIR}/eval.log"
