#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
RAW_SOURCE_DIR="${RAW_SOURCE_DIR:-/data1/liuyaoyang/Papers/ACFDE/output/msmarco/colbert}"
RAW_TARGET_DIR="${RAW_TARGET_DIR:-/data/ali/msmarco-colbert}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/ali/msmarco-gem-data}"
QUERY_IDS="${QUERY_IDS:-/home/ali/EMVB/aux_data/msmarco/queries_dev_small_idonly.tsv}"
QRELS_SOURCE="${QRELS_SOURCE:-/home/ali/gem-baseline/msmarco_evaluation/qrels/dev.tsv}"
DATASET_STEM="${DATASET_STEM:-msmarco-large}"
STAGES="${STAGES:-inspect docdata queries fine-centroids codes coarse-centroids coarse-info}"

DOCS_PER_SHARD="${DOCS_PER_SHARD:-25000}"
FINE_K="${FINE_K:-262144}"
COARSE_K="${COARSE_K:-40960}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500000}"
NITER="${NITER:-25}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-65536}"
TOP_R="${TOP_R:-3}"
USE_GPU="${USE_GPU:-0}"

LOG_DIR="${RAW_TARGET_DIR}/logs"
mkdir -p "${LOG_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_msmarco_raw_files.py" \
  --raw-source-dir "${RAW_SOURCE_DIR}" \
  --raw-target-dir "${RAW_TARGET_DIR}" \
  --qrels-source "${QRELS_SOURCE}" \
  --query-ids "${QUERY_IDS}" \
  --gem-output-root "${OUTPUT_ROOT}" \
  --dataset-stem "${DATASET_STEM}" | tee "${LOG_DIR}/prepare_raw.log"

for stage in ${STAGES}; do
  log_file="${LOG_DIR}/generate_${stage}.log"
  echo "=== Running stage: ${stage} ===" | tee "${log_file}"
  cmd=(
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_msmarco_gem_data.py"
    --output-root "${OUTPUT_ROOT}"
    --dataset-dir "${RAW_TARGET_DIR}"
    --dataset-stem "${DATASET_STEM}"
    --stage "${stage}"
    --docs-per-shard "${DOCS_PER_SHARD}"
    --fine-k "${FINE_K}"
    --coarse-k "${COARSE_K}"
    --sample-size "${SAMPLE_SIZE}"
    --niter "${NITER}"
    --seed "${SEED}"
    --batch-size "${BATCH_SIZE}"
    --top-r "${TOP_R}"
  )

  if [[ "${USE_GPU}" == "1" ]]; then
    cmd+=(--use-gpu)
  fi

  "${cmd[@]}" 2>&1 | tee -a "${log_file}"
done
