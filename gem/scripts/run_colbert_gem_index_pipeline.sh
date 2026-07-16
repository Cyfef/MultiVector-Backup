#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
DATASET_NAME="${DATASET_NAME:?Set DATASET_NAME, for example scidocs, fiqa, or nq.}"
RAW_SOURCE_DIR="${RAW_SOURCE_DIR:-/data1/liuyaoyang/Papers/ACFDE/output/${DATASET_NAME}/colbert}"
RAW_TARGET_DIR="${RAW_TARGET_DIR:-/data/ali/${DATASET_NAME}-colbert}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/ali/${DATASET_NAME}-gem-data}"
INDEX_ROOT="${INDEX_ROOT:-/data/ali/${DATASET_NAME}-gem-index}"
RESULTS_DIR="${RESULTS_DIR:-/home/ali/gem-baseline/results/${DATASET_NAME}_gem_index_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
DATASET_STEM="${DATASET_STEM:-${DATASET_NAME}}"

DOCS_PER_SHARD="${DOCS_PER_SHARD:-25000}"
FINE_K="${FINE_K:-32768}"
COARSE_K="${COARSE_K:-1024}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1000000}"
NITER="${NITER:-25}"
SEED="${SEED:-123}"
BATCH_SIZE="${BATCH_SIZE:-65536}"
TOP_R="${TOP_R:-3}"
USE_GPU="${USE_GPU:-1}"
STAGES="${STAGES:-inspect docdata queries fine-centroids codes coarse-centroids coarse-info build-index}"

M_INDEX="${M_INDEX:-24}"
EF_INDEX="${EF_INDEX:-80}"
BUILD_THREADS="${BUILD_THREADS:-80}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
CLUSTER_DISTANCE_BLOCK_ROWS="${CLUSTER_DISTANCE_BLOCK_ROWS:-1024}"
BASE_FP32="${BASE_FP32:-0}"
REPAIR_ON_BUILD="${REPAIR_ON_BUILD:-0}"
REPAIR_ON_LOAD="${REPAIR_ON_LOAD:-0}"
SKIP_SEARCH_AFTER_BUILD="${SKIP_SEARCH_AFTER_BUILD:-0}"
RERANK_LIST="${RERANK_LIST:-128,256,512}"
EF_LIST="${EF_LIST:-1000,4000,8000,16000}"
QRELS_FILE="${QRELS_FILE:-}"
QUERIES_FILE="${QUERIES_FILE:-}"
CORPUS_FILE="${CORPUS_FILE:-}"
QRELS_QUERY_ORDER="${QRELS_QUERY_ORDER:-queries-jsonl}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${INDEX_ROOT}"
PIPELINE_LOG="${LOG_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
CONFIG_FILE="${RESULTS_DIR}/run_config.txt"

{
  echo "DATASET_NAME=${DATASET_NAME}"
  echo "RAW_SOURCE_DIR=${RAW_SOURCE_DIR}"
  echo "RAW_TARGET_DIR=${RAW_TARGET_DIR}"
  echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
  echo "INDEX_ROOT=${INDEX_ROOT}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "DATASET_STEM=${DATASET_STEM}"
  echo "DOCS_PER_SHARD=${DOCS_PER_SHARD}"
  echo "FINE_K=${FINE_K}"
  echo "COARSE_K=${COARSE_K}"
  echo "SAMPLE_SIZE=${SAMPLE_SIZE}"
  echo "NITER=${NITER}"
  echo "SEED=${SEED}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "TOP_R=${TOP_R}"
  echo "USE_GPU=${USE_GPU}"
  echo "STAGES=${STAGES}"
  echo "M_INDEX=${M_INDEX}"
  echo "EF_INDEX=${EF_INDEX}"
  echo "BUILD_THREADS=${BUILD_THREADS}"
  echo "SEARCH_THREADS=${SEARCH_THREADS}"
  echo "CLUSTER_DISTANCE_BLOCK_ROWS=${CLUSTER_DISTANCE_BLOCK_ROWS}"
  echo "BASE_FP32=${BASE_FP32}"
  echo "REPAIR_ON_BUILD=${REPAIR_ON_BUILD}"
  echo "REPAIR_ON_LOAD=${REPAIR_ON_LOAD}"
  echo "SKIP_SEARCH_AFTER_BUILD=${SKIP_SEARCH_AFTER_BUILD}"
  echo "RERANK_LIST=${RERANK_LIST}"
  echo "EF_LIST=${EF_LIST}"
  echo "QRELS_FILE=${QRELS_FILE}"
  echo "QUERIES_FILE=${QUERIES_FILE}"
  echo "CORPUS_FILE=${CORPUS_FILE}"
  echo "QRELS_QUERY_ORDER=${QRELS_QUERY_ORDER}"
} > "${CONFIG_FILE}"

echo "Pipeline log: ${PIPELINE_LOG}"
echo "Run config: ${CONFIG_FILE}"

run_preprocess_stage() {
  local stage="$1"
  local gpu_arg=()
  if [[ "${USE_GPU}" == "1" ]]; then
    gpu_arg=(--use-gpu)
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_msmarco_gem_data.py" \
    --output-root "${OUTPUT_ROOT}" \
    --dataset-dir "${RAW_TARGET_DIR}" \
    --dataset-stem "${DATASET_STEM}" \
    --stage "${stage}" \
    --docs-per-shard "${DOCS_PER_SHARD}" \
    --fine-k "${FINE_K}" \
    --coarse-k "${COARSE_K}" \
    --sample-size "${SAMPLE_SIZE}" \
    --niter "${NITER}" \
    --seed "${SEED}" \
    --batch-size "${BATCH_SIZE}" \
    --top-r "${TOP_R}" \
    "${gpu_arg[@]}"
}

{
  date --iso-8601=seconds
  echo "Preparing raw NPY aliases"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_colbert_raw_files.py" \
    --raw-source "${RAW_SOURCE_DIR}" \
    --raw-target "${RAW_TARGET_DIR}" \
    --dataset-stem "${DATASET_STEM}"

  if [[ -n "${QRELS_FILE}" && -n "${QUERIES_FILE}" && -n "${CORPUS_FILE}" ]]; then
    echo "Preparing numeric qrels for internal GEM metrics"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/prepare_beir_qrels_for_gem.py" \
      --queries "${QUERIES_FILE}" \
      --corpus "${CORPUS_FILE}" \
      --qrels "${QRELS_FILE}" \
      --output "${OUTPUT_ROOT}/qdata/qrels.tsv" \
      --query-order "${QRELS_QUERY_ORDER}"
  else
    echo "No complete qrels/queries/corpus paths provided; internal GEM qrels will be skipped."
  fi

  for stage in ${STAGES}; do
    case "${stage}" in
      inspect|docdata|queries|fine-centroids|codes|coarse-centroids|coarse-info)
        echo "=== Running stage: ${stage} ==="
        run_preprocess_stage "${stage}"
        ;;
      build-index)
        echo "=== Running stage: build-index ==="
        env \
          REPO_ROOT="${REPO_ROOT}" \
          RESULTS_DIR="${RESULTS_DIR}" \
          LOG_DIR="${LOG_DIR}" \
          INDEX_ROOT="${INDEX_ROOT}" \
          GEM_DATASET="generic" \
          GEM_DATASET_NAME="${DATASET_NAME}" \
          GEM_DATASET_PATH="${OUTPUT_ROOT}" \
          GEM_NUM_CLUSTER="${FINE_K}" \
          GEM_NUM_GRAPH_CLUSTER="${COARSE_K}" \
          GEM_REBUILD="1" \
          GEM_SKIP_SEARCH="${SKIP_SEARCH_AFTER_BUILD}" \
          GEM_BUILD_THREADS="${BUILD_THREADS}" \
          GEM_SEARCH_THREADS="${SEARCH_THREADS}" \
          GEM_EF_INDEX="${EF_INDEX}" \
          GEM_CLUSTER_DISTANCE_BLOCK_ROWS="${CLUSTER_DISTANCE_BLOCK_ROWS}" \
          GEM_MSMARCO_BASE_FP32="${BASE_FP32}" \
          GEM_APPLY_REPAIR_ON_BUILD="${REPAIR_ON_BUILD}" \
          GEM_APPLY_REPAIR_ON_LOAD="${REPAIR_ON_LOAD}" \
          GEM_EF_LIST="${EF_LIST}" \
          GEM_RERANK_LIST="${RERANK_LIST}" \
          KEEP_SHELL_ON_EXIT="0" \
          bash "${REPO_ROOT}/scripts/run_msmarco_gem_logged.sh"
        ;;
      search)
        echo "=== Running stage: search ==="
        env \
          REPO_ROOT="${REPO_ROOT}" \
          RESULTS_DIR="${RESULTS_DIR}" \
          LOG_DIR="${LOG_DIR}" \
          INDEX_ROOT="${INDEX_ROOT}" \
          GEM_DATASET="generic" \
          GEM_DATASET_NAME="${DATASET_NAME}" \
          GEM_DATASET_PATH="${OUTPUT_ROOT}" \
          GEM_NUM_CLUSTER="${FINE_K}" \
          GEM_NUM_GRAPH_CLUSTER="${COARSE_K}" \
          GEM_REBUILD="0" \
          GEM_SKIP_SEARCH="0" \
          GEM_BUILD_THREADS="${BUILD_THREADS}" \
          GEM_SEARCH_THREADS="${SEARCH_THREADS}" \
          GEM_EF_INDEX="${EF_INDEX}" \
          GEM_CLUSTER_DISTANCE_BLOCK_ROWS="${CLUSTER_DISTANCE_BLOCK_ROWS}" \
          GEM_MSMARCO_BASE_FP32="${BASE_FP32}" \
          GEM_APPLY_REPAIR_ON_BUILD="${REPAIR_ON_BUILD}" \
          GEM_APPLY_REPAIR_ON_LOAD="${REPAIR_ON_LOAD}" \
          GEM_EF_LIST="${EF_LIST}" \
          GEM_RERANK_LIST="${RERANK_LIST}" \
          KEEP_SHELL_ON_EXIT="0" \
          bash "${REPO_ROOT}/scripts/run_msmarco_gem_logged.sh"
        ;;
      *)
        echo "Unknown stage: ${stage}" >&2
        exit 2
        ;;
    esac
  done
  date --iso-8601=seconds
  echo "Pipeline finished"
} 2>&1 | tee -a "${PIPELINE_LOG}"
