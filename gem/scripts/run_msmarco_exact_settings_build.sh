#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
RAW_TARGET_DIR="${RAW_TARGET_DIR:-/data/ali/msmarco-colbert}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/data/ali/msmarco-gem-data}"
VARIANT_OUTPUT_ROOT="${VARIANT_OUTPUT_ROOT:-/data1/ali/msmarco-gem-data-rmax10}"
RESULTS_DIR="${RESULTS_DIR:-/home/ali/gem-baseline/results/msmarco_gem_20260422_exact_24_80_r10}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
INDEX_ROOT="${INDEX_ROOT:-/data1/ali/msmarco-gem-data/example_index}"
TRAIN_QRELS="${TRAIN_QRELS:-/home/ali/gem-baseline/msmarco_evaluation/qrels/train.tsv}"

DOCS_PER_SHARD="${DOCS_PER_SHARD:-25000}"
FINE_K="${FINE_K:-262144}"
COARSE_K="${COARSE_K:-40960}"
TOP_R="${TOP_R:-10}"
SHORTCUT_SAMPLE_RATIO="${SHORTCUT_SAMPLE_RATIO:-0.2}"
SHORTCUT_SEED="${SHORTCUT_SEED:-123}"
BUILD_THREADS="${BUILD_THREADS:-90}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
EF_INDEX="${EF_INDEX:-80}"

DATASET_STEM="${DATASET_STEM:-msmarco-large}"
EDGE_FILE="${EDGE_FILE:-${VARIANT_OUTPUT_ROOT}/aux/msmarco_shortcut_edges_sample20_seed${SHORTCUT_SEED}.tsv}"
PREP_LOG="${LOG_DIR}/prepare_exact_dataset.log"
CONFIG_FILE="${RESULTS_DIR}/run_config.txt"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${VARIANT_OUTPUT_ROOT}/cdata" "${VARIANT_OUTPUT_ROOT}/aux"

if [[ ! -e "${VARIANT_OUTPUT_ROOT}/docdata" ]]; then
  ln -s "${BASE_OUTPUT_ROOT}/docdata" "${VARIANT_OUTPUT_ROOT}/docdata"
fi
if [[ ! -e "${VARIANT_OUTPUT_ROOT}/qdata" ]]; then
  ln -s "${BASE_OUTPUT_ROOT}/qdata" "${VARIANT_OUTPUT_ROOT}/qdata"
fi
if [[ ! -e "${VARIANT_OUTPUT_ROOT}/cdata/centroids.npy" ]]; then
  ln -s "${BASE_OUTPUT_ROOT}/cdata/centroids.npy" "${VARIANT_OUTPUT_ROOT}/cdata/centroids.npy"
fi
if [[ ! -e "${VARIANT_OUTPUT_ROOT}/cdata/coarse_centroids.npy" ]]; then
  ln -s "${BASE_OUTPUT_ROOT}/cdata/coarse_centroids.npy" "${VARIANT_OUTPUT_ROOT}/cdata/coarse_centroids.npy"
fi

{
  echo "=== Exact MSMARCO Build Prep (non-paper baseline) ==="
  date --iso-8601=seconds
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "RAW_TARGET_DIR=${RAW_TARGET_DIR}"
  echo "BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT}"
  echo "VARIANT_OUTPUT_ROOT=${VARIANT_OUTPUT_ROOT}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "INDEX_ROOT=${INDEX_ROOT}"
  echo "TRAIN_QRELS=${TRAIN_QRELS}"
  echo "DOCS_PER_SHARD=${DOCS_PER_SHARD}"
  echo "FINE_K=${FINE_K}"
  echo "COARSE_K=${COARSE_K}"
  echo "TOP_R=${TOP_R}"
  echo "SHORTCUT_SAMPLE_RATIO=${SHORTCUT_SAMPLE_RATIO}"
  echo "SHORTCUT_SEED=${SHORTCUT_SEED}"
  echo "BUILD_THREADS=${BUILD_THREADS}"
  echo "SEARCH_THREADS=${SEARCH_THREADS}"
  echo "EF_INDEX=${EF_INDEX}"
  echo "EDGE_FILE=${EDGE_FILE}"
  echo "==============================="
  echo "NOTE=This script uses fixed top-r coarse assignment and legacy offline shortcut edges. It is not paper-faithful."
} | tee "${PREP_LOG}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_msmarco_gem_data.py" \
  --output-root "${VARIANT_OUTPUT_ROOT}" \
  --dataset-dir "${RAW_TARGET_DIR}" \
  --dataset-stem "${DATASET_STEM}" \
  --stage coarse-info \
  --docs-per-shard "${DOCS_PER_SHARD}" \
  --fine-k "${FINE_K}" \
  --coarse-k "${COARSE_K}" \
  --top-r "${TOP_R}" 2>&1 | tee -a "${PREP_LOG}"

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/generate_msmarco_shortcut_edges.py" \
  --qrels "${TRAIN_QRELS}" \
  --output "${EDGE_FILE}" \
  --sample-ratio "${SHORTCUT_SAMPLE_RATIO}" \
  --seed "${SHORTCUT_SEED}" 2>&1 | tee -a "${PREP_LOG}"

{
  echo "dataset_root=${VARIANT_OUTPUT_ROOT}/"
  echo "index_root=${INDEX_ROOT}"
  echo "fine_k=${FINE_K}"
  echo "coarse_k=${COARSE_K}"
  echo "graph_M=24"
  echo "ef_construction=${EF_INDEX}"
  echo "query_cluster_filter_t=4"
  echo "adaptive_cluster_cutoff_r_max=${TOP_R}"
  echo "shortcut_edge_sample_ratio=${SHORTCUT_SAMPLE_RATIO}"
  echo "shortcut_edge_seed=${SHORTCUT_SEED}"
  echo "shortcut_edge_file=${EDGE_FILE}"
  echo "build_threads=${BUILD_THREADS}"
  echo "search_threads=${SEARCH_THREADS}"
} > "${CONFIG_FILE}"

env \
  RESULTS_DIR="${RESULTS_DIR}" \
  INDEX_ROOT="${INDEX_ROOT}" \
  GEM_REBUILD=1 \
  GEM_BUILD_THREADS="${BUILD_THREADS}" \
  GEM_SEARCH_THREADS="${SEARCH_THREADS}" \
  GEM_EF_INDEX="${EF_INDEX}" \
  GEM_MSMARCO_DATASET_PATH="${VARIANT_OUTPUT_ROOT}/" \
  GEM_APPLY_REPAIR_ON_BUILD=1 \
  GEM_SHORTCUT_MODE=legacy \
  GEM_SHORTCUT_EDGE_FILE="${EDGE_FILE}" \
  OMP_NUM_THREADS="${BUILD_THREADS}" \
  OPENBLAS_NUM_THREADS="${BUILD_THREADS}" \
  bash "${REPO_ROOT}/scripts/run_msmarco_gem_logged.sh"
