#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-/data/ali/msmarco-gem-data}"
VARIANT_OUTPUT_ROOT="${VARIANT_OUTPUT_ROOT:-/data1/ali/msmarco-gem-paper-faithful}"
RESULTS_DIR="${RESULTS_DIR:-/home/ali/gem-baseline/results/msmarco_gem_20260422_paper_faithful}"
LOG_DIR="${LOG_DIR:-${RESULTS_DIR}/logs}"
INDEX_ROOT="${INDEX_ROOT:-/data1/ali/msmarco-gem-data/example_index_paper}"
TRAIN_QRELS="${TRAIN_QRELS:-/home/ali/gem-baseline/msmarco_evaluation/qrels/train.tsv}"
TRAIN_QUERY_EMBS_NPY="${TRAIN_QUERY_EMBS_NPY:-}"

FINE_K="${FINE_K:-262144}"
COARSE_K="${COARSE_K:-40960}"
R_MAX="${R_MAX:-10}"
QUERY_TOP_T="${QUERY_TOP_T:-4}"
SHORTCUT_SAMPLE_RATIO="${SHORTCUT_SAMPLE_RATIO:-0.2}"
SHORTCUT_SEED="${SHORTCUT_SEED:-123}"
SHORTCUT_TOPF="${SHORTCUT_TOPF:-100}"
SHORTCUT_EF_SEARCH="${SHORTCUT_EF_SEARCH:-${SHORTCUT_TOPF}}"
BUILD_THREADS="${BUILD_THREADS:-90}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
EF_INDEX="${EF_INDEX:-80}"
CLUSTER_ENTRY_SEED="${CLUSTER_ENTRY_SEED:-123}"

PREP_LOG="${LOG_DIR}/prepare_paper_dataset.log"
CONFIG_FILE="${RESULTS_DIR}/run_config.txt"
SAMPLED_SHORTCUT_QEMBS="${VARIANT_OUTPUT_ROOT}/aux/paper_shortcut_train_qembs_sample${SHORTCUT_SAMPLE_RATIO}_seed${SHORTCUT_SEED}.npy"
SAMPLED_SHORTCUT_QRELS="${VARIANT_OUTPUT_ROOT}/aux/paper_shortcut_train_qrels_sample${SHORTCUT_SAMPLE_RATIO}_seed${SHORTCUT_SEED}.tsv"
SAMPLED_SHORTCUT_META="${VARIANT_OUTPUT_ROOT}/aux/paper_shortcut_train_pairs_sample${SHORTCUT_SAMPLE_RATIO}_seed${SHORTCUT_SEED}.json"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${VARIANT_OUTPUT_ROOT}/cdata" "${VARIANT_OUTPUT_ROOT}/aux" "${INDEX_ROOT}"

if [[ -z "${TRAIN_QUERY_EMBS_NPY}" ]]; then
  echo "TRAIN_QUERY_EMBS_NPY is required for paper-faithful adaptive cutoff and shortcut injection." | tee "${PREP_LOG}"
  exit 1
fi
if [[ ! -f "${TRAIN_QUERY_EMBS_NPY}" ]]; then
  echo "Missing TRAIN_QUERY_EMBS_NPY: ${TRAIN_QUERY_EMBS_NPY}" | tee "${PREP_LOG}"
  exit 1
fi

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
  echo "=== Paper-Faithful MSMARCO Build Prep ==="
  date --iso-8601=seconds
  echo "REPO_ROOT=${REPO_ROOT}"
  echo "BASE_OUTPUT_ROOT=${BASE_OUTPUT_ROOT}"
  echo "VARIANT_OUTPUT_ROOT=${VARIANT_OUTPUT_ROOT}"
  echo "RESULTS_DIR=${RESULTS_DIR}"
  echo "INDEX_ROOT=${INDEX_ROOT}"
  echo "TRAIN_QRELS=${TRAIN_QRELS}"
  echo "TRAIN_QUERY_EMBS_NPY=${TRAIN_QUERY_EMBS_NPY}"
  echo "FINE_K=${FINE_K}"
  echo "COARSE_K=${COARSE_K}"
  echo "R_MAX=${R_MAX}"
  echo "QUERY_TOP_T=${QUERY_TOP_T}"
  echo "SHORTCUT_SAMPLE_RATIO=${SHORTCUT_SAMPLE_RATIO}"
  echo "SHORTCUT_SEED=${SHORTCUT_SEED}"
  echo "SHORTCUT_TOPF=${SHORTCUT_TOPF}"
  echo "SHORTCUT_EF_SEARCH=${SHORTCUT_EF_SEARCH}"
  echo "BUILD_THREADS=${BUILD_THREADS}"
  echo "SEARCH_THREADS=${SEARCH_THREADS}"
  echo "EF_INDEX=${EF_INDEX}"
  echo "CLUSTER_ENTRY_SEED=${CLUSTER_ENTRY_SEED}"
  echo "repair_on_build=0"
  echo "repair_on_load=0"
  echo "======================================="
} | tee "${PREP_LOG}"

"${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/build_msmarco_paper_coarse_info.py" \
  --output-root "${VARIANT_OUTPUT_ROOT}" \
  --mode adaptive \
  --r-max "${R_MAX}" \
  --query-top-t "${QUERY_TOP_T}" \
  --train-query-embs "${TRAIN_QUERY_EMBS_NPY}" \
  --train-qrels "${TRAIN_QRELS}" 2>&1 | tee -a "${PREP_LOG}"

"${PYTHON_BIN}" -u "${REPO_ROOT}/scripts/sample_msmarco_paper_training_pairs.py" \
  --train-query-embs "${TRAIN_QUERY_EMBS_NPY}" \
  --train-qrels "${TRAIN_QRELS}" \
  --sample-ratio "${SHORTCUT_SAMPLE_RATIO}" \
  --seed "${SHORTCUT_SEED}" \
  --output-query-embs "${SAMPLED_SHORTCUT_QEMBS}" \
  --output-qrels "${SAMPLED_SHORTCUT_QRELS}" \
  --metadata-out "${SAMPLED_SHORTCUT_META}" 2>&1 | tee -a "${PREP_LOG}"

{
  echo "dataset_root=${VARIANT_OUTPUT_ROOT}/"
  echo "index_root=${INDEX_ROOT}"
  echo "fine_k=${FINE_K}"
  echo "coarse_k=${COARSE_K}"
  echo "graph_M=24"
  echo "ef_construction=${EF_INDEX}"
  echo "query_cluster_filter_t=${QUERY_TOP_T}"
  echo "adaptive_cluster_cutoff_enabled=1"
  echo "adaptive_cluster_cutoff_r_max=${R_MAX}"
  echo "fixed_top_r_disabled_by_default=1"
  echo "shortcut_mode=paper"
  echo "shortcut_sample_ratio=${SHORTCUT_SAMPLE_RATIO}"
  echo "shortcut_seed=${SHORTCUT_SEED}"
  echo "shortcut_topf=${SHORTCUT_TOPF}"
  echo "shortcut_ef_search=${SHORTCUT_EF_SEARCH}"
  echo "repair_on_build=0"
  echo "repair_on_load=0"
  echo "cluster_entry_seed=${CLUSTER_ENTRY_SEED}"
  echo "train_query_embs_npy=${TRAIN_QUERY_EMBS_NPY}"
  echo "sampled_shortcut_qembs=${SAMPLED_SHORTCUT_QEMBS}"
  echo "sampled_shortcut_qrels=${SAMPLED_SHORTCUT_QRELS}"
  echo "assumption_shortcut_topf=paper_does_not_explicitly_specify_fprime_in_text_used_here"
} > "${CONFIG_FILE}"

env \
  RESULTS_DIR="${RESULTS_DIR}" \
  INDEX_ROOT="${INDEX_ROOT}" \
  GEM_REBUILD=1 \
  GEM_BUILD_THREADS="${BUILD_THREADS}" \
  GEM_SEARCH_THREADS="${SEARCH_THREADS}" \
  GEM_EF_INDEX="${EF_INDEX}" \
  GEM_CLUSTER_ENTRY_SEED="${CLUSTER_ENTRY_SEED}" \
  GEM_MSMARCO_DATASET_PATH="${VARIANT_OUTPUT_ROOT}/" \
  GEM_APPLY_REPAIR_ON_BUILD=0 \
  GEM_APPLY_REPAIR_ON_LOAD=0 \
  GEM_SHORTCUT_MODE=paper \
  GEM_PAPER_SHORTCUT_TRAIN_QEMBS="${SAMPLED_SHORTCUT_QEMBS}" \
  GEM_PAPER_SHORTCUT_TRAIN_QRELS="${SAMPLED_SHORTCUT_QRELS}" \
  GEM_SHORTCUT_TOPF="${SHORTCUT_TOPF}" \
  GEM_PAPER_SHORTCUT_EF_SEARCH="${SHORTCUT_EF_SEARCH}" \
  OMP_NUM_THREADS="${BUILD_THREADS}" \
  OPENBLAS_NUM_THREADS="${BUILD_THREADS}" \
  bash "${REPO_ROOT}/scripts/run_msmarco_gem_logged.sh"
