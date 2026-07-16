#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
WAIT_FOR_FILE="${WAIT_FOR_FILE:-}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${WAIT_FOR_FILE}" ]]; then
  echo "Waiting for ${WAIT_FOR_FILE}"
  while [[ ! -f "${WAIT_FOR_FILE}" ]]; do
    sleep 60
  done
fi

run_gem_search() {
  local name="$1"
  shift
  echo "=== t-sweep search: ${name} ==="
  env \
    REPO_ROOT="${REPO_ROOT}" \
    GEM_REBUILD="0" \
    GEM_SKIP_SEARCH="0" \
    GEM_EF_INDEX="80" \
    GEM_SEARCH_THREADS="1" \
    GEM_BUILD_THREADS="80" \
    GEM_MSMARCO_BASE_FP32="0" \
    GEM_APPLY_REPAIR_ON_LOAD="0" \
    GEM_NPROB_LIST="4,8,16" \
    GEM_RERANK_LIST="${GEM_RERANK_LIST:-1024}" \
    GEM_EF_LIST="${GEM_EF_LIST:-24000}" \
    KEEP_SHELL_ON_EXIT="0" \
    "$@" \
    bash "${REPO_ROOT}/scripts/run_msmarco_gem_logged.sh"
}

evaluate_run() {
  local results_dir="$1"
  local queries="$2"
  local corpus="$3"
  local qrels="$4"
  local output_prefix="$5"
  local query_order="${6:-queries-jsonl}"
  echo "=== evaluate: ${output_prefix} ==="
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_scidocs_gem_results.py" \
    --queries "${queries}" \
    --corpus "${corpus}" \
    --qrels "${qrels}" \
    --runs-glob "${results_dir}/*_t*_rerank*_ef*.tsv" \
    --k-values 10 100 \
    --query-order "${query_order}" \
    --output-csv "${results_dir}/${output_prefix}_eval.csv" \
    > "${results_dir}/${output_prefix}_eval.txt"
}

MSMARCO_RESULTS="${REPO_ROOT}/results/msmarco_gem_${RUN_ID}_t_sweep_serial"
mkdir -p "${MSMARCO_RESULTS}"
run_gem_search "msmarco" \
  RESULTS_DIR="${MSMARCO_RESULTS}" \
  LOG_DIR="${MSMARCO_RESULTS}/logs" \
  INDEX_ROOT="/data1/ali/msmarco-gem-data/example_index" \
  GEM_DATASET="msmarco" \
  GEM_MSMARCO_DATASET_PATH="/data/ali/msmarco-gem-data/" \
  GEM_RERANK_LIST="2048"
evaluate_run \
  "${MSMARCO_RESULTS}" \
  "${REPO_ROOT}/msmarco_evaluation/queries_full.jsonl" \
  "${REPO_ROOT}/msmarco_evaluation/corpus.jsonl" \
  "${REPO_ROOT}/msmarco_evaluation/qrels/dev.tsv" \
  "msmarco_t_sweep"

SCIDOCS_RESULTS="${REPO_ROOT}/results/scidocs_gem_${RUN_ID}_t_sweep_serial"
mkdir -p "${SCIDOCS_RESULTS}"
run_gem_search "scidocs" \
  RESULTS_DIR="${SCIDOCS_RESULTS}" \
  LOG_DIR="${SCIDOCS_RESULTS}/logs" \
  INDEX_ROOT="/data/ali/scidocs-gem-index" \
  GEM_DATASET="generic" \
  GEM_DATASET_NAME="scidocs" \
  GEM_DATASET_PATH="/data/ali/scidocs-gem-data" \
  GEM_NUM_CLUSTER="4096" \
  GEM_NUM_GRAPH_CLUSTER="512"
evaluate_run \
  "${SCIDOCS_RESULTS}" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/queries.jsonl" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/corpus.jsonl" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv" \
  "scidocs_t_sweep"

FIQA_RESULTS="${REPO_ROOT}/results/fiqa_gem_${RUN_ID}_t_sweep_serial"
mkdir -p "${FIQA_RESULTS}"
run_gem_search "fiqa" \
  RESULTS_DIR="${FIQA_RESULTS}" \
  LOG_DIR="${FIQA_RESULTS}/logs" \
  INDEX_ROOT="/data/ali/fiqa-gem-index" \
  GEM_DATASET="generic" \
  GEM_DATASET_NAME="fiqa" \
  GEM_DATASET_PATH="/data/ali/fiqa-gem-data" \
  GEM_NUM_CLUSTER="32768" \
  GEM_NUM_GRAPH_CLUSTER="1024"
evaluate_run \
  "${FIQA_RESULTS}" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/queries.jsonl" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/corpus.jsonl" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/qrels/test.tsv" \
  "fiqa_t_sweep" \
  "qrels-first-seen"

NQ_RESULTS="${REPO_ROOT}/results/nq_gem_${RUN_ID}_t_sweep_serial"
mkdir -p "${NQ_RESULTS}"
run_gem_search "nq" \
  RESULTS_DIR="${NQ_RESULTS}" \
  LOG_DIR="${NQ_RESULTS}/logs" \
  INDEX_ROOT="/data/ali/nq-gem-index" \
  GEM_DATASET="generic" \
  GEM_DATASET_NAME="nq" \
  GEM_DATASET_PATH="/data/ali/nq-gem-data" \
  GEM_NUM_CLUSTER="262144" \
  GEM_NUM_GRAPH_CLUSTER="40960"

echo "Serial t-sweeps finished."
