#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
RESULTS_DIR="${RESULTS_DIR:-/home/ali/gem-baseline/results/msmarco_gem_20260421_fp32_loadindex}"
INDEX_ROOT="${INDEX_ROOT:-/data1/ali/msmarco-gem-data/example_index}"
LOGGED_RUNNER="${LOGGED_RUNNER:-${REPO_ROOT}/scripts/run_msmarco_gem_logged.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${REPO_ROOT}/scripts/evaluate_scidocs_gem_results.py}"
MSMARCO_EVAL_DIR="${MSMARCO_EVAL_DIR:-${REPO_ROOT}/msmarco_evaluation}"

mkdir -p "${RESULTS_DIR}"

export KEEP_SHELL_ON_EXIT=0
export GEM_REBUILD="${GEM_REBUILD:-0}"
export GEM_MSMARCO_BASE_FP32="${GEM_MSMARCO_BASE_FP32:-1}"
export GEM_SEARCH_THREADS="${GEM_SEARCH_THREADS:-1}"
export GEM_BUILD_THREADS="${GEM_BUILD_THREADS:-90}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-90}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-90}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export GEM_RESULTS_DIR="${GEM_RESULTS_DIR:-${RESULTS_DIR}}"
export INDEX_ROOT
export GEM_INDEX_ROOT="${GEM_INDEX_ROOT:-${INDEX_ROOT}}"

printf 'Starting MSMARCO search-only fp32 rerank pass\n'
printf 'RESULTS_DIR=%s\n' "${RESULTS_DIR}"
printf 'GEM_REBUILD=%s\n' "${GEM_REBUILD}"
printf 'GEM_MSMARCO_BASE_FP32=%s\n' "${GEM_MSMARCO_BASE_FP32}"
printf 'GEM_SEARCH_THREADS=%s\n' "${GEM_SEARCH_THREADS}"

bash "${LOGGED_RUNNER}"

QUERIES_DEV="${RESULTS_DIR}/queries_dev_6980.jsonl"
tail -n 6980 "${MSMARCO_EVAL_DIR}/queries_full.jsonl" > "${QUERIES_DEV}"

python "${EVAL_SCRIPT}" \
  --queries "${QUERIES_DEV}" \
  --corpus "${MSMARCO_EVAL_DIR}/corpus.jsonl" \
  --qrels "${MSMARCO_EVAL_DIR}/qrels/dev.tsv" \
  --runs-glob "${RESULTS_DIR}/*.tsv" \
  --output-csv "${RESULTS_DIR}/msmarco_dev6980_eval.csv" \
  --log-file "$(ls -t "${RESULTS_DIR}/logs"/gem_run_*.log | head -n 1)" \
  > "${RESULTS_DIR}/msmarco_dev6980_eval.txt" 2>&1

printf 'Finished benchmark and evaluation\n'
