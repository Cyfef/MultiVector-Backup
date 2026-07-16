#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
USE_GPU="${USE_GPU:-1}"
LOCK_FILE="${LOCK_FILE:-/tmp/gem_colbert_dataset_pipeline.lock}"
DOCS_PER_SHARD="${DOCS_PER_SHARD:-25000}"
TOP_R="${TOP_R:-10}"
EF_INDEX="${EF_INDEX:-80}"
BUILD_THREADS="${BUILD_THREADS:-80}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"

launch_dataset() {
  local name="$1"
  local fine_k="$2"
  local coarse_k="$3"
  local sample_size="$4"
  local rerank_list="$5"
  local ef_list="$6"
  local qrels_file="$7"
  local queries_file="$8"
  local corpus_file="$9"
  local qrels_query_order="${10}"

  local session="gem-${name}"
  local raw_source="/data1/liuyaoyang/Papers/ACFDE/output/${name}/colbert"
  local raw_target="/data/ali/${name}-colbert"
  local output_root="/data/ali/${name}-gem-data"
  local index_root="/data/ali/${name}-gem-index"
  local results_dir="/home/ali/gem-baseline/results/${name}_gem_index_${RUN_ID}"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists, leaving it unchanged: ${session}"
    return
  fi

  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && echo 'Waiting for ${LOCK_FILE} before starting ${name}' && flock '${LOCK_FILE}' env CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' REPO_ROOT='${REPO_ROOT}' PYTHON_BIN='${PYTHON_BIN}' DATASET_NAME='${name}' RAW_SOURCE_DIR='${raw_source}' RAW_TARGET_DIR='${raw_target}' OUTPUT_ROOT='${output_root}' INDEX_ROOT='${index_root}' RESULTS_DIR='${results_dir}' DOCS_PER_SHARD='${DOCS_PER_SHARD}' FINE_K='${fine_k}' COARSE_K='${coarse_k}' SAMPLE_SIZE='${sample_size}' TOP_R='${TOP_R}' USE_GPU='${USE_GPU}' EF_INDEX='${EF_INDEX}' BUILD_THREADS='${BUILD_THREADS}' SEARCH_THREADS='${SEARCH_THREADS}' RERANK_LIST='${rerank_list}' EF_LIST='${ef_list}' QRELS_FILE='${qrels_file}' QUERIES_FILE='${queries_file}' CORPUS_FILE='${corpus_file}' QRELS_QUERY_ORDER='${qrels_query_order}' bash '${REPO_ROOT}/scripts/run_colbert_gem_index_pipeline.sh'; exec bash"

  echo "started ${session}"
  echo "  results=${results_dir}"
  echo "  index=${index_root}"
}

launch_dataset \
  "scidocs" \
  "4096" \
  "512" \
  "1000000" \
  "512,1024" \
  "4000,8000,16000,24000" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/queries.jsonl" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/corpus.jsonl" \
  "queries-jsonl"

launch_dataset \
  "fiqa" \
  "32768" \
  "1024" \
  "2000000" \
  "512,1024" \
  "4000,8000,16000,24000" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/qrels/test.tsv" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/queries.jsonl" \
  "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/corpus.jsonl" \
  "qrels-first-seen"

launch_dataset \
  "nq" \
  "131072" \
  "20480" \
  "2000000" \
  "512,1024" \
  "4000,8000,16000,24000" \
  "" \
  "" \
  "" \
  "queries-jsonl"
