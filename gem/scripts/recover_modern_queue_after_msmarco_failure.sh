#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
PYTHON_BIN="${PYTHON_BIN:-/data/ali/gem-baseline/bin/python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
CUDA_DEVICE="${CUDA_DEVICE:-3}"
USE_GPU="${USE_GPU:-1}"
DOCS_PER_SHARD="${DOCS_PER_SHARD:-25000}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
BUILD_THREADS="${BUILD_THREADS:-90}"
SEARCH_THREADS="${SEARCH_THREADS:-1}"
M_INDEX="${M_INDEX:-24}"
EF_INDEX="${EF_INDEX:-80}"
POLL_SECS="${POLL_SECS:-30}"
QUEUE_ROOT="${QUEUE_ROOT:-/tmp/gem_modern_colbert_recovery_${RUN_ID}}"

mkdir -p "${QUEUE_ROOT}/started" "${QUEUE_ROOT}/done" "${QUEUE_ROOT}/pending"

kill_if_exists() {
  local session="$1"
  if tmux has-session -t "${session}" 2>/dev/null; then
    tmux kill-session -t "${session}"
  fi
}

launch_session() {
  local order="$1"
  local session="$2"
  local dataset_name="$3"
  local dataset_stem="$4"
  local raw_source="$5"
  local raw_target="$6"
  local output_root="$7"
  local index_root="$8"
  local results_dir="$9"
  local fine_k="${10}"
  local coarse_k="${11}"
  local sample_size="${12}"
  local top_r="${13}"
  local stages="${14}"
  local prev_started="${15}"

  local started_marker="${QUEUE_ROOT}/started/$(printf '%02d' "${order}")_${dataset_name}"
  local done_marker="${QUEUE_ROOT}/done/$(printf '%02d' "${order}")_${dataset_name}"

  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && \
      echo 'Queued ${dataset_name}: fine_k=${fine_k} coarse_k=${coarse_k} top_r=${top_r} stages=${stages}' && \
      env REPO_ROOT='${REPO_ROOT}' PYTHON_BIN='${PYTHON_BIN}' CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' USE_GPU='${USE_GPU}' \
        DATASET_NAME='${dataset_name}' DATASET_STEM='${dataset_stem}' RAW_SOURCE_DIR='${raw_source}' RAW_TARGET_DIR='${raw_target}' \
        OUTPUT_ROOT='${output_root}' INDEX_ROOT='${index_root}' RESULTS_DIR='${results_dir}' DOCS_PER_SHARD='${DOCS_PER_SHARD}' \
        FINE_K='${fine_k}' COARSE_K='${coarse_k}' SAMPLE_SIZE='${sample_size}' TOP_R='${top_r}' STAGES='${stages}' \
        M_INDEX='${M_INDEX}' EF_INDEX='${EF_INDEX}' BUILD_THREADS='${BUILD_THREADS}' SEARCH_THREADS='${SEARCH_THREADS}' \
        SKIP_SEARCH_AFTER_BUILD='1' MAX_PARALLEL='${MAX_PARALLEL}' POLL_SECS='${POLL_SECS}' QUEUE_ROOT='${QUEUE_ROOT}' \
        SESSION_NAME='${session}' STARTED_MARKER='${started_marker}' DONE_MARKER='${done_marker}' \
        PREV_STARTED_MARKER='${prev_started}' bash '${REPO_ROOT}/scripts/run_ordered_gem_queue_session.sh'; \
      code=\$?; echo '${session} EXIT_CODE='\"\$code\"; exec bash"
}

kill_if_exists "gem-msmarco-modern-colbert"
kill_if_exists "gem-clerc-modern-colbert"
kill_if_exists "gem-clef-modern-colbert"
kill_if_exists "gem-fiqa-modern-colbert"
kill_if_exists "gem-scidocs-modern-colbert"

launch_session \
  0 \
  "gem-msmarco-modern-colbert" \
  "msmarco-modern-colbert" \
  "msmarco_modern_colbert" \
  "/data/ali/msmarco-modern-colbert" \
  "/data/ali/msmarco-modern-gem-raw" \
  "/data/ali/msmarco-modern-gem-data" \
  "/data/ali/msmarco-modern-gem-index" \
  "/home/ali/gem-baseline/results/msmarco_modern_colbert_gem_index_${RUN_ID}_rebuild" \
  "262144" \
  "40960" \
  "500000" \
  "10" \
  "build-index" \
  ""

launch_session \
  1 \
  "gem-clerc-modern-colbert" \
  "clerc-modern-colbert" \
  "clerc_modern_colbert" \
  "/data/ali/clerc-modern-colbert" \
  "/data/ali/clerc-modern-gem-raw" \
  "/data/ali/clerc-modern-gem-data" \
  "/data/ali/clerc-modern-gem-index" \
  "/home/ali/gem-baseline/results/clerc_modern_colbert_gem_index_${RUN_ID}" \
  "65536" \
  "8192" \
  "2000000" \
  "5" \
  "inspect docdata queries fine-centroids codes coarse-centroids coarse-info build-index" \
  "${QUEUE_ROOT}/started/00_msmarco-modern-colbert"

launch_session \
  2 \
  "gem-clef-modern-colbert" \
  "clef-modern-colbert" \
  "clef_modern_colbert" \
  "/data/ali/clef-modern-colbert" \
  "/data/ali/clef-modern-gem-raw" \
  "/data/ali/clef-modern-gem-data" \
  "/data/ali/clef-modern-gem-index" \
  "/home/ali/gem-baseline/results/clef_modern_colbert_gem_index_${RUN_ID}" \
  "65536" \
  "8192" \
  "2000000" \
  "5" \
  "inspect docdata queries fine-centroids codes coarse-centroids coarse-info build-index" \
  "${QUEUE_ROOT}/started/01_clerc-modern-colbert"

launch_session \
  3 \
  "gem-fiqa-modern-colbert" \
  "fiqa-modern-colbert" \
  "fiqa_modern_colbert" \
  "/data/ali/fiqa-modern-colbert" \
  "/data/ali/fiqa-modern-gem-raw" \
  "/data/ali/fiqa-modern-gem-data" \
  "/data/ali/fiqa-modern-gem-index" \
  "/home/ali/gem-baseline/results/fiqa_modern_colbert_gem_index_${RUN_ID}" \
  "32768" \
  "1024" \
  "2000000" \
  "10" \
  "inspect docdata queries fine-centroids codes coarse-centroids coarse-info build-index" \
  "${QUEUE_ROOT}/started/02_clef-modern-colbert"

launch_session \
  4 \
  "gem-scidocs-modern-colbert" \
  "scidocs-modern-colbert" \
  "scidocs_modern_colbert" \
  "/data/ali/scidocs-modern-colbert" \
  "/data/ali/scidocs-modern-gem-raw" \
  "/data/ali/scidocs-modern-gem-data" \
  "/data/ali/scidocs-modern-gem-index" \
  "/home/ali/gem-baseline/results/scidocs_modern_colbert_gem_index_${RUN_ID}" \
  "4096" \
  "512" \
  "1000000" \
  "10" \
  "inspect docdata queries fine-centroids codes coarse-centroids coarse-info build-index" \
  "${QUEUE_ROOT}/started/03_fiqa-modern-colbert"

echo "Recovery queue root: ${QUEUE_ROOT}"
