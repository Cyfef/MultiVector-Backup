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
QUEUE_ROOT="${QUEUE_ROOT:-/tmp/gem_modern_colbert_index_queue_${RUN_ID}}"
STAGES="${STAGES:-inspect docdata queries fine-centroids codes coarse-centroids coarse-info build-index}"
POLL_SECS="${POLL_SECS:-30}"

mkdir -p "${QUEUE_ROOT}/started" "${QUEUE_ROOT}/done" "${QUEUE_ROOT}/pending"

declare -a ORDERED_DATASETS=(
  "msmarco-modern-colbert|262144|40960|500000|10|8841824|626923386"
  "nq-modern-colbert|131072|20480|2000000|5|2681469|266464562"
  "clerc-modern-colbert|65536|8192|2000000|5|530427|138439536"
  "clef-modern-colbert|65536|8192|2000000|5|435837|115608962"
  "fiqa-modern-colbert|32768|1024|2000000|10|57639|7695260"
  "scidocs-modern-colbert|4096|512|1000000|10|25658|4819277"
)

sanitize_name() {
  local name="$1"
  echo "${name//-/_}"
}

launch_dataset() {
  local order="$1"
  local name="$2"
  local fine_k="$3"
  local coarse_k="$4"
  local sample_size="$5"
  local top_r="$6"
  local docs="$7"
  local vectors="$8"

  local session="gem-${name}"
  local stem
  stem="$(sanitize_name "${name}")"
  local raw_source="/data/ali/${name}"
  local raw_target="/data/ali/${name%-colbert}-gem-raw"
  local output_root="/data/ali/${name%-colbert}-gem-data"
  local index_root="/data/ali/${name%-colbert}-gem-index"
  local results_dir="/home/ali/gem-baseline/results/${stem}_gem_index_${RUN_ID}"
  local started_marker="${QUEUE_ROOT}/started/$(printf '%02d' "${order}")_${name}"
  local done_marker="${QUEUE_ROOT}/done/$(printf '%02d' "${order}")_${name}"
  local prev_started_marker=""

  if (( order > 0 )); then
    local prev_name
    prev_name="$(echo "${ORDERED_DATASETS[$((order - 1))]}" | cut -d'|' -f1)"
    prev_started_marker="${QUEUE_ROOT}/started/$(printf '%02d' "$((order - 1))")_${prev_name}"
  fi

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists, leaving it unchanged: ${session}"
    return
  fi

  tmux new-session -d -s "${session}" \
    "cd '${REPO_ROOT}' && \
      echo 'Queued ${name}: docs=${docs} vectors=${vectors} fine_k=${fine_k} coarse_k=${coarse_k} top_r=${top_r}' && \
      env REPO_ROOT='${REPO_ROOT}' PYTHON_BIN='${PYTHON_BIN}' CUDA_VISIBLE_DEVICES='${CUDA_DEVICE}' USE_GPU='${USE_GPU}' \
        DATASET_NAME='${name}' DATASET_STEM='${stem}' RAW_SOURCE_DIR='${raw_source}' RAW_TARGET_DIR='${raw_target}' \
        OUTPUT_ROOT='${output_root}' INDEX_ROOT='${index_root}' RESULTS_DIR='${results_dir}' DOCS_PER_SHARD='${DOCS_PER_SHARD}' \
        FINE_K='${fine_k}' COARSE_K='${coarse_k}' SAMPLE_SIZE='${sample_size}' TOP_R='${top_r}' STAGES='${STAGES}' \
        M_INDEX='${M_INDEX}' EF_INDEX='${EF_INDEX}' BUILD_THREADS='${BUILD_THREADS}' SEARCH_THREADS='${SEARCH_THREADS}' \
        SKIP_SEARCH_AFTER_BUILD='1' MAX_PARALLEL='${MAX_PARALLEL}' POLL_SECS='${POLL_SECS}' QUEUE_ROOT='${QUEUE_ROOT}' \
        SESSION_NAME='${session}' STARTED_MARKER='${started_marker}' DONE_MARKER='${done_marker}' \
        PREV_STARTED_MARKER='${prev_started_marker}' bash '${REPO_ROOT}/scripts/run_ordered_gem_queue_session.sh'; \
      code=\$?; echo '${session} EXIT_CODE='\"\$code\"; exec bash"

  echo "queued ${session}"
  echo "  dataset=${name} docs=${docs} vectors=${vectors}"
  echo "  raw_source=${raw_source}"
  echo "  raw_target=${raw_target}"
  echo "  output=${output_root}"
  echo "  index=${index_root}"
  echo "  results=${results_dir}"
}

echo "Queue root: ${QUEUE_ROOT}"
echo "Max parallel sessions: ${MAX_PARALLEL}"
echo "Order: largest to smallest by corpus vectors"

for i in "${!ORDERED_DATASETS[@]}"; do
  IFS='|' read -r name fine_k coarse_k sample_size top_r docs vectors <<< "${ORDERED_DATASETS[$i]}"
  launch_dataset "${i}" "${name}" "${fine_k}" "${coarse_k}" "${sample_size}" "${top_r}" "${docs}" "${vectors}"
done
