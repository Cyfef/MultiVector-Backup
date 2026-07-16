#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ali/gem-baseline}"
QUEUE_ROOT="${QUEUE_ROOT:?Set QUEUE_ROOT}"
SESSION_NAME="${SESSION_NAME:?Set SESSION_NAME}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
POLL_SECS="${POLL_SECS:-30}"
PREV_STARTED_MARKER="${PREV_STARTED_MARKER:-}"
STARTED_MARKER="${STARTED_MARKER:?Set STARTED_MARKER}"
DONE_MARKER="${DONE_MARKER:?Set DONE_MARKER}"
PENDING_CLAIM="${QUEUE_ROOT}/pending/${SESSION_NAME}.claim"
DISPATCH_LOCK="${QUEUE_ROOT}/dispatch.lock"

mkdir -p "${QUEUE_ROOT}/pending" "${QUEUE_ROOT}/started" "${QUEUE_ROOT}/done"

cleanup() {
  rm -f "${PENDING_CLAIM}"
}
trap cleanup EXIT

worker_count() {
  pgrep -fc 'generate_msmarco_gem_data.py|example_vecset_search_gem' || true
}

pending_count() {
  find "${QUEUE_ROOT}/pending" -maxdepth 1 -type f 2>/dev/null | wc -l
}

if [[ -n "${PREV_STARTED_MARKER}" ]]; then
  echo "[${SESSION_NAME}] waiting for predecessor marker: ${PREV_STARTED_MARKER}"
  while [[ ! -f "${PREV_STARTED_MARKER}" ]]; do
    sleep "${POLL_SECS}"
  done
fi

baseline_workers=0
while true; do
  exec 9>"${DISPATCH_LOCK}"
  flock -w 5 9 || {
    exec 9>&-
    sleep "${POLL_SECS}"
    continue
  }

  workers="$(worker_count)"
  pending="$(pending_count)"
  total_pending_workers=$((workers + pending))

  if (( total_pending_workers < MAX_PARALLEL )); then
    : > "${PENDING_CLAIM}"
    baseline_workers="${workers}"
    echo "[${SESSION_NAME}] claimed launch slot with workers=${workers} pending=${pending}"
    flock -u 9
    exec 9>&-
    break
  fi

  echo "[${SESSION_NAME}] queue full, workers=${workers} pending=${pending}; sleeping ${POLL_SECS}s"
  flock -u 9
  exec 9>&-
  sleep "${POLL_SECS}"
done

echo "[${SESSION_NAME}] starting pipeline"
bash "${REPO_ROOT}/scripts/run_colbert_gem_index_pipeline.sh" &
pipeline_pid=$!

started=0
for _ in $(seq 1 120); do
  if ! kill -0 "${pipeline_pid}" 2>/dev/null; then
    break
  fi

  current_workers="$(worker_count)"
  if (( current_workers > baseline_workers )); then
    : > "${STARTED_MARKER}"
    rm -f "${PENDING_CLAIM}"
    started=1
    echo "[${SESSION_NAME}] worker detected, released queue gate"
    break
  fi

  sleep 5
done

if (( started == 0 )); then
  : > "${STARTED_MARKER}"
  rm -f "${PENDING_CLAIM}"
  echo "[${SESSION_NAME}] no worker detected before grace period ended; released queue gate to avoid deadlock"
fi

set +e
wait "${pipeline_pid}"
code=$?
set -e

: > "${DONE_MARKER}"
echo "[${SESSION_NAME}] finished with exit code ${code}"
exit "${code}"
