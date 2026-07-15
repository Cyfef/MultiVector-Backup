#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <username> <dataset> [manifest]"
  exit 1
fi

USERNAME="$1"
DATASET="$2"
MANIFEST="${3:-/data1/${USERNAME}/Dataset/multi-vector-retrieval/FlatData/${DATASET}/manifest.json}"
EMBED_ROOT="/data1/${USERNAME}/Dataset/multi-vector-retrieval/Embedding/${DATASET}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd /home/ali/plaid-index

if [[ ! -f "${EMBED_ROOT}/base_embedding/encoding0_float32.npy" ]]; then
  echo "Canonical embedding files not found for ${DATASET}. Bootstrapping Plaid build first."
  "${PYTHON_BIN}" script/flat_multivector/build_plaid_from_flat_dataset.py \
    --username "${USERNAME}" \
    --dataset "${DATASET}" \
    --manifest "${MANIFEST}"
fi

"${PYTHON_BIN}" script/evaluation/eval_igp.py \
  --username "${USERNAME}" \
  --dataset_name "${DATASET}"

"${PYTHON_BIN}" script/flat_multivector/eval_flat_groundtruth.py \
  --username "${USERNAME}" \
  --dataset "${DATASET}" \
  --manifest "${MANIFEST}" \
  --method IGP
