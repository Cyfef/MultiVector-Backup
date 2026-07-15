#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <username> <dataset> [manifest]"
  exit 1
fi

USERNAME="$1"
DATASET="$2"
MANIFEST="${3:-/data1/${USERNAME}/Dataset/multi-vector-retrieval/FlatData/${DATASET}/manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd /home/ali/plaid-index

"${PYTHON_BIN}" script/flat_multivector/build_plaid_from_flat_dataset.py \
  --username "${USERNAME}" \
  --dataset "${DATASET}" \
  --manifest "${MANIFEST}"

"${PYTHON_BIN}" script/evaluation/eval_plaid.py \
  --username "${USERNAME}" \
  --dataset "${DATASET}"

"${PYTHON_BIN}" script/flat_multivector/eval_flat_groundtruth.py \
  --username "${USERNAME}" \
  --dataset "${DATASET}" \
  --manifest "${MANIFEST}" \
  --method plaid
