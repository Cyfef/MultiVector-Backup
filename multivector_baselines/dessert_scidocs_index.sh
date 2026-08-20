#!/usr/bin/env bash

set -euo pipefail

USERNAME='chenyifeng'
DATASET="scidocs"
MANIFEST="/data1/${USERNAME}/Dataset/multi-vector-retrieval/FlatData/${DATASET}/manifest.json"
    
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES=2
export CXX=/usr/bin/g++-9
export CC=/usr/bin/gcc-9

"${PYTHON_BIN}" script/flat_multivector/build_plaid_from_flat_dataset.py \
--username "${USERNAME}" \
--dataset "${DATASET}" \
--manifest "${MANIFEST}" 

INDEX_ROOT="/data1/${USERNAME}/Dataset/multi-vector-retrieval/Index/${DATASET}/plaid"

"${PYTHON_BIN}" script/evaluation/eval_dessert.py \
  --username "${USERNAME}" \
  --dataset "${DATASET}"

"${PYTHON_BIN}" script/flat_multivector/eval_flat_groundtruth.py \
  --username "${USERNAME}" \
  --dataset "${DATASET}" \
  --manifest "${MANIFEST}" \
  --method dessert