#!/usr/bin/env bash

set -euo pipefail

USERNAME='chenyifeng'
DATASET="scidocs"
MANIFEST="/data1/${USERNAME}/Dataset/multi-vector-retrieval/FlatData/${DATASET}/manifest.json"

partitions_multiplier_list=(4 16 32)          # --num_partitions_multiplier
typical_doclen_list=(10 120 500)
sample_multiplier_list=(2 16 32)            # --kmeans_sample_multiplier
          
PYTHON_BIN="${PYTHON_BIN:-python}"

export CUDA_VISIBLE_DEVICES=2
export CXX=/usr/bin/g++-9
export CC=/usr/bin/gcc-9

for p in "${partitions_multiplier_list[@]}"; do
  for s in "${sample_multiplier_list[@]}"; do
    for t in "${typical_doclen_list[@]}"; do
      echo "========================================"
      echo "Running combination: p=$p, s=$s, t=$t"
      echo "========================================"

      "${PYTHON_BIN}" script/flat_multivector/build_plaid_from_flat_dataset.py \
        --username "${USERNAME}" \
        --dataset "${DATASET}" \
        --manifest "${MANIFEST}" \
        --num_partitions_multiplier "${p}" \
        --kmeans_sample_multiplier "${s}" \
        --typical_doclen "${t}" \
        --index_subdir "p${p}s${s}t${t}"

      "${PYTHON_BIN}" script/evaluation/eval_plaid.py \
        --username "${USERNAME}" \
        --dataset "${DATASET}" \
        --index_subdir "p${p}s${s}t${t}"

      "${PYTHON_BIN}" script/flat_multivector/eval_flat_groundtruth_subdir.py \
        --username "${USERNAME}" \
        --dataset "${DATASET}" \
        --manifest "${MANIFEST}" \
        --method plaid \
        --index_subdir "p${p}s${s}t${t}"
    done
  done
done