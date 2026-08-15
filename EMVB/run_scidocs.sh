#!/bin/bash

# ========== 全局配置 ==========
DATASET="scidocs"                       
DATA_ROOT="/data1/chenyifeng/Data_backup"
WORK_ROOT="/data1/chenyifeng/MultiVector-Backup/EMVB/work"

# 由 DATASET 派生的目录
COLBERT_DIR="${DATA_ROOT}/${DATASET}/colbert"
BEIR_DIR="${DATA_ROOT}/${DATASET}/beir"
INDEX_DIR="${WORK_ROOT}/indexes/${DATASET}_colbert/emvb_ivfpq_l2_nlist4096_m32"
RESULTS_DIR="${WORK_ROOT}/results/${DATASET}_colbert/emvb_ivfpq_l2_nlist4096_m32"
LOG_DIR="${WORK_ROOT}/logs"

# ========== 准备工作 ==========
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ========== 构建 EMVB 索引 ==========
echo "Building EMVB index for dataset: ${DATASET}"
python -u prepare_emvb_data.py \
  --base-embeddings "${COLBERT_DIR}/corpus_points.npy" \
  --base-offsets "${COLBERT_DIR}/corpus_offsets.npy" \
  --query-embeddings "${COLBERT_DIR}/query_points.npy" \
  --query-offsets "${COLBERT_DIR}/query_offsets.npy" \
  --output-dir "${INDEX_DIR}" \
  --nlist 4096 \
  --pq-m 32 \
  --add-batch-size 200000 \
  --faiss-threads 48 \
  2>&1 | tee "${LOG_DIR}/prepare_${DATASET}_colbert.log"

# ========== 运行比例扫描 ==========
echo "Running ratio sweep for dataset: ${DATASET}"
python run_emvb_ratio_sweep.py \
  --dataset "${DATASET}" \
  --dataset-dir "${BEIR_DIR}" \
  --split test \
  --index-dir "${INDEX_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --ratios 0.25 0.5 1 2 4 8 16 32 \
  --k-values 100 \
  --query-ids-file "${COLBERT_DIR}/query_ids.txt" \
  --query-id-mode positional

echo "All done for dataset: ${DATASET}"