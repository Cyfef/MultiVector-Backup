#!/bin/bash

# ========== 全局配置 ==========
DATASET="scidocs"                       
DATA_ROOT="/data1/chenyifeng/Data_backup"
WORK_ROOT="/data1/chenyifeng/MultiVector-Backup/EMVB/work"

# 由 DATASET 派生的目录
COLBERT_DIR="${DATA_ROOT}/${DATASET}/colbert"
BEIR_DIR="${DATA_ROOT}/${DATASET}/beir"
LOG_DIR="${WORK_ROOT}/logs"
mkdir -p "${LOG_DIR}"

# ========== 定义要遍历的参数列表（请根据实际需要修改） ==========
NLIST_VALUES=(2048 4096 8192)          # nlist 候选值
PQ_M_VALUES=(16 32 64)                # pq-m 候选值

# ========== 两层循环构建 EMVB 索引 ==========
for NLIST in "${NLIST_VALUES[@]}"; do
  for PQ_M in "${PQ_M_VALUES[@]}"; do
    echo "=========================================="
    echo "Building EMVB index with nlist=${NLIST}, pq-m=${PQ_M}"
    echo "=========================================="
    
    # 动态生成索引目录和结果目录（包含参数信息）
    INDEX_DIR="${WORK_ROOT}/indexes/${DATASET}_colbert/emvb_ivfpq_l2_nlist${NLIST}_m${PQ_M}"
    RESULTS_DIR="${WORK_ROOT}/results/${DATASET}_colbert/emvb_ivfpq_l2_nlist${NLIST}_m${PQ_M}"
    mkdir -p "${INDEX_DIR}" "${RESULTS_DIR}"
    
    # 日志文件（区分参数）
    LOG_FILE="${LOG_DIR}/prepare_${DATASET}_colbert_nlist${NLIST}_m${PQ_M}.log"
    
    # 运行构建命令
    python -u prepare_emvb_data.py \
      --base-embeddings "${COLBERT_DIR}/corpus_points.npy" \
      --base-offsets "${COLBERT_DIR}/corpus_offsets.npy" \
      --query-embeddings "${COLBERT_DIR}/query_points.npy" \
      --query-offsets "${COLBERT_DIR}/query_offsets.npy" \
      --output-dir "${INDEX_DIR}" \
      --nlist "${NLIST}" \
      --pq-m "${PQ_M}" \
      --add-batch-size 200000 \
      --faiss-threads 48 \
      2>&1 | tee "${LOG_FILE}"

    # 运行比例扫描
    echo "Running ratio sweep for dataset: ${DATASET}"
    python run_emvb_ratio_sweep.py \
      --dataset "${DATASET}" \
      --dataset-dir "${BEIR_DIR}" \
      --split test \
      --index-dir "${INDEX_DIR}" \
      --results-dir "${RESULTS_DIR}" \
      --ratios 1 \
      --k-values 100 \
      --query-ids-file "${COLBERT_DIR}/query_ids.txt" \
      --query-id-mode positional
  done
done

echo "All done for dataset: ${DATASET}"