#!/bin/bash

# ==================== 全局变量设置 ====================
DATASET="scidocs"
BASE_DIR="/data1/chenyifeng/MultiVector-Backup/WARP"
DATA_BASE="/data1/chenyifeng/Data_backup"

EMBEDDING_DIR="${DATA_BASE}/${DATASET}/colbert"
BEIR_DIR="${DATA_BASE}/${DATASET}/beir"

INDEX_ROOT="${BASE_DIR}/indexes"
INDEX_NAME="beir-${DATASET}.split=test.precomputed=colbert.nbits=2"

# 输出根目录
OUTPUT_ROOT="${BASE_DIR}/my_runs/${DATASET}-colbert"
mkdir -p "${OUTPUT_ROOT}"

# Python 解释器（根据实际环境修改）
PYTHON_BIN="${CONDA_PREFIX}/bin/python"

export CXX=/usr/bin/g++-9

export CUDA_VISIBLE_DEVICES=3

# ==================== 定义搜索网格 ====================
thresholds=(0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55)
ndocs_list=(256 512 1024 2048 4096 8192)
ncells_values=(1 2 3 4 8 10 16 32 40 64 128 256 512)

# ==================== 构建索引（若已存在则跳过） ====================
INDEX_PATH="${INDEX_ROOT}/${INDEX_NAME}"
if [ -d "${INDEX_PATH}" ] && [ "$(ls -A ${INDEX_PATH})" ]; then
    echo "索引已存在: ${INDEX_PATH}，跳过构建。"
else
    echo "开始构建索引..."
    $PYTHON_BIN utility/index_from_embeddings.py \
      --dataset "$DATASET" \
      --embedding-dir "$EMBEDDING_DIR" \
      --index-root "$INDEX_ROOT" \
      --index-name "$INDEX_NAME" \
      --nbits 2 \
      --threads 48 \
      --max-partitions 32768 \
      --sample-per-centroid 64 \
      --max-sample-embeddings 2000000 \
      --chunk-size 50000
fi

# 环境变量（确保单线程，稳定测量 QPS）
export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1


# ==================== 网格搜索主循环 ====================
for thr in "${thresholds[@]}"; do
  for ndocs in "${ndocs_list[@]}"; do
    echo "======================================================"
    echo "运行: threshold=${thr}, ndocs=${ndocs}"
    echo "======================================================"

    # 为当前参数组合创建子目录
    sub_dir="${OUTPUT_ROOT}/thr${thr}_ndocs${ndocs}"
    mkdir -p "${sub_dir}"

    # 输出 CSV 路径
    csv_path="${sub_dir}/sweep.csv"

    # baseline 名称（可根据需要调整）
    baseline="warp_precomputed_colbert_nbits2_thr${thr}_ndocs${ndocs}"

    # 执行 sweep_ncells.py
    $PYTHON_BIN utility/sweep_ncells.py \
      --dataset "$DATASET" \
      --dataset-dir "$BEIR_DIR" \
      --embedding-dir "$EMBEDDING_DIR" \
      --index-root "$INDEX_ROOT" \
      --index "$INDEX_NAME" \
      --split test \
      --nbits 2 \
      --k 100 \
      --baseline "$baseline" \
      --centroid-score-threshold "$thr" \
      --ndocs "$ndocs" \
      --ncells ${ncells_values[@]} \
      --output-csv "$csv_path"

    echo "完成 threshold=${thr}, ndocs=${ndocs}，结果保存在 ${csv_path}"
    echo ""
  done
done

echo "==================== 网格搜索全部完成 ===================="
echo "所有结果存放在 ${OUTPUT_ROOT} 下的各子目录中。"