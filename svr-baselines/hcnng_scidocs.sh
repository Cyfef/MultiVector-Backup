#!/bin/bash

# ==================== 全局变量设置 ====================
DATASET="scidocs"                                      # 数据集名称
BASE_DIR="/data1/chenyifeng/MultiVector-Backup/svr-baselines"   # 仓库根目录
DATA_BASE="/data1/chenyifeng/Data_backup"              # 原始数据根目录

# 数据集路径（colbert 格式）
DATASET_DIR="${DATA_BASE}/${DATASET}/single/colbert_sv"

# 本实验运行目录
RUN_DIR="${BASE_DIR}/runs/${DATASET}-single"
INPUTS_DIR="${RUN_DIR}/inputs"
HCNNG_DIR="${RUN_DIR}/hcnng"
HNSWLIB_DIR="${RUN_DIR}/hnswlib"
LOGS_DIR="${RUN_DIR}/logs"

# 数据集文件前缀
PREFIX="${DATASET}_sv"

max_calc_lists=(10 50 100 200 400 500 1000 2000 3000 4000 5000)

# ==================== 准备数据 ====================
mkdir -p "${INPUTS_DIR}" "${HCNNG_DIR}" "${HNSWLIB_DIR}" "${LOGS_DIR}"

python3 $BASE_DIR/runs/prepare_sv_data.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${INPUTS_DIR}" \
  --num-queries 1000 \
  --dataset-prefix "${PREFIX}"

# ==================== 构建 HCNNG 索引 ====================
$BASE_DIR/hcnng/hcnng \
  "${DATASET_DIR}/${PREFIX}_base.fvecs" \
  1000 \
  20 \
  "${HCNNG_DIR}/index.ivecs"

# ==================== 查询（遍历 max_calc） ====================
for max_calc in ${max_calc_lists[*]}
do
    # 为每个 max_calc 创建独立子目录
    sub_dir="${HCNNG_DIR}/maxcalc_${max_calc}"
    mkdir -p "${sub_dir}"

    # 结果文件和日志文件（按 max_calc 命名）
    ans_file="${sub_dir}/ans_k100_scored.tsv"
    qps_file="${sub_dir}/qps_k100_scored.tsv"
    log_file="${LOGS_DIR}/hcnng_k100_scored_maxcalc_${max_calc}.log"

    $BASE_DIR/hcnng/search \
        "${DATASET_DIR}/${PREFIX}_base.fvecs" \
        "${INPUTS_DIR}/query_1000.fvecs" \
        "${INPUTS_DIR}/groundtruth_1000.ivecs" \
        "${HCNNG_DIR}/index.ivecs" \
        100 \
        ${max_calc} \
        "${ans_file}" \
        "${qps_file}" \
        > "${log_file}" 2>&1
done

# ==================== 评估 BEIR 指标（每个 max_calc 单独评估） ====================
for max_calc in ${max_calc_lists[*]}
do
    sub_dir="${HCNNG_DIR}/maxcalc_${max_calc}"
    ans_file="${sub_dir}/ans_k100_scored.tsv"
    json_file="${sub_dir}/beir_metrics_k100_scored.json"

    python3 $BASE_DIR/runs/eval_beir_metrics.py \
      --groundtruth "${DATASET_DIR}/${PREFIX}_groundtruth_origin.ivecs" \
      --results "${ans_file}" \
      --k-values 100 \
      --output-json "${json_file}"
done