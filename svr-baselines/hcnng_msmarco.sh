#!/bin/bash

# ==================== 全局变量设置 ====================
DATASET="msmarco"                                      # 数据集名称
BASE_DIR="/data1/chenyifeng/MultiVector-Backup/svr-baselines"   # 仓库根目录
DATA_BASE="/data1/chenyifeng/Data_backup"              # 原始数据根目录
NUM_QUERIES=1000

# 数据集路径（colbert 格式）
DATASET_DIR="${DATA_BASE}/${DATASET}/single/colbert_sv"

# 本实验运行目录
RUN_DIR="${BASE_DIR}/runs/${DATASET}-single"
INPUTS_DIR="${RUN_DIR}/inputs"
HCNNG_DIR="${RUN_DIR}/hcnng"
LOGS_DIR="${RUN_DIR}/logs"

# 数据集文件前缀
PREFIX="${DATASET}_sv"

# ==================== 超参数网格 ====================
# minsize_cl (n) 和 num_cl (m) 的候选值
n_list=(500 1000 1500 2000)
m_list=(5 10 15 20 25 30 35 40)

#  max_calc
max_calc_list=(10 20 30 40 50 80 100 200 300 400 500 700 1000 2000 3000 4000 5000)   

# ==================== 准备数据（只执行一次） ====================
mkdir -p "${INPUTS_DIR}" "${HCNNG_DIR}" "${HNSWLIB_DIR}" "${LOGS_DIR}"

python3 $BASE_DIR/runs/prepare_sv_data.py \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${INPUTS_DIR}" \
  --num-queries $NUM_QUERIES \
  --dataset-prefix "${PREFIX}"

# ==================== 遍历所有 (n, m) 组合 ====================
for n in "${n_list[@]}"; do
  for m in "${m_list[@]}"; do
    # 为该组合创建独立目录
    combo_dir="${HCNNG_DIR}/n${n}_m${m}"
    mkdir -p "${combo_dir}"

    echo "=================================================="
    echo "Building graph for n=${n}, m=${m} ..."
    echo "=================================================="

    # ---------- 1. 建图 ----------
    $BASE_DIR/hcnng/hcnng \
      "${DATASET_DIR}/${PREFIX}_base.fvecs" \
      ${n} \
      ${m} \
      "${combo_dir}/index.ivecs"

    # ---------- 2. 对每个 max_calc 执行搜索和评估 ----------
    for max_calc in "${max_calc_list[@]}"; do
      echo "  Running search for max_calc=${max_calc} ..."

      # 每个 max_calc 单独子目录
      sub_dir="${combo_dir}/maxcalc_${max_calc}"
      mkdir -p "${sub_dir}"

      ans_file="${sub_dir}/ans_k100_scored.tsv"
      qps_file="${sub_dir}/qps_k100_scored.tsv"
      log_file="${LOGS_DIR}/hcnng_n${n}_m${m}_maxcalc${max_calc}.log"

      # 执行搜索
      OMP_NUM_THREADS=1 $BASE_DIR/hcnng/search \
          "${DATASET_DIR}/${PREFIX}_base.fvecs" \
          "${INPUTS_DIR}/query_1000.fvecs" \
          "${INPUTS_DIR}/groundtruth_1000.ivecs" \
          "${combo_dir}/index.ivecs" \
          100 \
          ${max_calc} \
          "${ans_file}" \
          "${qps_file}" \
          > "${log_file}" 2>&1

      # 评估 BEIR 指标
      json_file="${sub_dir}/beir_metrics_k100_scored.json"
      python3 $BASE_DIR/runs/eval_beir_metrics.py \
          --groundtruth "${DATASET_DIR}/${PREFIX}_groundtruth_origin.ivecs" \
          --results "${ans_file}" \
          --k-values 100 \
          --output-json "${json_file}"
    done

    echo "Finished n=${n}, m=${m}."
    echo ""
  done
done

echo "All parameter combinations completed."

# ==================== 汇总所有组合的结果 ====================
echo "开始汇总各组合的指标..."
python3 $BASE_DIR/runs/aggregate_hcnng.py "${HCNNG_DIR}"
echo "所有汇总完成。"