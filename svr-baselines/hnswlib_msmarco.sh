#!/bin/bash

# ==================== 全局变量设置 ====================
DATASET="scidocs"
BASE_DIR="/data1/chenyifeng/MultiVector-Backup/svr-baselines"
DATA_BASE="/data1/chenyifeng/Data_backup"
NUM_QUERIES=1000

DATASET_DIR="${DATA_BASE}/${DATASET}/single/colbert_sv"
RUN_DIR="${BASE_DIR}/runs/${DATASET}-single"
INPUTS_DIR="${RUN_DIR}/inputs"
HNSWLIB_DIR="${RUN_DIR}/hnswlib"
LOGS_DIR="${RUN_DIR}/logs"

PREFIX="${DATASET}_sv"

# ==================== 超参数网格 ====================
# M 和 ef_construction 的组合（索引构建参数）
M_list=(2 3 6 8 12 20 32 40 64)
ef_construction_list=(100 200 400 600 800 1000)

# ef_search 扫描范围（查询参数）
ef_search_list=(5 10 20 30 50 100 150 200 300 500 800 1000 1200 2000 3000 4000 5000 6000 7000 8000)


# ==================== 遍历所有 (M, ef_construction) 组合 ====================
for M in "${M_list[@]}"; do
  for ef_construction in "${ef_construction_list[@]}"; do
    # 该组合的独立目录
    combo_dir="${HNSWLIB_DIR}/M${M}_efc${ef_construction}"
    mkdir -p "${combo_dir}"

    INDEX_FILE="${combo_dir}/index.bin"

    echo "=================================================="
    echo "Processing M=${M}, ef_construction=${ef_construction} ..."
    echo "=================================================="

    # ---------- 1. 构建索引（如果不存在） ----------
    if [ ! -f "${INDEX_FILE}" ]; then
        echo "  Building index (this may take a while)..."
        # 注意：构建时使用多线程（build_threads=96），索引保存到组合目录
        $BASE_DIR/runs/hnswlib_clerc_runner \
            "${DATASET_DIR}/${PREFIX}_base.fvecs" \
            "${INPUTS_DIR}/query_${NUM_QUERIES}.fvecs" \
            "${INDEX_FILE}" \
            "${combo_dir}/dummy.tsv" \
            ${NUM_QUERIES} \
            100 \
            ${M} \
            ${ef_construction} \
            10 \
            96 \
            1 \
            > "${LOGS_DIR}/build_M${M}_efc${ef_construction}.log" 2>&1
    else
        echo "  Index already exists, skipping build."
    fi

    # ---------- 2. 对每个 ef_search 执行搜索和评估 ----------
    for ef_search in "${ef_search_list[@]}"; do
        echo "  Running search for ef_search=${ef_search} ..."

        # 每个 ef_search 独立子目录
        sub_dir="${combo_dir}/ef_${ef_search}"
        mkdir -p "${sub_dir}"

        RESULT_FILE="${sub_dir}/ans_k100_scored.tsv"
        LOG_FILE="${LOGS_DIR}/hnswlib_M${M}_efc${ef_construction}_ef${ef_search}.log"

        # 执行搜索（使用已构建的索引）
        $BASE_DIR/runs/hnswlib_clerc_runner \
            "${DATASET_DIR}/${PREFIX}_base.fvecs" \
            "${INPUTS_DIR}/query_${NUM_QUERIES}.fvecs" \
            "${INDEX_FILE}" \
            "${RESULT_FILE}" \
            ${NUM_QUERIES} \
            100 \
            ${M} \
            ${ef_construction} \
            ${ef_search} \
            96 \
            1 \
            > "${LOG_FILE}" 2>&1

        # 评估 BEIR 指标
        JSON_FILE="${sub_dir}/beir_metrics_k100_scored.json"
        GROUNDTRUTH="${DATASET_DIR}/${PREFIX}_groundtruth_origin.ivecs"

        python3 ${BASE_DIR}/runs/eval_beir_metrics.py \
            --groundtruth "${GROUNDTRUTH}" \
            --results "${RESULT_FILE}" \
            --k-values 100 \
            --output-json "${JSON_FILE}"
    done

    echo "Finished M=${M}, ef_construction=${ef_construction}."
    echo ""
  done
done

echo "All parameter combinations completed."

# Argument meaning:

# - `1000`: number of queries to run
# - `100`: return top-100 neighbors
# - `16`: `M`
# - `200`: `ef_construction`
# - `100`: `ef_search`
# - `96`: build threads
# - `1`: search threads

# Important:

# - search is intentionally one query at a time with one search thread
# - the logged `[Search Time]` excludes result-file writing


# ==================== 汇总所有组合的结果 ====================
echo "开始汇总各组合的指标..."
python3 ${BASE_DIR}/runs/aggregate_hnswlib.py \
    --base-dir "${HNSWLIB_DIR}" \
    --logs-dir "${LOGS_DIR}"
echo "所有汇总完成。"