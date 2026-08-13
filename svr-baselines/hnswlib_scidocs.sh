DATASET="scidocs"                                      # 数据集名称
NUM_QUERIES=1000

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

# 索引文件
INDEX_FILE="${HNSWLIB_DIR}/index.bin"

ef_search_lists=(5 10 20 30 50 100 150 200 300 500 800 1000 1200 2000 3000 4000 5000 6000 7000 8000)

for ef_search in ${ef_search_lists[*]}
do
    EF_OUT_DIR="${HNSWLIB_DIR}/ef_search_${ef_search}"
    mkdir -p "${EF_OUT_DIR}"

    RESULT_FILE="${EF_OUT_DIR}/ans_k100_scored.tsv"
    LOG_FILE="${LOGS_DIR}/hnswlib_ef_${ef_search}.log"

    echo "Running hnswlib with ef_search=${ef_search}" | tee -a "${LOG_FILE}"

    $BASE_DIR/runs/hnswlib_clerc_runner \
        "${DATASET_DIR}/${PREFIX}_base.fvecs" \
        "${INPUTS_DIR}/query_${NUM_QUERIES}.fvecs" \
        "${INDEX_FILE}" \
        "${RESULT_FILE}" \
        ${NUM_QUERIES} \
        100 \
        16 \
        200 \
        ${ef_search} \
        96 \
        1 \
        > "${LOG_FILE}" 2>&1
done

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


# ==================== 评估 BEIR 指标（每个 ef_search 单独评估） ====================
for ef_search in ${ef_search_lists[*]}
do

    EF_OUT_DIR="${HNSWLIB_DIR}/ef_search_${ef_search}"
    RESULT_FILE="${EF_OUT_DIR}/ans_k100_scored.tsv"
    JSON_FILE="${EF_OUT_DIR}/beir_metrics_k100_scored.json"
    GROUNDTRUTH="${DATASET_DIR}/${PREFIX}_groundtruth_origin.ivecs"

    python3 ${BASE_DIR}/runs/eval_beir_metrics.py \
        --groundtruth "${GROUNDTRUTH}" \
        --results "${RESULT_FILE}" \
        --k-values 100 \
        --output-json "${JSON_FILE}"
done