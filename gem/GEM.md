# GEM Baseline Reproduction Guide

This README explains how to run the GEM baseline in this repository from raw ColBERT-style `.npy` embeddings to final evaluated metrics. It is written for the data layout used in our experiments, especially datasets under:

```bash
/data1/liuyaoyang/Papers/ACFDE/output/<dataset>/colbert
/data1/liuyaoyang/Papers/ACFDE/datasets/<dataset>
```

The quickest correctness check is SciDocs. The expected SciDocs outputs from our completed run are copied in:

```bash
/data/ali/gem/results/scidocs_verification
```

Use that first to verify the evaluator and result format before building a new index.

## 1. Directory Layout

The handoff directory is:

```bash
/data/ali/gem
```

Expected contents:

```text
/data/ali/gem/
  README.md
  hnswlib/
  scripts/
  results/scidocs_verification/
```

The original working repository is:

```bash
/home/ali/gem-baseline
```

Large preprocessed/index data are not duplicated into `/data/ali/gem` by default. Existing SciDocs data and index are here:

```bash
/data/ali/scidocs-gem-data
/data/ali/scidocs-gem-index
```

SciDocs raw ColBERT source and BEIR-format qrels are here:

```bash
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert
/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs
```

## 2. Environment Setup

### 2.1 System Dependencies

The C++ GEM binary requires:

```text
cmake
g++
zlib
Eigen3
OpenBLAS
OpenMP
```

On Ubuntu-like systems:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake zlib1g-dev libeigen3-dev libopenblas-dev
```

The current `hnswlib/CMakeLists.txt` links OpenBLAS from:

```text
/usr/lib/x86_64-linux-gnu/libopenblas.so
```

If OpenBLAS is installed somewhere else, edit this line in `hnswlib/CMakeLists.txt`:

```cmake
set(OpenBLAS_LIBRARIES "/usr/lib/x86_64-linux-gnu/libopenblas.so")
```

### 2.2 Python Environment

Use Python 3.10+ or the existing environment:

```bash
/data/ali/gem-baseline/bin/python
```

The environment used in our run has:

```text
numpy 2.4.4
faiss 1.14.1
tqdm 4.67.3
```

To create a new environment:

```bash
python3 -m venv /data/ali/gem-venv
source /data/ali/gem-venv/bin/activate
pip install --upgrade pip
pip install numpy tqdm faiss-cpu
```

If you have GPU FAISS installed, use it for centroid training and code assignment:

```bash
python - <<'PY'
import faiss
print("faiss version:", faiss.__version__)
print("num gpus:", faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else "unknown")
PY
```

If GPU FAISS is unavailable, set `USE_GPU=0` in the commands below. CPU preprocessing is slower but should produce the same file format.

### 2.3 Build The GEM Binary

From the handoff directory:

```bash
cd /data/ali/gem
cmake -S hnswlib -B hnswlib/build
cmake --build hnswlib/build --target example_vecset_search_gem -j 8
```

Expected binary:

```bash
/data/ali/gem/hnswlib/build/example_vecset_search_gem
```

If running from the original repository:

```bash
cd /home/ali/gem-baseline
cmake -S hnswlib -B hnswlib/build
cmake --build hnswlib/build --target example_vecset_search_gem -j 8
```

## 3. Input Data Contract

The generic GEM pipeline expects raw ColBERT data as `.npy` files. The raw source directory must contain:

```text
corpus_points.npy
corpus_offsets.npy
query_points.npy
query_offsets.npy
```

For dataset name `scidocs`, the raw source directory is:

```bash
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert
```

### 3.1 Raw Embedding Files

`corpus_points.npy`

```text
shape: (total_document_vectors, dim)
dtype: float32 or float16
meaning: all document token vectors concatenated together
```

`corpus_offsets.npy`

```text
shape: (num_documents + 1,)
dtype: int64 or integer-compatible
meaning: offsets into corpus_points.npy
document i vectors are corpus_points[corpus_offsets[i]:corpus_offsets[i + 1]]
```

`query_points.npy`

```text
shape: (total_query_vectors, dim)
dtype: float32 or float16
meaning: all query token vectors concatenated together
```

`query_offsets.npy`

```text
shape: (num_queries + 1,)
dtype: int64 or integer-compatible
meaning: offsets into query_points.npy
query i vectors are query_points[query_offsets[i]:query_offsets[i + 1]]
```

For our ColBERT runs:

```text
dim = 128
```

SciDocs raw data currently has:

```bash
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/corpus_points.npy
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/corpus_offsets.npy
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/query_points.npy
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/query_offsets.npy
```

### 3.2 Offset Validity Requirements

Offsets must satisfy:

```text
offsets.ndim == 1
len(offsets) == number_of_items + 1
offsets[0] == 0
offsets is monotonic non-decreasing
offsets[-1] == points.shape[0]
```

Per-item vector counts are computed as:

```python
counts = np.diff(offsets).astype(np.int64)
```

No document/query should have a negative vector count. Zero-length documents are not recommended and may break scoring.

### 3.3 BEIR-Style Evaluation Files

For evaluation and numeric qrels preparation, the dataset directory should contain:

```text
queries.jsonl
corpus.jsonl
qrels/test.tsv or qrels/dev.tsv
```

`queries.jsonl` rows must contain:

```json
{"_id": "query_id", "text": "...", "metadata": {}}
```

`corpus.jsonl` rows must contain:

```json
{"_id": "doc_id", "title": "...", "text": "...", "metadata": {}}
```

The qrels TSV must have a header:

```text
query-id	corpus-id	score
```

Only rows with `score > 0` are used as positives.

## 4. GEM Preprocessed Data Format

The preprocessing pipeline writes a GEM dataset root:

```bash
/data/ali/<dataset>-gem-data
```

For SciDocs:

```bash
/data/ali/scidocs-gem-data
```

Expected files:

```text
docdata/encoding*_float16.npy
docdata/doclens*.npy
docdata/doc_codes_*.npy
qdata/qembs.npy
qdata/filterd_query.npy
qdata/filterd_query_len.npy
qdata/qrels.tsv
cdata/centroids.npy
cdata/coarse_centroids.npy
cdata/coarse_cluster_info.txt
```

### 4.1 Document Shards

`docdata/encoding<i>_float16.npy`

```text
shape: (vectors_in_shard, dim)
dtype: float16
meaning: full document vectors for documents in shard i
```

`docdata/doclens<i>.npy`

```text
shape: (documents_in_shard,)
dtype: int64
meaning: number of vectors for each document in this shard
sum(doclens<i>) == encoding<i>_float16.shape[0]
```

`docdata/doc_codes_<i>.npy`

```text
shape: (vectors_in_shard,)
dtype: int32
meaning: fine centroid id for each document vector
```

### 4.2 Query Files

If all queries have the same number of vectors, preprocessing writes:

```text
qdata/qembs.npy
shape: (num_queries, query_vec_count, dim)
dtype: float32
```

If query lengths vary, preprocessing writes:

```text
qdata/filterd_query.npy
shape: (total_query_vectors, dim)
dtype: float32

qdata/filterd_query_len.npy
shape: (num_queries,)
dtype: int64
```

SciDocs uses dense query layout:

```text
/data/ali/scidocs-gem-data/qdata/qembs.npy
```

### 4.3 Clustering Files

`cdata/centroids.npy`

```text
shape: (fine_k, dim)
dtype: float16
meaning: fine quantization codebook C_quant
```

`cdata/coarse_centroids.npy`

```text
shape: (coarse_k, dim)
dtype: float32
meaning: coarse routing/index clusters C_index
```

`cdata/coarse_cluster_info.txt`

```text
one line per coarse cluster
each line contains integer document labels assigned to that coarse cluster
```

The current default assignment keeps the top `TOP_R` coarse clusters per document. For SciDocs verification, `TOP_R=10`.

### 4.4 Numeric qrels For Internal GEM Metrics

`qdata/qrels.tsv` is optional for search, but useful for internal logging. It contains numeric internal IDs:

```text
qid pid
```

No header is required by the C++ loader. This file is generated from BEIR qrels by:

```bash
scripts/prepare_beir_qrels_for_gem.py
```

## 5. Full SciDocs Build/Search/Evaluation From Raw NPY

This section rebuilds everything from the raw SciDocs `.npy` files.

### 5.1 Define Paths

```bash
export REPO_ROOT=/data/ali/gem
export PYTHON_BIN=/data/ali/gem-baseline/bin/python

export DATASET_NAME=scidocs
export DATASET_STEM=scidocs
export RAW_SOURCE_DIR=/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert
export RAW_TARGET_DIR=/data/ali/scidocs-colbert
export OUTPUT_ROOT=/data/ali/scidocs-gem-data
export INDEX_ROOT=/data/ali/scidocs-gem-index
export RESULTS_DIR=/data/ali/gem/results/scidocs_full_run_$(date +%Y%m%d_%H%M%S)

export QUERIES_FILE=/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/queries.jsonl
export CORPUS_FILE=/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/corpus.jsonl
export QRELS_FILE=/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv
export QRELS_QUERY_ORDER=queries-jsonl
```

If using the original repository instead:

```bash
export REPO_ROOT=/home/ali/gem-baseline
```

### 5.2 Hyperparameters For SciDocs Verification

These are the exact settings from the saved SciDocs verification run:

```bash
export DOCS_PER_SHARD=25000
export FINE_K=4096
export COARSE_K=512
export SAMPLE_SIZE=1000000
export NITER=25
export SEED=123
export BATCH_SIZE=65536
export TOP_R=10
export USE_GPU=1

export M_INDEX=24
export EF_INDEX=80
export BUILD_THREADS=90
export SEARCH_THREADS=1
export BASE_FP32=0
export REPAIR_ON_BUILD=0
export REPAIR_ON_LOAD=0
export SKIP_SEARCH_AFTER_BUILD=0

export RERANK_LIST=512,1024
export EF_LIST=4000,8000,16000,24000
```

If GPU FAISS is not available:

```bash
export USE_GPU=0
```

### 5.3 Run The Complete Pipeline

```bash
cd "${REPO_ROOT}"

bash scripts/run_colbert_gem_index_pipeline.sh
```

This runs these stages:

```text
inspect
docdata
queries
fine-centroids
codes
coarse-centroids
coarse-info
build-index
```

What each stage does:

```text
inspect:
  prints raw embedding shapes and vector counts

docdata:
  writes docdata/encoding*_float16.npy and docdata/doclens*.npy

queries:
  writes qdata/qembs.npy or qdata/filterd_query.npy plus qdata/filterd_query_len.npy

fine-centroids:
  trains fine_k centroids from sampled document vectors and writes cdata/centroids.npy

codes:
  assigns one fine centroid code to every document vector and writes docdata/doc_codes_*.npy

coarse-centroids:
  clusters fine centroids into coarse_k coarse centroids and writes cdata/coarse_centroids.npy

coarse-info:
  builds cdata/coarse_cluster_info.txt using document top_r coarse-cluster memberships

build-index:
  builds GEM HNSW graph, saves index, and runs search unless SKIP_SEARCH_AFTER_BUILD=1
```

### 5.4 Run In tmux

For long builds, run in tmux:

```bash
tmux new-session -s gem-scidocs
cd /data/ali/gem
# export variables from sections 5.1 and 5.2
bash scripts/run_colbert_gem_index_pipeline.sh
```

Detach:

```text
Ctrl-b d
```

Reattach:

```bash
tmux attach -t gem-scidocs
```

Tail logs:

```bash
tail -f "${RESULTS_DIR}"/logs/pipeline_*.log
```

## 6. Search Only With An Existing Index

If preprocessing and index building are already done, run only search:

```bash
export REPO_ROOT=/data/ali/gem
export RESULTS_DIR=/data/ali/gem/results/scidocs_search_only_$(date +%Y%m%d_%H%M%S)
export LOG_DIR="${RESULTS_DIR}/logs"
export INDEX_ROOT=/data/ali/scidocs-gem-index

export GEM_DATASET=generic
export GEM_DATASET_NAME=scidocs
export GEM_DATASET_PATH=/data/ali/scidocs-gem-data
export GEM_NUM_CLUSTER=4096
export GEM_NUM_GRAPH_CLUSTER=512
export GEM_REBUILD=0
export GEM_SKIP_SEARCH=0

export GEM_EF_INDEX=80
export GEM_SEARCH_THREADS=1
export GEM_BUILD_THREADS=90
export GEM_MSMARCO_BASE_FP32=0
export GEM_APPLY_REPAIR_ON_LOAD=0
export GEM_NPROB=4
export GEM_EF_LIST=4000,8000,16000,24000
export GEM_RERANK_LIST=512,1024

cd "${REPO_ROOT}"
bash scripts/run_msmarco_gem_logged.sh
```

Important:

```text
GEM_REBUILD=0 means load existing index.
GEM_SEARCH_THREADS=1 gives single-threaded search timing.
GEM_NPROB is query cluster filter t.
GEM_EF_LIST is graph traversal budget sweep.
GEM_RERANK_LIST is final exact reranking candidate count sweep.
```

The result TSV format is:

```text
query_id<TAB>doc_id<TAB>score<TAB>rank
```

Example:

```text
0	12345	0.81234	1
```

The score is the positive version of the internal GEM distance/score used by the output writer.

## 7. Evaluation

Use:

```bash
scripts/evaluate_scidocs_gem_results.py
```

Despite the script name, it is generic for BEIR-style `queries.jsonl`, `corpus.jsonl`, and `qrels/*.tsv`.

### 7.1 Evaluate A SciDocs Result Directory

```bash
export REPO_ROOT=/data/ali/gem
export PYTHON_BIN=/data/ali/gem-baseline/bin/python
export RESULT_DIR=/data/ali/gem/results/scidocs_verification

"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_scidocs_gem_results.py" \
  --queries /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/queries.jsonl \
  --corpus /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/corpus.jsonl \
  --qrels /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv \
  --runs-glob "${RESULT_DIR}/*.tsv" \
  --k-values 10 100 \
  --output-csv "${RESULT_DIR}/scidocs_eval_recomputed.csv" \
  --log-file "${RESULT_DIR}/logs/gem_run_20260506_093112.log" \
  --query-order queries-jsonl
```

### 7.2 Query Order

Use:

```text
--query-order queries-jsonl
```

when GEM numeric query IDs follow the order of `queries.jsonl`. This is correct for SciDocs.

Use:

```text
--query-order qrels-first-seen
```

when the query embedding file contains only queries that appear in qrels, in the first-seen order of positive qrels. This was needed for some MSMARCO dev-only runs.

Always check evaluator output:

```text
mapped_positive_qrels should be nonzero
missing_positive_qrel_query_ids should be 0
missing_positive_qrel_doc_ids should be 0
common_queries should match the number of evaluated queries
```

For SciDocs verification, expected:

```text
queries_with_qrels = 1000
queries_with_results = 1000
common_queries = 1000
```

## 8. SciDocs Verification Results

The saved SciDocs verification run is:

```bash
/data/ali/gem/results/scidocs_verification
```

Original run settings:

```text
dataset=scidocs
fine_k=4096
coarse_k=512
top_r=10
M=24
ef_construction=80
GEM_NPROB=4
search_threads=1
base storage=fp16 in RAM
rerank_list=512,1024
ef_list=4000,8000,16000,24000
```

Expected evaluation CSV:

```bash
/data/ali/gem/results/scidocs_verification/scidocs_eval.csv
```

Expected metrics:

| result | MRR@10 | NDCG@10 | Recall@10 | MRR@100 | NDCG@100 | Recall@100 | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| rerank1024 ef4000 | 0.274601 | 0.152603 | 0.158183 | 0.286063 | 0.217920 | 0.350333 | 14.5520 |
| rerank1024 ef8000 | 0.274601 | 0.152581 | 0.158183 | 0.286047 | 0.217825 | 0.350133 | 13.7507 |
| rerank1024 ef16000 | 0.274601 | 0.152581 | 0.158183 | 0.286045 | 0.217892 | 0.350333 | 13.1653 |
| rerank1024 ef24000 | 0.274601 | 0.152581 | 0.158183 | 0.286045 | 0.217892 | 0.350333 | 13.1192 |
| rerank512 ef4000 | 0.273856 | 0.151965 | 0.156983 | 0.285269 | 0.215712 | 0.344233 | 21.1808 |
| rerank512 ef8000 | 0.273856 | 0.151845 | 0.156783 | 0.285254 | 0.215512 | 0.343633 | 19.9791 |
| rerank512 ef16000 | 0.273856 | 0.151845 | 0.156783 | 0.285252 | 0.215579 | 0.343833 | 18.6810 |
| rerank512 ef24000 | 0.273856 | 0.151845 | 0.156783 | 0.285252 | 0.215579 | 0.343833 | 18.7357 |

Small floating-point differences are possible across machines, compiler versions, BLAS versions, and FAISS versions. The result should be close.

## 9. Running Another Dataset

For another ColBERT dataset with the same raw `.npy` structure:

```bash
export REPO_ROOT=/data/ali/gem
export PYTHON_BIN=/data/ali/gem-baseline/bin/python

export DATASET_NAME=fiqa
export DATASET_STEM=fiqa
export RAW_SOURCE_DIR=/data1/liuyaoyang/Papers/ACFDE/output/fiqa/colbert
export RAW_TARGET_DIR=/data/ali/fiqa-colbert
export OUTPUT_ROOT=/data/ali/fiqa-gem-data
export INDEX_ROOT=/data/ali/fiqa-gem-index
export RESULTS_DIR=/data/ali/gem/results/fiqa_gem_$(date +%Y%m%d_%H%M%S)

export QUERIES_FILE=/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/queries.jsonl
export CORPUS_FILE=/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/corpus.jsonl
export QRELS_FILE=/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa/qrels/test.tsv
export QRELS_QUERY_ORDER=qrels-first-seen

export DOCS_PER_SHARD=25000
export FINE_K=32768
export COARSE_K=1024
export SAMPLE_SIZE=1000000
export TOP_R=10
export USE_GPU=1
export M_INDEX=24
export EF_INDEX=80
export BUILD_THREADS=90
export SEARCH_THREADS=1
export RERANK_LIST=512
export EF_LIST=4000,8000,16000

cd "${REPO_ROOT}"
bash scripts/run_colbert_gem_index_pipeline.sh
```

Then evaluate:

```bash
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/evaluate_scidocs_gem_results.py" \
  --queries "${QUERIES_FILE}" \
  --corpus "${CORPUS_FILE}" \
  --qrels "${QRELS_FILE}" \
  --runs-glob "${RESULTS_DIR}/*.tsv" \
  --k-values 1 5 10 20 50 100 \
  --output-csv "${RESULTS_DIR}/eval.csv" \
  --log-file "$(ls -t "${RESULTS_DIR}"/logs/gem_run_*.log | head -1)" \
  --query-order "${QRELS_QUERY_ORDER}"
```

## 10. Hyperparameter Meaning

`FINE_K`

```text
Number of fine quantization centroids |C_quant|.
Large values improve quantization resolution but increase preprocessing memory/time.
```

`COARSE_K`

```text
Number of coarse routing/index clusters |C_index|.
This controls graph cluster routing and index naming.
```

`TOP_R`

```text
Number of coarse clusters retained per document in coarse_cluster_info.txt.
Larger values increase routed candidate coverage but make search slower.
```

`M_INDEX`

```text
HNSW graph degree parameter M.
Current paper-style setting is 24.
```

`EF_INDEX`

```text
HNSW construction ef_construction.
Current paper-style setting is 80.
```

`GEM_NPROB`

```text
Query cluster filter t.
At search time, GEM selects top t coarse clusters per query vector, unions matching document labels, and restricts graph traversal to that routed subgraph.
```

`GEM_EF_LIST`

```text
Search-time graph traversal budget sweep.
Higher ef can improve recall but increases latency.
```

`GEM_RERANK_LIST`

```text
Number of graph candidates reranked by exact multivector score.
Higher rerankK can improve final metrics but increases latency.
```

`BASE_FP32`

```text
0: load document vectors from fp16 storage into fp16 RAM representation.
1: decode document vectors to fp32 in RAM.
fp32 may be useful for exactness experiments but uses more RAM.
```

`SEARCH_THREADS`

```text
Use 1 for single-threaded search timing.
Build threads are controlled separately by BUILD_THREADS.
```

## 11. Common Failures And Fixes

### 11.1 Metadata Mismatch When Loading Index

If search fails with an index metadata mismatch, check:

```text
GEM_DATASET_NAME
GEM_NUM_CLUSTER
GEM_NUM_GRAPH_CLUSTER
GEM_EF_INDEX
INDEX_ROOT
```

The index path is:

```text
${INDEX_ROOT}/${GEM_DATASET_NAME}Index${GEM_NUM_GRAPH_CLUSTER}_all_24_${GEM_EF_INDEX}/0.bin
```

For SciDocs:

```bash
/data/ali/scidocs-gem-index/scidocsIndex512_all_24_80/0.bin
```

### 11.2 Evaluation Gives `common_queries=0`

The query ID mapping is wrong. Retry evaluation with:

```bash
--query-order qrels-first-seen
```

or:

```bash
--query-order queries-jsonl
```

depending on how the query embedding file was generated.

### 11.3 Missing `qdata/qrels.tsv`

Search itself does not require qrels. Internal GEM recall logging and final evaluation require qrels. Generate internal numeric qrels with:

```bash
python scripts/prepare_beir_qrels_for_gem.py \
  --queries "${QUERIES_FILE}" \
  --corpus "${CORPUS_FILE}" \
  --qrels "${QRELS_FILE}" \
  --output "${OUTPUT_ROOT}/qdata/qrels.tsv" \
  --query-order "${QRELS_QUERY_ORDER}"
```

### 11.4 FAISS GPU Not Available

Set:

```bash
export USE_GPU=0
```

The pipeline will use CPU FAISS.

### 11.5 Search Is Slow

Check:

```text
GEM_NPROB
GEM_EF_LIST
GEM_RERANK_LIST
SEARCH_THREADS
```

For timing comparisons, keep `SEARCH_THREADS=1`.

### 11.6 Build Is Slow

Graph construction is expensive. Use:

```bash
export BUILD_THREADS=90
```

Do not increase parallelism if other large jobs are running on the server.

## 12. Minimal Verification Checklist

Run these commands after setup:

```bash
cd /data/ali/gem
cmake --build hnswlib/build --target example_vecset_search_gem -j 8

/data/ali/gem-baseline/bin/python scripts/evaluate_scidocs_gem_results.py \
  --queries /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/queries.jsonl \
  --corpus /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/corpus.jsonl \
  --qrels /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv \
  --runs-glob '/data/ali/gem/results/scidocs_verification/*.tsv' \
  --k-values 10 100 \
  --output-csv /data/ali/gem/results/scidocs_verification/scidocs_eval_recomputed.csv \
  --log-file /data/ali/gem/results/scidocs_verification/logs/gem_run_20260506_093112.log \
  --query-order queries-jsonl
```

Then compare:

```bash
cat /data/ali/gem/results/scidocs_verification/scidocs_eval.csv
cat /data/ali/gem/results/scidocs_verification/scidocs_eval_recomputed.csv
```

The values should match or be very close.

