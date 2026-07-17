# EMVB Baseline Reproduction Guide

This directory contains the minimal code needed to build an EMVB index from multi-vector embeddings, run EMVB retrieval, and evaluate the TSV ranking output with BEIR qrels.

The commands below are written for the current machine paths. If you move the package, replace `/data/ali/EMVB` with the new code root and keep the same input/output structure.

## 1. What This Baseline Does

EMVB is a multi-vector retrieval baseline. The input is one packed matrix of document token/vector embeddings and one packed matrix of query token/vector embeddings. Each document and query can have a different number of vectors, so a second array stores the number of vectors per document/query, or an offsets array from which those counts are computed.

The pipeline has four stages:

1. Build an IVFPQ index with FAISS over all document vectors.
2. Export the FAISS index into the files consumed by the EMVB C++ search binary.
3. Run `perf_emvb` to produce a ranking TSV file.
4. Map positional EMVB query/document ids back to dataset ids and evaluate against BEIR qrels.

Search is intentionally run with one thread in the sweep script so QPS is comparable across runs. Index construction can use many FAISS threads.

## 2. Package Layout

After this handoff is prepared, the package lives at:

```bash
/data/ali/EMVB
```

Important files:

```text
CMakeLists.txt                         C++ build configuration
src/perf_emvb.cpp                      EMVB search binary source
include/*.hpp, include/utils.cpp       EMVB C++ support code
external/faiss                         FAISS source used by CMake
external/cnpy                          NPY reader used by C++
external/cmd_line_parser               C++ command-line parser
prepare_emvb_data.py                   converts embeddings into an EMVB index directory
run_emvb_ratio_sweep.py                runs search for k=10/k=100 and ratio sweeps
evaluate_beir_emvb.py                  evaluates EMVB TSV output with BEIR qrels
scripts/run_scidocs_colbert_simple_verify.sh  complete SciDocs verification example
reference_results/scidocs_colbert_simple/...  known-good SciDocs outputs
```

Recommended working directories:

```text
/data/ali/EMVB/work/indexes            generated EMVB indexes
/data/ali/EMVB/work/results            generated search/evaluation outputs
/data/ali/EMVB/logs                    tmux or shell logs
```

## 3. Environment Setup

### 3.1 Use Existing Environments On This Machine

The previous runs used these Python environments:

```bash
/data/ali/gem-baseline/bin/python      # FAISS/numpy environment for index construction
/data/ali/env/bin/python               # BEIR evaluation environment
```

Set these variables before running commands:

```bash
export EMVB_ROOT=/data/ali/EMVB
export PREP_PY=/data/ali/gem-baseline/bin/python
export EVAL_PY=/data/ali/env/bin/python
```

If Intel oneAPI is installed, load MKL before building:

```bash
source /opt/intel/oneapi/setvars.sh
```

If that file does not exist, verify MKL is still visible:

```bash
echo "$MKLROOT"
ldconfig -p | grep -i mkl | head
```

### 3.2 Create A Fresh Environment If Needed

Use this only if the existing environments are unavailable:

```bash
conda create -n emvb python=3.10 -y
conda activate emvb
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu numpy tqdm cmake make gxx_linux-64 mkl mkl-devel -y
pip install beir
```

For CPU-only indexing:

```bash
conda install -c conda-forge faiss-cpu numpy tqdm cmake make gxx_linux-64 mkl mkl-devel -y
pip install beir
```

The C++ code is compiled with AVX-512 flags in the current `CMakeLists.txt`. Run this to confirm the CPU supports AVX-512:

```bash
lscpu | grep -i avx512
```

If no AVX-512 flags are listed, the binary may fail at runtime with an illegal instruction. In that case, remove the AVX-512-specific compile flags from `CMakeLists.txt` and rebuild, accepting slower search.

## 4. Build The C++ Search Binary

From the package root:

```bash
cd /data/ali/EMVB
mkdir -p build
cd build
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF ..
make -j"$(nproc)"
```

Verify that the binary exists:

```bash
ls -lh /data/ali/EMVB/build/perf_emvb
```

Expected result: an executable file named `perf_emvb`.

If CMake cannot find MKL, load oneAPI with `source /opt/intel/oneapi/setvars.sh` or install `mkl` and `mkl-devel` inside the conda environment.

## 5. Input Data Format

EMVB expects multi-vector embeddings stored in NumPy `.npy` files. The embeddings must be `float32` or convertible to `float32`.

### 5.1 Offsets-Style Input

This is the format used by the ColBERT datasets in `/data1/liuyaoyang/Papers/ACFDE/output/.../colbert`.

Required files:

```text
corpus_points.npy     shape = (N_doc_vectors, dim)
corpus_offsets.npy    shape = (n_docs + 1,)
query_points.npy      shape = (N_query_vectors, dim)
query_offsets.npy     shape = (n_queries + 1,)
```

Rules:

```text
corpus_offsets[0] must be 0
query_offsets[0] must be 0
offset arrays must be strictly increasing
sum(diff(corpus_offsets)) must equal corpus_points.shape[0]
sum(diff(query_offsets)) must equal query_points.shape[0]
corpus_points.shape[1] must equal query_points.shape[1]
embedding dimension must be divisible by pq_m
```

Example:

```text
corpus_offsets = [0, 4, 10, 11, 14]
document 0 uses corpus_points[0:4]
document 1 uses corpus_points[4:10]
document 2 uses corpus_points[10:11]
document 3 uses corpus_points[11:14]
```

### 5.2 Doclens-Style Input

This is equivalent to offsets, but stores counts directly.

Required files:

```text
corpus_points.npy     shape = (N_doc_vectors, dim)
doclens.npy           shape = (n_docs,)
query_points.npy      shape = (N_query_vectors, dim)
query_doclens.npy     shape = (n_queries,)
```

Rules:

```text
all doclens and query_doclens values must be positive
sum(doclens) must equal corpus_points.shape[0]
sum(query_doclens) must equal query_points.shape[0]
```

### 5.3 Dataset IDs

The EMVB C++ binary writes positional ids:

```text
query_position<TAB>document_position<TAB>rank<TAB>score
```

Evaluation must map these positions to real dataset ids.

For BEIR-style datasets, `evaluate_beir_emvb.py` loads ids from:

```text
dataset_dir/corpus.jsonl
dataset_dir/queries.jsonl
dataset_dir/qrels/<split>.tsv
```

The qrels file must be tab-separated and have this header:

```text
query-id    corpus-id    score
```

If your embeddings use a custom ordering that differs from `corpus.jsonl` or `queries.jsonl`, create plain text files with one id per line in embedding order:

```text
corpus_ids.txt
query_ids.txt
```

Then pass:

```bash
--corpus-ids-file /path/to/corpus_ids.txt
--query-ids-file /path/to/query_ids.txt
--query-id-mode positional
```

If you evaluate a query subset, pass `--restrict-to-run-queries`.

## 6. Build An EMVB Index

This command builds the known SciDocs simple ColBERT index.

Input embeddings:

```text
/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert
```

BEIR dataset/qrels:

```text
/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs
```

Index output:

```text
/data/ali/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32
```

Run:

```bash
cd /data/ali/EMVB
mkdir -p logs

$PREP_PY -u prepare_emvb_data.py \
  --base-embeddings /data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/corpus_points.npy \
  --base-offsets /data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/corpus_offsets.npy \
  --query-embeddings /data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/query_points.npy \
  --query-offsets /data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert/query_offsets.npy \
  --output-dir /data/ali/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  --nlist 4096 \
  --pq-m 32 \
  --add-batch-size 200000 \
  --faiss-threads 48 \
  2>&1 | tee /data/ali/EMVB/logs/prepare_scidocs_colbert_simple.log
```

Expected SciDocs metadata from the known run:

```text
n_docs = 25657
n_queries = 1000
n_doc_vectors = 4820738
n_query_vectors = 48000
embedding_dim = 128
query_max_terms = 48
nlist = 4096
pq_m = 32
train_size = 163840
elapsed_seconds = about 222 seconds on the previous CPU run
```

The index directory must contain:

```text
alldoclens.npy
centroids.npy
centroids_to_pids.txt
faiss_ivfpq.index
index_assignment.npy
metadata.json
pq_centroids.npy
queries_id.txt
query_embeddings.npy
residuals.npy
```

## 7. Run One Search Manually

This command runs the base `k=10` SciDocs search using one thread.

```bash
cd /data/ali/EMVB
mkdir -p /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
/data/ali/EMVB/build/perf_emvb \
  -k 10 \
  -nprobe 4 \
  -thresh 0.4 \
  -out-second-stage 512 \
  -thresh-query 0.5 \
  -n-doc-to-score 4000 \
  -queries-id-file /data/ali/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/queries_id.txt \
  -alldoclens-path /data/ali/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/alldoclens.npy \
  -index-dir-path /data/ali/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  -out-file /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/results_k10.tsv \
  > /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10.log 2>&1
```

The output TSV has four columns and no header:

```text
query_position<TAB>document_position<TAB>rank<TAB>score
```

The log should contain a line like:

```text
Average Elapsed Time per query <number>
```

`evaluate_beir_emvb.py` parses this line and computes QPS as:

```text
qps = 1 / (avg_query_time_ns / 1e9)
```

## 8. Evaluate One Search

Evaluate the manual `k=10` SciDocs run:

```bash
cd /data/ali/EMVB

$EVAL_PY evaluate_beir_emvb.py \
  --dataset-dir /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs \
  --split test \
  --run /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/results_k10.tsv \
  --k-values 10 \
  --run-log /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10.log \
  --search-threads 1 \
  --output-json /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_k10.json \
  --output-csv /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_k10.csv
```

Metrics reported:

```text
NDCG@k
MAP@k
MRR@k
Recall@k
P@k
avg_query_time_ns
avg_query_time_s
qps
```

## 9. Run The Ratio Sweep

Use the sweep script to run ratios `0.25`, `0.5`, `1`, and `2` for both `k=10` and `k=100`.

```bash
cd /data/ali/EMVB
mkdir -p /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32

PYTHON_BIN=$EVAL_PY $EVAL_PY run_emvb_ratio_sweep.py \
  --dataset scidocs \
  --dataset-dir /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs \
  --split test \
  --index-dir /data/ali/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  --results-dir /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  --ratios 0.25 0.5 1 2 \
  --k-values 10 100 \
  --query-id-mode positional
```

The base search settings before applying the ratio are:

```text
k=10:  nprobe=4, thresh=0.4, thresh_query=0.5, out_second_stage=512,  n_doc_to_score=4000
k=100: nprobe=4, thresh=0.4, thresh_query=0.5, out_second_stage=1024, n_doc_to_score=4000
```

For each ratio:

```text
nprobe = ceil(base_nprobe * ratio), minimum 1
out_second_stage = round(base_out_second_stage * ratio), minimum k
n_doc_to_score = round(base_n_doc_to_score * ratio), minimum k
thresh and thresh_query stay fixed
```

The sweep writes:

```text
results_k10_r025.tsv
results_k10_r050.tsv
results_k10_r100.tsv
results_k10_r200.tsv
results_k100_r025.tsv
results_k100_r050.tsv
results_k100_r100.tsv
results_k100_r200.tsv
metrics_k*.json
metrics_k*.csv
run_k*.log
metrics_summary.csv
```

If a TSV or JSON already exists, the script reuses it. To force a rerun, delete the corresponding `results_*.tsv`, `run_*.log`, `metrics_*.json`, and `metrics_*.csv` files.

## 10. One-Command SciDocs Verification

The package includes:

```bash
/data/ali/EMVB/scripts/run_scidocs_colbert_simple_verify.sh
```

Run:

```bash
bash /data/ali/EMVB/scripts/run_scidocs_colbert_simple_verify.sh
```

The script:

1. Builds `build/perf_emvb` if it does not exist.
2. Builds the SciDocs simple ColBERT EMVB index if `metadata.json` does not exist.
3. Runs the full ratio sweep for `k=10` and `k=100`.
4. Writes generated results to `/data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32`.

Compare the generated summary to:

```bash
/data/ali/EMVB/reference_results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_summary.csv
```

Small QPS differences are expected across machines and load conditions. Metric values should match unless the index is rebuilt with a different FAISS version, random seed, input order, or search settings.

## 11. Known SciDocs Verification Results

These are the known-good simple ColBERT SciDocs results from the previous run:

| profile | ratio | k | nprobe | out_second_stage | n_doc_to_score | NDCG | MAP | MRR | Recall | P | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r025 | 0.25 | 10 | 1 | 128 | 1000 | 0.13850 | 0.07869 | 0.25645 | 0.14188 | 0.07000 | 48.17404 |
| r050 | 0.50 | 10 | 2 | 256 | 2000 | 0.14352 | 0.08207 | 0.26324 | 0.14708 | 0.07260 | 26.14865 |
| base | 1.00 | 10 | 4 | 512 | 4000 | 0.14614 | 0.08380 | 0.26547 | 0.15078 | 0.07450 | 15.71787 |
| r200 | 2.00 | 10 | 8 | 1024 | 8000 | 0.14791 | 0.08466 | 0.26880 | 0.15283 | 0.07550 | 7.57924 |
| r025 | 0.25 | 100 | 1 | 256 | 1000 | 0.19936 | 0.09458 | 0.27280 | 0.31163 | 0.01534 | 29.40055 |
| r050 | 0.50 | 100 | 2 | 512 | 2000 | 0.20634 | 0.09758 | 0.27627 | 0.32827 | 0.01616 | 15.40626 |
| base | 1.00 | 100 | 4 | 1024 | 4000 | 0.21245 | 0.09974 | 0.27955 | 0.34247 | 0.01686 | 8.32598 |
| r200 | 2.00 | 100 | 8 | 2048 | 8000 | 0.21502 | 0.10050 | 0.28038 | 0.34957 | 0.01721 | 4.23844 |

## 12. Running A New Dataset

Prepare these paths:

```bash
export DATASET_NAME=my_dataset
export EMB_DIR=/path/to/embeddings
export BEIR_DIR=/path/to/beir_dataset
export SPLIT=test
export INDEX_DIR=/data/ali/EMVB/work/indexes/$DATASET_NAME/emvb_ivfpq_l2_nlist4096_m32
export RESULT_DIR=/data/ali/EMVB/work/results/$DATASET_NAME/emvb_ivfpq_l2_nlist4096_m32
```

For offsets-style data:

```bash
$PREP_PY -u /data/ali/EMVB/prepare_emvb_data.py \
  --base-embeddings $EMB_DIR/corpus_points.npy \
  --base-offsets $EMB_DIR/corpus_offsets.npy \
  --query-embeddings $EMB_DIR/query_points.npy \
  --query-offsets $EMB_DIR/query_offsets.npy \
  --output-dir $INDEX_DIR \
  --nlist 4096 \
  --pq-m 32 \
  --add-batch-size 200000 \
  --faiss-threads 48
```

For doclens-style data, replace the offset arguments:

```bash
  --doclens $EMB_DIR/doclens.npy \
  --query-doclens $EMB_DIR/query_doclens.npy
```

Then run the sweep:

```bash
PYTHON_BIN=$EVAL_PY $EVAL_PY /data/ali/EMVB/run_emvb_ratio_sweep.py \
  --dataset $DATASET_NAME \
  --dataset-dir $BEIR_DIR \
  --split $SPLIT \
  --index-dir $INDEX_DIR \
  --results-dir $RESULT_DIR \
  --ratios 0.25 0.5 1 2 \
  --k-values 10 100 \
  --query-id-mode positional
```

Add custom id files if the embedding order differs from BEIR file order:

```bash
  --query-ids-file /path/to/query_ids.txt \
  --corpus-ids-file /path/to/corpus_ids.txt \
  --query-id-mode positional
```

Use a query subset by creating a plain text file of positional query ids and passing it to the sweep:

```bash
seq 0 999 > /data/ali/EMVB/work/query_ids_first1000.txt

PYTHON_BIN=$EVAL_PY $EVAL_PY /data/ali/EMVB/run_emvb_ratio_sweep.py \
  --dataset clerc \
  --dataset-dir /path/to/clerc/qrels_dataset \
  --split test \
  --index-dir $INDEX_DIR \
  --results-dir $RESULT_DIR \
  --ratios 0.25 0.5 1 2 \
  --k-values 10 100 \
  --search-query-ids-file /data/ali/EMVB/work/query_ids_first1000.txt \
  --restrict-to-run-queries \
  --query-id-mode positional
```

## 13. Running In tmux

Start a long index build:

```bash
tmux new -s emvb_scidocs
cd /data/ali/EMVB
bash scripts/run_scidocs_colbert_simple_verify.sh
```

Detach:

```text
Ctrl-b then d
```

List sessions:

```bash
tmux ls
```

Reattach:

```bash
tmux attach -t emvb_scidocs
```

Watch logs:

```bash
tail -f /data/ali/EMVB/logs/prepare_scidocs_colbert_simple.log
tail -f /data/ali/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10_r100.log
```

## 14. Troubleshooting

If evaluation metrics are all zero, check these first:

```text
1. The qrels path and split are correct, for example dataset_dir/qrels/test.tsv.
2. The query ids in the run match the qrels query ids after mapping.
3. The corpus ids in the run match the qrels corpus ids after mapping.
4. You used --query-id-mode positional when the TSV query column is 0, 1, 2, ...
5. You passed --query-ids-file and --corpus-ids-file when embedding order differs from BEIR order.
6. If using a query subset, you passed --restrict-to-run-queries.
```

If `prepare_emvb_data.py` fails with a shape error:

```text
1. Confirm embeddings are 2D.
2. Confirm offsets are 1D and start at 0.
3. Confirm offsets are strictly increasing.
4. Confirm sum(diff(offsets)) equals the number of embedding rows.
5. Confirm document and query embedding dimensions match.
6. Confirm embedding_dim % pq_m == 0.
```

If the C++ binary crashes with an illegal instruction:

```text
The current build uses AVX-512. Check lscpu for AVX512 support or rebuild after removing AVX-512 flags.
```

If FAISS GPU indexing fails:

```text
Use CPU indexing by removing --use-gpu. Keep --faiss-threads high for faster construction.
```

If QPS is missing:

```text
Pass --run-log to evaluate_beir_emvb.py and confirm the log contains "Average Elapsed Time per query".
```

If the sweep seems to skip work:

```text
The sweep reuses existing TSV and JSON files. Delete the specific result and metric files to force rerun.
```
