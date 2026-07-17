# WARP Baseline Runbook

This directory contains the WARP baseline code used to run retrieval over precomputed ColBERT-style multi-vector embeddings. The workflow is:

1. Set up Python.
2. Check that the BEIR-style dataset files and packed embedding files have the required shape.
3. Build a WARP index from the packed document embeddings.
4. Run WARP search with packed query embeddings.
5. Evaluate the run against `qrels`.
6. Verify the installation by reproducing the saved SciDocs simple ColBERT result.

The commands below assume the student works from:

```bash
cd /data/ali/WARP
```

## 1. Directory Layout

After preparation, `/data/ali/WARP` should contain:

```text
/data/ali/WARP/
  README.md
  conda_env.yml
  conda_env_cpu.yml
  executor.py
  utils.py
  warp/
  utility/
  .deps/
  indexes/
    beir-scidocs.split=test.precomputed=colbert.nbits=2/
  results/
    scidocs.split=test.hyperparam_sweep.csv
    scidocs/
      split=test.baseline=...queries=1000.tsv
  data/
    scidocs -> /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs
  embeddings/
    scidocs-colbert -> /data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert
```

The `data/` and `embeddings/` entries are symlinks to the existing large files. They are intentionally not duplicated.

## 2. Environment Setup

### Option A: use the existing prepared environment

This machine already has a working CPU environment:

```bash
cd /data/ali/WARP

export PYTHON_BIN=/data/ali/warp-baseline-cpu/bin/python
export LD_LIBRARY_PATH=/data/ali/warp-baseline-cpu/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/data/ali/WARP:${PYTHONPATH:-}
export TORCH_EXTENSIONS_DIR=/tmp/torch-ext
export HF_HOME=/tmp/hf-cache
export TRANSFORMERS_CACHE=/tmp/hf-cache

export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

Check that Python can import the required packages:

```bash
$PYTHON_BIN - <<'PY'
import numpy, torch, faiss, transformers, ujson
print("python ok")
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
PY
```

`cuda available` can be `False`. WARP search is CPU-oriented. Building an index is faster with GPU FAISS, but the documented SciDocs verification works on CPU.

### Option B: create a new conda environment

Use this if the prepared environment is not available:

```bash
cd /data/ali/WARP

conda env create -f conda_env_cpu.yml
conda activate warp
export PYTHON_BIN="$(which python)"
export PYTHONPATH=/data/ali/WARP:${PYTHONPATH:-}
export TORCH_EXTENSIONS_DIR=/tmp/torch-ext
export HF_HOME=/tmp/hf-cache
export TRANSFORMERS_CACHE=/tmp/hf-cache
```

For a GPU build environment, use `conda_env.yml` instead of `conda_env_cpu.yml`.

## 3. Input Data Format

WARP in this baseline does not tokenize raw text during indexing or search. It expects precomputed multi-vector embeddings.

Each experiment needs two inputs:

1. A BEIR-style dataset directory containing text IDs and relevance labels.
2. A packed embedding directory containing NumPy arrays for document/query token embeddings.

### 3.1 BEIR-style dataset directory

For SciDocs simple ColBERT:

```bash
DATASET_DIR=/data/ali/WARP/data/scidocs
```

Required files:

```text
$DATASET_DIR/corpus.jsonl
$DATASET_DIR/queries.jsonl
$DATASET_DIR/qrels/test.tsv
```

`corpus.jsonl` has one JSON object per document. Required field:

```json
{"_id": "document id", "title": "optional title", "text": "document text"}
```

`queries.jsonl` has one JSON object per query. Required fields:

```json
{"_id": "query id", "text": "query text"}
```

`qrels/test.tsv` is tab-separated with this header:

```text
query-id    corpus-id    score
```

The SciDocs simple ColBERT dataset currently has:

```text
corpus.jsonl:  25,657 documents
queries.jsonl: 1,000 queries
qrels/test.tsv: 29,928 relevance rows over 1,000 queries
```

### 3.2 Packed embedding directory

For SciDocs simple ColBERT:

```bash
EMBEDDING_DIR=/data/ali/WARP/embeddings/scidocs-colbert
```

Required files:

```text
$EMBEDDING_DIR/corpus_points.npy
$EMBEDDING_DIR/corpus_offsets.npy
$EMBEDDING_DIR/query_points.npy
$EMBEDDING_DIR/query_offsets.npy
```

Optional ID mapping files:

```text
$EMBEDDING_DIR/corpus_ids.json
$EMBEDDING_DIR/query_ids.json
```

The embedding arrays must have these shapes:

```text
corpus_points.npy   shape = [total_document_tokens, embedding_dim], dtype float32
corpus_offsets.npy  shape = [num_documents + 1], dtype int32 or int64
query_points.npy    shape = [total_query_tokens, embedding_dim], dtype float32
query_offsets.npy   shape = [num_queries + 1], dtype int32 or int64
```

Offsets define which token vectors belong to each document or query:

```text
document i vectors = corpus_points[corpus_offsets[i] : corpus_offsets[i + 1]]
query j vectors    = query_points[query_offsets[j] : query_offsets[j + 1]]
```

The first offset must be `0`. The final offset must equal the number of rows in the matching points array.

For SciDocs simple ColBERT the exact packed shapes are:

```text
corpus_points.npy   (4,820,738, 128) float32
corpus_offsets.npy  (25,658,) int32, last value 4,820,738
query_points.npy    (48,000, 128) float32
query_offsets.npy   (1,001,) int32, last value 48,000
```

Because this embedding directory does not include `corpus_ids.json` or `query_ids.json`, the code assumes:

```text
document embedding order == corpus.jsonl order
query embedding order    == qrels/test.tsv query order
```

Check the shapes yourself:

```bash
$PYTHON_BIN - <<'PY'
import numpy as np
from pathlib import Path

base = Path("/data/ali/WARP/embeddings/scidocs-colbert")
for name in ["corpus_points.npy", "corpus_offsets.npy", "query_points.npy", "query_offsets.npy"]:
    arr = np.load(base / name, mmap_mode="r")
    print(name, arr.shape, arr.dtype, "last=" + str(int(arr[-1])) if "offsets" in name else "")
PY
```

## 4. Build a WARP Index

The staged SciDocs index is already available at:

```bash
INDEX_ROOT=/data/ali/WARP/indexes
INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2
```

To build the same index from the packed embeddings, choose a new output name or delete the old index first. The index builder refuses to overwrite a non-empty index directory.

```bash
cd /data/ali/WARP

export DATASET=scidocs
export EMBEDDING_DIR=/data/ali/WARP/embeddings/scidocs-colbert
export INDEX_ROOT=/data/ali/WARP/indexes
export INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2.rebuild

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
```

Important indexing parameters:

```text
--nbits 2                    residual quantization bits used by this baseline
--max-partitions 32768       upper bound on k-means centroids
--sample-per-centroid 64     training sample budget per centroid
--max-sample-embeddings 2000000
--chunk-size 50000           documents per saved index chunk
```

Expected output files inside `$INDEX_ROOT/$INDEX_NAME/`:

```text
metadata.json
plan.json
centroids.pt
avg_residual.pt
buckets.pt
ivf.pid.pt
0.codes.pt
0.residuals.pt
0.metadata.json
doclens.0.json
```

For SciDocs simple ColBERT, the built index is about `193M`.

## 5. Run WARP Search

Use `utility/sweep_ncells.py` to run search and evaluate metrics. This command runs one setting and writes both:

1. a metrics CSV, and
2. a ranked TSV run file with `query-id`, `corpus-id`, `rank`, `score`.

```bash
cd /data/ali/WARP

export DATASET=scidocs
export DATASET_DIR=/data/ali/WARP/data/scidocs
export EMBEDDING_DIR=/data/ali/WARP/embeddings/scidocs-colbert
export INDEX_ROOT=/data/ali/WARP/indexes
export INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2
export OUT_DIR=/data/ali/WARP/my_runs/scidocs-colbert
mkdir -p "$OUT_DIR"

$PYTHON_BIN utility/sweep_ncells.py \
  --dataset "$DATASET" \
  --dataset-dir "$DATASET_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-root "$INDEX_ROOT" \
  --index "$INDEX_NAME" \
  --split test \
  --nbits 2 \
  --k 100 \
  --baseline warp_precomputed_colbert_nbits2_thr0.45_ndocs1024 \
  --centroid-score-threshold 0.45 \
  --ndocs 1024 \
  --ncells 1 \
  --output-csv "$OUT_DIR/scidocs_colbert_verify.csv" \
  --tsv-output "$OUT_DIR/scidocs_colbert_verify.tsv"
```

Expected terminal line:

```text
ncells=1 NDCG@10=0.15076 NDCG@100=0.21951 Recall@100=0.36043
```

The elapsed time and QPS can change by machine load. The quality metrics should match, or differ only by tiny floating-point noise.

To sweep multiple `ncells` values and all three candidate settings used in the saved baseline:

```bash
$PYTHON_BIN utility/sweep_ncells.py \
  --dataset scidocs \
  --dataset-dir /data/ali/WARP/data/scidocs \
  --embedding-dir /data/ali/WARP/embeddings/scidocs-colbert \
  --index-root /data/ali/WARP/indexes \
  --index beir-scidocs.split=test.precomputed=colbert.nbits=2 \
  --split test \
  --nbits 2 \
  --k 100 \
  --baseline warp_precomputed_colbert_nbits2_thr0.5_ndocs256 \
  --centroid-score-threshold 0.5 \
  --ndocs 256 \
  --ncells 1 2 4 6 8 10 12 14 16 18 20 \
  --output-csv /data/ali/WARP/my_runs/scidocs_colbert_thr0.5_ndocs256.csv
```

Repeat with:

```text
baseline                                      threshold  ndocs
warp_precomputed_colbert_nbits2_thr0.45_ndocs1024  0.45       1024
warp_precomputed_colbert_nbits2_thr0.4_ndocs4096   0.4        4096
```

## 6. Evaluate a Saved TSV Run Against Qrels

The search command above already computes metrics from qrels. To explicitly evaluate a TSV run file afterward, use:

```bash
$PYTHON_BIN utility/evaluate_run_tsv.py \
  --run-tsv /data/ali/WARP/my_runs/scidocs-colbert/scidocs_colbert_verify.tsv \
  --dataset-dir /data/ali/WARP/data/scidocs \
  --split test \
  --metrics-k 10 100 \
  --output-csv /data/ali/WARP/my_runs/scidocs-colbert/scidocs_colbert_verify.eval.csv
```

Expected JSON output for the recommended SciDocs simple ColBERT setting:

```json
{
  "map_10": 0.08635,
  "map_100": 0.10212,
  "mrr_10": 0.27125,
  "mrr_100": 0.28302,
  "ndcg_10": 0.15076,
  "ndcg_100": 0.21951,
  "num_qrels_queries": 1000,
  "num_run_queries": 1000,
  "p_10": 0.0776,
  "p_100": 0.01778,
  "recall_10": 0.15723,
  "recall_100": 0.36043
}
```

## 7. Saved SciDocs Simple ColBERT Results

The saved result files are in:

```text
/data/ali/WARP/results/scidocs.split=test.hyperparam_sweep.csv
/data/ali/WARP/results/scidocs/
```

The recommended verification row is:

```text
dataset: scidocs
split: test
baseline: warp_precomputed_colbert_nbits2_thr0.45_ndocs1024
index_name: beir-scidocs.split=test.precomputed=colbert.nbits=2
k: 100
ncells: 1
centroid_score_threshold: 0.45
ndocs: 1024
num_queries: 1000
ndcg_10: 0.15076
map_10: 0.08635
recall_10: 0.15723
p_10: 0.0776
ndcg_100: 0.21951
map_100: 0.10212
recall_100: 0.36043
p_100: 0.01778
mrr_10: 0.27125
mrr_100: 0.28302
```

Evaluate the saved TSV directly:

```bash
$PYTHON_BIN utility/evaluate_run_tsv.py \
  --run-tsv "/data/ali/WARP/results/scidocs/split=test.baseline=warp_precomputed_colbert_nbits2_thr0.45_ndocs1024.ncells=1.thr=0.45.ndocs=1024.k=100.queries=1000.tsv" \
  --dataset-dir /data/ali/WARP/data/scidocs \
  --split test \
  --metrics-k 10 100
```

This should print the same metrics shown above.

## 8. Running Another Dataset

To run another dataset, prepare the same two inputs:

```text
<dataset_dir>/corpus.jsonl
<dataset_dir>/queries.jsonl
<dataset_dir>/qrels/<split>.tsv

<embedding_dir>/corpus_points.npy
<embedding_dir>/corpus_offsets.npy
<embedding_dir>/query_points.npy
<embedding_dir>/query_offsets.npy
```

If the embedding order is different from the JSONL/qrels order, also provide:

```text
<embedding_dir>/corpus_ids.json
<embedding_dir>/query_ids.json
```

Then use the same commands with different variables:

```bash
export DATASET=<dataset_label>
export SPLIT=test
export DATASET_DIR=<path_to_dataset_dir>
export EMBEDDING_DIR=<path_to_embedding_dir>
export INDEX_ROOT=/data/ali/WARP/indexes
export INDEX_NAME=beir-${DATASET}.split=${SPLIT}.precomputed=colbert.nbits=2
```

For MS MARCO, this code defaults to `dev` split when the dataset label is `msmarco` or `msmarco_small`; otherwise it defaults to `test`. Passing `--split` explicitly is safest.

## 9. Common Problems

`FileExistsError: Index directory already exists and is not empty`

Use a new `--index-name`, or intentionally remove the old index directory before rebuilding.

`Query count mismatch`

The number of queries in `qrels/<split>.tsv` does not match `query_offsets.npy`, and there is no `query_ids.json` to map IDs. Add `query_ids.json` or regenerate query embeddings in qrels order.

`Corpus count mismatch`

The number of corpus rows does not match `corpus_offsets.npy`, or `corpus_ids.json` length does not match. Fix the packed embedding export.

`No CUDA runtime is found`

This warning is acceptable for CPU search. It does not prevent the documented SciDocs verification from running.

Torch extension rebuilds are slow on the first run

The first search can compile C++ extensions into `$TORCH_EXTENSIONS_DIR`. Later runs reuse the compiled extensions.
