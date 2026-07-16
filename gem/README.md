# GEM

### Environment

##### System Dependencies

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

##### Python Environment

```bash
conda create -n GEM python=3.11 -y
conda activate GEM

pip install numpy==2.4.4 tqdm==4.67.3 # faiss-cpu==1.14.3

# if nvidia-cuda-toolkit is ok
pip install faiss-gpu
```

##### Build The GEM Binary

```bash
cd gem
cmake -S hnswlib -B hnswlib/build
cmake --build hnswlib/build --target example_vecset_search_gem -j 8
```

### Run

This section rebuilds everything from the raw SciDocs `.npy` files.

### 5.1 Define Paths

```bash
export REPO_ROOT=/data1/chenyifeng/MultiVector-Backup/gem
export PYTHON_BIN=/data1/chenyifeng/miniconda3/envs/GEM/bin/python3
export DATASET_NAME=scidocs
export DATASET_STEM=scidocs
export RAW_SOURCE_DIR=/data1/chenyifeng/scidocs/colbert
export RAW_TARGET_DIR=/data1/chenyifeng/scidocs-colbert
export OUTPUT_ROOT=/data1/chenyifeng/scidocs-gem-data
export INDEX_ROOT=/data1/chenyifeng/scidocs-gem-index
export RESULTS_DIR=/data1/chenyifeng/MultiVector-Backup/gem/results/scidocs_full_run_$(date +%Y%m%d_%H%M%S)

export QUERIES_FILE=/data1/chenyifeng/scidocs/beir/queries.jsonl
export CORPUS_FILE=/data1/chenyifeng/scidocs/beir/corpus.jsonl
export QRELS_FILE=/data1/chenyifeng/scidocs/beir/qrels/test.tsv
export QRELS_QUERY_ORDER=queries-jsonl
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
