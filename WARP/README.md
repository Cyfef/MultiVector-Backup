# WARP

### Environment

```bash
conda env create -f conda_env_cpu.yml
conda activate warp

conda install "mkl<2025" packaging -y
conda install -c conda-forge gcc_linux-64 gxx_linux-64
conda install -c nvidia cuda-nvcc cuda-cudart-dev
conda install -c conda-forge libxcrypt libxcrypt-dev

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

export CUDAHOSTCXX=$CXX

export PYTHON_BIN="$(which python)"
export PYTHONPATH=/data/ali/WARP:${PYTHONPATH:-}
export TORCH_EXTENSIONS_DIR=/data1/chenyifeng/tmp/torch-ext
export HF_HOME=/data1/chenyifeng/tmp/hf-cache
export TRANSFORMERS_CACHE=/data1/chenyifeng/tmp/hf-cache
```

For a GPU build environment, use `conda_env.yml` instead of `conda_env_cpu.yml`.


### Run

1. Build a WARP Index

```bash
cd WARP

export DATASET=scidocs
export EMBEDDING_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/embeddings/colbert
export INDEX_ROOT=/data1/chenyifeng/MultiVector-Backup/WARP/indexes
export INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2.rebuild
export CXX=/usr/bin/g++-9   

python utility/index_from_embeddings.py \
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


2. Run WARP Search

Use `utility/sweep_ncells.py` to run search and evaluate metrics. This command runs one setting and writes both:

1. a metrics CSV, and
2. a ranked TSV run file with `query-id`, `corpus-id`, `rank`, `score`.

```bash
cd WARP

export CPATH=/usr/include:$CPATH
export CFLAGS="-I/usr/include"
export CXXFLAGS="-I/usr/include"

export DATASET=scidocs
export DATASET_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/data/scidocs/beir
export EMBEDDING_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/embeddings/colbert
export INDEX_ROOT=/data1/chenyifeng/MultiVector-Backup/WARP/indexes
export INDEX_NAME=beir-scidocs.split=test.precomputed=colbert.nbits=2.rebuild
export OUT_DIR=/data1/chenyifeng/MultiVector-Backup/WARP/my_runs/scidocs-colbert
mkdir -p "$OUT_DIR"

python utility/sweep_ncells.py \
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