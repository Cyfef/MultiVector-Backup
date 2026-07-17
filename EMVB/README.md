# EMVB

### Environment

```bash
conda create -n EMVB python=3.10 -y
conda activate EMVB
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

Build The C++ Search Binary:

```
cd EMVB
mkdir -p build
cd build
cmake -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DCMAKE_CXX_FLAGS="-march=native" ..
make -j"$(nproc)"
```

Expected result: an executable file named `perf_emvb` in `/build` .


### Run

1. Build An EMVB Index

```bash
cd EMVB
mkdir -p logs

python -u prepare_emvb_data.py \
  --base-embeddings /data1/chenyifeng/scidocs/colbert/corpus_points.npy \
  --base-offsets /data1/chenyifeng/scidocs/colbert/corpus_offsets.npy \
  --query-embeddings /data1/chenyifeng/scidocs/colbert/query_points.npy \
  --query-offsets /data1/chenyifeng/scidocs/colbert/query_offsets.npy \
  --output-dir /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  --nlist 4096 \
  --pq-m 32 \
  --add-batch-size 200000 \
  --faiss-threads 48 \
  2>&1 | tee /data1/chenyifeng/MultiVector-Backup/EMVB/logs/prepare_scidocs_colbert_simple.log
```

2. Run One Search Manually

This command runs the base `k=10` SciDocs search using one thread.

```bash
cd /data/ali/EMVB
mkdir -p /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32

OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
BLIS_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 \
/data1/chenyifeng/MultiVector-Backup/EMVB/build/perf_emvb \
  -k 10 \
  -nprobe 4 \
  -thresh 0.4 \
  -out-second-stage 512 \
  -thresh-query 0.5 \
  -n-doc-to-score 4000 \
  -queries-id-file /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/queries_id.txt \
  -alldoclens-path /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/alldoclens.npy \
  -index-dir-path /data1/chenyifeng/MultiVector-Backup/EMVB/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32 \
  -out-file /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/results_k10.tsv \
  > /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10.log 2>&1
```


3. Evaluate One Search

Evaluate the manual `k=10` SciDocs run:

```bash
cd EMVB

python evaluate_beir_emvb.py \
  --dataset-dir /data1/chenyifeng/scidocs/beir \
  --split test \
  --run /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/results_k10.tsv \
  --k-values 10 \
  --run-log /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/run_k10.log \
  --search-threads 1 \
  --output-json /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_k10.json \
  --output-csv /data1/chenyifeng/MultiVector-Backup/EMVB/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32/metrics_k10.csv
```