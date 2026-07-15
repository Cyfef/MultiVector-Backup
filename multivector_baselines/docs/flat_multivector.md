# Flat Multi-Vector Datasets: Step-by-Step Run Guide

This guide explains how to run the existing Plaid, Dessert, MUVERA, and IGP pipelines on a dataset stored in the following flat multi-vector format:

- one big document embedding matrix
- one 1D document length array
- one big query embedding matrix
- one 1D query length array
- optional ground-truth file

This workflow does **not** modify the model code. It only adds a separate preparation layer and helper scripts that convert the flat arrays into the file layout the existing code already expects.

## Validated Reference Run

Use this section as the canonical procedure. The later sections remain useful background, but the commands and metrics below are the fully checked reference run.

Validated reference:

- dataset: `scidocs-large-multi-flat-test`
- date: `2026-05-21`
- repo: `/home/ali/plaid-index`
- validated Python path on this machine: `/data/ali/plaid-index/bin/python`
- qrels: `/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv`
- manifest: `/data/ali/Dataset/multi-vector-retrieval/FlatData/scidocs-large-multi-flat-test/manifest.json`
- summary CSV: `/home/ali/plaid-index/results/flat_validation/scidocs-large-multi-flat-test/scidocs-large-multi-flat-test-flat-eval-summary.csv`

Exact input source paths used in the validated run:

- source dataset root: `/data/ali/scidocs-large-multi`
- document embeddings: `/data/ali/scidocs-large-multi/full_multi_embeddings_scidocs-large.npy`
- document lengths: `/data/ali/scidocs-large-multi/full_multi_chunk_num_scidocs-large.npy`
- query embeddings: `/data/ali/scidocs-large-multi/full_multi_embeddings_scidocs-large_query.npy`
- query lengths: `/data/ali/scidocs-large-multi/full_multi_chunk_num_scidocs-large_query.npy`
- query id map: `/data/ali/scidocs-modern-colbert/query_ids.json`
- corpus id map: `/data/ali/scidocs-modern-colbert/corpus_ids.json`
- qrels root: `/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs`
- qrels file: `/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv`

Exact runtime output paths created by the validated run:

- flat prepared dataset: `/data/ali/Dataset/multi-vector-retrieval/FlatData/scidocs-large-multi-flat-test`
- raw runtime dataset: `/data/ali/Dataset/multi-vector-retrieval/RawData/scidocs-large-multi-flat-test`
- canonical embedding files: `/data/ali/Dataset/multi-vector-retrieval/Embedding/scidocs-large-multi-flat-test`
- Plaid index: `/data/ali/Dataset/multi-vector-retrieval/Index/scidocs-large-multi-flat-test/plaid`
- answer TSVs: `/data/ali/Dataset/multi-vector-retrieval/Result/answer`
- performance JSONs: `/data/ali/Dataset/multi-vector-retrieval/Result/performance`

Validated environment on this machine:

```bash
export REPO=/home/ali/plaid-index
export PYTHON_BIN=/data/ali/plaid-index/bin/python
export PYTHONPATH=/home/ali/plaid-index:/home/ali/plaid-index/baseline/ColBERT
export COLBERT_CHECKPOINT_PATH=/data/ali/colbertv2.0
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/data/ali/plaid-index/lib:/home/ali/miniconda3/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=3
cd /home/ali/plaid-index
```

Recommended user-owned environment variables for a new machine:

```bash
export REPO=/home/ali/plaid-index
export ENV_ROOT=/path/to/your/conda/env
export PYTHON_BIN=$ENV_ROOT/bin/python
export PYTHONPATH=/home/ali/plaid-index:/home/ali/plaid-index/baseline/ColBERT
export COLBERT_CHECKPOINT_PATH=/data/ali/colbertv2.0
export CUDA_HOME=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}
cd /home/ali/plaid-index
```

From-scratch environment creation command for an external user:

```bash
conda create -p /path/to/your/conda/env python=3.10.18 pip setuptools wheel -y
conda activate /path/to/your/conda/env
```

Validated environment path on this machine:

```bash
conda activate /data/ali/plaid-index
```

Python dependency files in this repo:

- pip requirements: `/home/ali/plaid-index/requirements.txt`
- conda environment file: `/home/ali/plaid-index/environment.yml`

Important note about `environment.yml`:

- it contains a stale `prefix:` from another machine
- if you use it directly, remove the final `prefix:` line first
- the simpler reproducible path is to create the Python 3.10 env manually and then install `requirements.txt`

Recommended Python package install for an external user:

```bash
conda activate /path/to/your/conda/env
pip install -r /home/ali/plaid-index/requirements.txt
```

Core Python packages that are required by the validated run and are pinned in `requirements.txt`:

- `torch==2.7.1`
- `torchaudio==2.7.1`
- `torchvision==0.22.1`
- `faiss-gpu-cu12==1.11.0`
- `transformers==4.52.4`
- `numpy==1.26.4`
- `pandas==2.3.0`
- `tqdm==4.67.1`
- `ujson==5.10.0`
- `pybind11==2.13.6`
- `cmake==3.23.3`
- `huggingface-hub==0.33.1`
- `requests==2.32.4`
- `PyYAML==6.0.2`

Native prerequisites for `MUVERA` and `IGP`:

```bash
sudo apt update
sudo apt install -y build-essential git libeigen3-dev libspdlog-dev g++-9 libopenblas-dev zlib1g-dev cmake
```

GPU runtime assumptions from the validated run:

- NVIDIA GPU available
- CUDA toolkit path: `/usr/local/cuda-12.8`
- GPU-enabled Python wheels installed from `requirements.txt`

Validated local Parlay install:

```bash
git clone https://github.com/cmuparlay/parlaylib.git /data/ali/parlaylib
cmake -S /data/ali/parlaylib -B /data/ali/parlaylib/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/data/ali/parlay-install
cmake --build /data/ali/parlaylib/build -j
cmake --install /data/ali/parlaylib/build
```

If `MUVERA` or `IGP` need a clean rebuild against the correct Python:

```bash
cd /home/ali/plaid-index
rm -rf build
cmake -S /home/ali/plaid-index -B /home/ali/plaid-index/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DParlay_DIR=/data/ali/parlay-install/share/parlay/cmake \
  -DPython_EXECUTABLE=$ENV_ROOT/bin/python \
  -DPython_INCLUDE_DIR=$ENV_ROOT/include/python3.10 \
  -DPython_LIBRARY=$ENV_ROOT/lib/libpython3.10.so
cmake --build /home/ali/plaid-index/build -j
```

Validated rebuild path on this machine used:

```bash
cmake -S /home/ali/plaid-index -B /home/ali/plaid-index/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DParlay_DIR=/data/ali/parlay-install/share/parlay/cmake \
  -DPython_EXECUTABLE=/data/ali/plaid-index/bin/python \
  -DPython_INCLUDE_DIR=/data/ali/plaid-index/include/python3.10 \
  -DPython_LIBRARY=/data/ali/plaid-index/lib/libpython3.10.so
```

Import check:

```bash
LD_LIBRARY_PATH=$ENV_ROOT/lib:${LD_LIBRARY_PATH:-} \
$ENV_ROOT/bin/python -c "import MUVERA, IGP; print('imports_ok')"
```

Validated import check on this machine:

```bash
LD_LIBRARY_PATH=/data/ali/plaid-index/lib:/home/ali/miniconda3/lib \
/data/ali/plaid-index/bin/python -c "import MUVERA, IGP; print('imports_ok')"
```

Exact validated SciDocs preparation command:

```bash
$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username ali \
  --dataset scidocs-large-multi-flat-test \
  --doc-embeddings /data/ali/scidocs-large-multi/full_multi_embeddings_scidocs-large.npy \
  --doc-lens /data/ali/scidocs-large-multi/full_multi_chunk_num_scidocs-large.npy \
  --query-embeddings /data/ali/scidocs-large-multi/full_multi_embeddings_scidocs-large_query.npy \
  --query-lens /data/ali/scidocs-large-multi/full_multi_chunk_num_scidocs-large_query.npy \
  --qrels-tsv /data1/liuyaoyang/Papers/ACFDE/datasets/scidocs/qrels/test.tsv \
  --query-ids-json /data/ali/scidocs-modern-colbert/query_ids.json \
  --corpus-ids-json /data/ali/scidocs-modern-colbert/corpus_ids.json \
  --force
```

Exact validated method commands:

```bash
$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/build_plaid_from_flat_dataset.py \
  --username ali \
  --dataset scidocs-large-multi-flat-test

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_plaid.py \
  --username ali \
  --dataset scidocs-large-multi-flat-test

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_dessert.py \
  --username ali \
  --dataset scidocs-large-multi-flat-test

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_muvera.py \
  --username ali \
  --dataset_name scidocs-large-multi-flat-test

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_igp.py \
  --username ali \
  --dataset_name scidocs-large-multi-flat-test
```

Exact validated evaluation command:

```bash
$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/eval_flat_groundtruth.py \
  --username ali \
  --dataset scidocs-large-multi-flat-test \
  --output-csv /home/ali/plaid-index/results/flat_validation/scidocs-large-multi-flat-test/scidocs-large-multi-flat-test-flat-eval-summary.csv
```

Validated logs:

- Plaid: `/home/ali/plaid-index/results/flat_validation/scidocs-large-multi-flat-test/plaid_eval_20260521.log`
- Dessert: `/home/ali/plaid-index/results/flat_validation/scidocs-large-multi-flat-test/dessert_20260521.log`
- MUVERA: `/home/ali/plaid-index/results/flat_validation/scidocs-large-multi-flat-test/muvera_20260521_retry.log`
- IGP: `/home/ali/plaid-index/results/flat_validation/scidocs-large-multi-flat-test/igp_20260521_retry.log`

Validated build times:

| Method | Build metric |
| --- | --- |
| Plaid | `build_index_time = 38.273 s`, `encode_passage_time = 4.840 s` |
| Dessert | `build_index_time_except_centroid = 324.785 s` |
| MUVERA | `build_index_time = 152.100 s` |
| IGP | `build_index_time = 46.857 s` |

Best completed `top10` rows from the validated summary CSV:

| Method | Config | QPS | Avg query ms | Recall@10 | MRR@10 | Success@10 | NDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plaid | `ndocs=10000, ncells=64, centroid_score_threshold=0.50` | `2.491` | `401.434` | `0.2052` | `0.3345` | `0.585` | `0.1943` |
| Dessert | `n_table=64, initial_filter_k=1000, nprobe_query=2` | `135.814` | `7.363` | `0.0577` | `0.1811` | `0.258` | `0.2251` |
| MUVERA | `k_sim=3, d_proj=32, r_reps=8, n_candidate=100` | `231.102` | `4.327` | `0.0417` | `0.0971` | `0.160` | `0.0453` |
| IGP | `n_centroid=1024, n_bit=2, nprobe=32, probe_topk=1000` | `4.201` | `238.037` | `0.0694` | `0.1492` | `0.259` | `0.0712` |

Best completed `top100` rows from the validated summary CSV:

| Method | Config | QPS | Avg query ms | Recall@100 | MRR@100 | Success@100 | NDCG@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plaid | `ndocs=10000, ncells=64, centroid_score_threshold=0.50` | `2.369` | `422.141` | `0.4495` | `0.3468` | `0.864` | `0.2777` |
| Dessert | `n_table=64, initial_filter_k=2000, nprobe_query=4` | `104.483` | `9.571` | `0.1802` | `0.1809` | `0.552` | `0.6758` |
| MUVERA | `k_sim=3, d_proj=32, r_reps=8, n_candidate=1000` | `9.538` | `104.842` | `0.2001` | `0.2141` | `0.551` | `0.1410` |
| IGP | `n_centroid=1024, n_bit=2, nprobe=32, probe_topk=5000` | `1.142` | `875.735` | `0.3856` | `0.2871` | `0.794` | `0.2307` |

## Hyperparameters You Can Change

An external user can change method hyperparameters directly in the original method scripts. The helper flat-data scripts do not define the method sweeps; they only prepare data and call the original code.

The exact files to edit are:

- Plaid: `/home/ali/plaid-index/script/evaluation/eval_plaid.py`
- Dessert: `/home/ali/plaid-index/script/evaluation/eval_dessert.py`
- MUVERA: `/home/ali/plaid-index/script/evaluation/eval_muvera.py`
- IGP: `/home/ali/plaid-index/script/evaluation/eval_igp.py`

The main configuration blocks live in each script under `config_l["local"]`.

### Plaid hyperparameters

File:

- `/home/ali/plaid-index/script/evaluation/eval_plaid.py`

Main tunable fields:

- `topk_l`
- `grid_search`
- `grid_search_para[10]["ndocs"]`
- `grid_search_para[10]["ncells"]`
- `grid_search_para[10]["centroid_score_threshold"]`
- `grid_search_para[10]["n_thread"]`
- `grid_search_para[100]["ndocs"]`
- `grid_search_para[100]["ncells"]`
- `grid_search_para[100]["centroid_score_threshold"]`
- `grid_search_para[100]["n_thread"]`

Meaning:

- `ndocs`: how many candidate documents are refined
- `ncells`: how many IVF cells are searched
- `centroid_score_threshold`: centroid pruning threshold
- `n_thread`: CPU threads used in retrieval
- `topk_l`: which result depths are run

Validated run values:

- `topk_l = [10, 100]`
- for `top10` and `top100`:
  - `ndocs = [10, 50, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]`
  - `ncells = [64]`
  - `centroid_score_threshold = [0.5]`
  - `n_thread = [1]`

### Dessert hyperparameters

File:

- `/home/ali/plaid-index/script/evaluation/eval_dessert.py`

Main tunable fields:

- `topk_l`
- `grid_search`
- `grid_search_para[10]["n_table"]`
- `grid_search_para[10]["initial_filter_k"]`
- `grid_search_para[10]["nprobe_query"]`
- `grid_search_para[10]["remove_centroid_dupes"]`
- `grid_search_para[10]["n_thread"]`
- `grid_search_para[100]["n_table"]`
- `grid_search_para[100]["initial_filter_k"]`
- `grid_search_para[100]["nprobe_query"]`
- `grid_search_para[100]["remove_centroid_dupes"]`
- `grid_search_para[100]["n_thread"]`

Meaning:

- `n_table`: number of Dessert hash tables for the built index
- `initial_filter_k`: size of the candidate set before reranking
- `nprobe_query`: query-time probing breadth
- `remove_centroid_dupes`: whether duplicate centroid hits are removed
- `n_thread`: CPU threads used in retrieval

Validated run values:

- `topk_l = [10, 100]`
- for `top10`:
  - `n_table = [64]`
  - `initial_filter_k = [10, 50, 100, 500, 1000]`
  - `nprobe_query = [2]`
  - `remove_centroid_dupes = [True]`
  - `n_thread = [1]`
- for `top100`:
  - `n_table = [64]`
  - `initial_filter_k = [100, 200, 500, 1000, 2000, 5000, 8000, 10000]`
  - `nprobe_query = [2, 4]`
  - `remove_centroid_dupes = [True]`
  - `n_thread = [1]`

### MUVERA hyperparameters

File:

- `/home/ali/plaid-index/script/evaluation/eval_muvera.py`

Main tunable fields:

- `topk_l`
- `is_debug`
- `build_index_parameter_l`
- `grid_search`
- `grid_search_para[10]["n_candidate"]`
- `grid_search_para[100]["n_candidate"]`

Build-time tunable fields inside `build_index_parameter_l[0]`:

- `k_sim`
- `d_proj`
- `r_reps`
- `R`
- `L`
- `alpha`
- `n_centroid_per_subspace`
- `dim_per_subspace`

Meaning:

- `k_sim`: number of sign partitions per repetition
- `d_proj`: projected dimension per repetition
- `r_reps`: number of repetitions
- `R`, `L`, `alpha`: graph/index construction hyperparameters
- `n_centroid_per_subspace`: PQ centroids per subspace
- `dim_per_subspace`: PQ subspace width
- `n_candidate`: number of candidates searched at retrieval time

Validated run values:

- `topk_l = [10, 100]`
- `is_debug = True`
- build config:
  - `k_sim = 3`
  - `d_proj = 32`
  - `r_reps = 8`
  - `R = 20`
  - `L = 50`
  - `alpha = 1.2`
  - `n_centroid_per_subspace = 256`
  - `dim_per_subspace = 16`
- retrieval:
  - `top10 n_candidate = [20, 50, 100]`
  - `top100 n_candidate = [100, 500, 1000]`

### IGP hyperparameters

File:

- `/home/ali/plaid-index/script/evaluation/eval_igp.py`

Main tunable fields:

- `topk_l`
- `is_debug`
- `build_index_parameter_l`
- `grid_search`
- `grid_search_para[10]["nprobe"]`
- `grid_search_para[10]["probe_topk"]`
- `grid_search_para[100]["nprobe"]`
- `grid_search_para[100]["probe_topk"]`

Build-time tunable fields inside `build_index_parameter_l[0]`:

- `n_centroid`
- `n_bit`

Meaning:

- `n_centroid`: number of coarse centroids
- `n_bit`: residual/product quantization bit width used in the IGP build path
- `nprobe`: how many inverted lists are searched
- `probe_topk`: how many candidates are kept after probing

Validated run values:

- `topk_l = [10, 100]`
- `is_debug = True`
- build config:
  - `n_centroid = 1024`
  - `n_bit = 2`
- retrieval:
  - `top10`:
    - `nprobe = [32]`
    - `probe_topk = [10, 50, 100, 500, 1000]`
  - `top100`:
    - `nprobe = [32, 128]`
    - `probe_topk = [100, 500, 1000, 5000]`

### How to change the sweep safely

If an external user wants to change the benchmark settings:

1. edit only the `config_l["local"]` block in the original method script
2. keep the dataset preparation scripts unchanged
3. rerun the method
4. rerun `eval_flat_groundtruth.py`
5. compare the new row in the summary CSV against the validated reference rows above

## 1. What the data format means

The key idea is:

- `doc_embeddings.npy` contains **all document vectors concatenated together**
- `doc_lens.npy` tells you how many vectors belong to each document
- `query_embeddings.npy` contains **all query vectors concatenated together**
- `query_lens.npy` tells you how many vectors belong to each query

Example:

```text
doc_lens = [3, 2, 4]
```

Then:

- document `0` uses rows `0:3`
- document `1` uses rows `3:5`
- document `2` uses rows `5:9`

This is exactly what you described:

- if `chunk_npy[0] == 3`, then the first `3` embeddings belong to document `0`
- then the next `chunk_npy[1]` embeddings belong to document `1`
- and so on

The same rule applies to queries.

## 2. Example: `clerc-med-multi`

For `/data/ali/clerc-med-multi`, the structure is:

- `full_multi_embeddings_clerc_med.npy`
- `full_multi_chunk_num_clerc_med.npy`
- `full_multi_embeddings_clerc_med_query.npy`
- `full_multi_chunk_num_clerc_med_query.npy`
- `clerc-med-multi_groundtruth.ivecs`

Meaning:

- `full_multi_embeddings_clerc_med.npy` is the full document embedding matrix
- `full_multi_chunk_num_clerc_med.npy` is the document length array
- `full_multi_embeddings_clerc_med_query.npy` is the full query embedding matrix
- `full_multi_chunk_num_clerc_med_query.npy` is the query length array
- `clerc-med-multi_groundtruth.ivecs` stores local integer ground truth for each query

## 3. What the preparation step creates

The existing methods in this repo expect a canonical runtime layout under:

```text
/data/<username>/Dataset/multi-vector-retrieval/
```

The new preparation script creates:

```text
/data/<username>/Dataset/multi-vector-retrieval/
├── FlatData/<dataset>/
│   ├── manifest.json
│   ├── doc_embeddings/transformed_embeddings/
│   │   ├── doc_count
│   │   ├── embeddings.0.pt
│   │   ├── embeddings.1.pt
│   │   └── ...
│   ├── query_embeddings/transformed_embeddings/
│   │   └── query_n_vec_length.npy
│   └── query_groundtruth/
│       └── queries.gnd.jsonl
└── RawData/<dataset>/document/
    ├── collection.tsv
    ├── queries.dev.tsv
    ├── queries.gnd.jsonl -> symlink into FlatData
    └── transformed_embeddings -> symlink into FlatData
```

Important:

- this preparation step does **not** rewrite any model code
- it does **not** replace the old pipeline
- it creates a separate dataset entry that the existing methods can consume
- the canonical `Embedding/<dataset>/...` files used by Plaid search, MUVERA, and IGP are materialized by the Plaid build step

## 4. New scripts added for this workflow

These are the new helper files:

- [script/flat_multivector/prepare_flat_multivector_dataset.py](/home/ali/plaid-index/script/flat_multivector/prepare_flat_multivector_dataset.py)
- [script/flat_multivector/build_plaid_from_flat_dataset.py](/home/ali/plaid-index/script/flat_multivector/build_plaid_from_flat_dataset.py)
- [script/flat_multivector/eval_flat_groundtruth.py](/home/ali/plaid-index/script/flat_multivector/eval_flat_groundtruth.py)
- [script/flat_multivector/run_plaid_flat.sh](/home/ali/plaid-index/script/flat_multivector/run_plaid_flat.sh)
- [script/flat_multivector/run_dessert_flat.sh](/home/ali/plaid-index/script/flat_multivector/run_dessert_flat.sh)
- [script/flat_multivector/run_muvera_flat.sh](/home/ali/plaid-index/script/flat_multivector/run_muvera_flat.sh)
- [script/flat_multivector/run_igp_flat.sh](/home/ali/plaid-index/script/flat_multivector/run_igp_flat.sh)

## 5. Environment requirements

You need a Python environment that can already run the repo’s existing methods.

At minimum:

- Python
- `numpy`
- `torch`
- `tqdm`
- `faiss` for MUVERA / some pipelines
- the repo’s existing dependencies

For IGP and MUVERA native modules, the repo already expects C++ compilation support. On a clean machine, install:

- `cmake`
- `g++-9` or another compatible GCC
- `libeigen3-dev`
- `libspdlog-dev`
- `libopenblas-dev`
- `zlib1g-dev`
- `pybind11`

If your repo build already works for MUVERA and IGP, you do not need to change those scripts. The existing `script/evaluation/eval_muvera.py` and `script/evaluation/eval_igp.py` compile and run the native modules themselves.

## 6. Step 1: Prepare a flat dataset

General command:

```bash
python script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username ali \
  --dataset <new-dataset-name> \
  --doc-embeddings <doc-embeddings.npy> \
  --doc-lens <doc-lens.npy> \
  --query-embeddings <query-embeddings.npy> \
  --query-lens <query-lens.npy> \
  --groundtruth-ivecs <groundtruth.ivecs> \
  --force
```

Example for `clerc-med-multi`:

```bash
python script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --doc-embeddings /data/ali/clerc-med-multi/full_multi_embeddings_clerc_med.npy \
  --doc-lens /data/ali/clerc-med-multi/full_multi_chunk_num_clerc_med.npy \
  --query-embeddings /data/ali/clerc-med-multi/full_multi_embeddings_clerc_med_query.npy \
  --query-lens /data/ali/clerc-med-multi/full_multi_chunk_num_clerc_med_query.npy \
  --groundtruth-ivecs /data/ali/clerc-med-multi/clerc-med-multi_groundtruth.ivecs \
  --force
```

Optional:

- use `--max-queries 1000` if you want to prepare only the first 1000 queries
- use `--batch-size-docs 2500` to change shard size

What this step does:

- validates that `sum(doc_lens) == number of document vectors`
- validates that `sum(query_lens) == number of query vectors`
- writes document `.pt` shards in the format expected by existing Plaid indexing code
- writes `query_n_vec_length.npy`
- writes local integer `queries.gnd.jsonl`
- creates `collection.tsv`
- creates `queries.dev.tsv`
- writes a manifest file with all paths

What this step does **not** do:

- it does not create `Embedding/<dataset>/base_embedding/encoding*.npy`
- it does not create the Plaid index

Those are produced by the Plaid build step in the next stage.

## 7. Step 2: Check the manifest

After preparation, inspect:

- `/data/ali/Dataset/multi-vector-retrieval/FlatData/<dataset>/manifest.json`

This file records:

- the original source files
- the prepared runtime paths
- document count
- query count
- embedding dimension

This manifest is what the helper scripts use.

## 8. Step 3: Build and run Plaid

There are two ways to do this.

### Option A: explicit commands

Build the Plaid index:

```bash
python script/flat_multivector/build_plaid_from_flat_dataset.py \
  --username ali \
  --dataset clerc-med-multi-flat
```

Run Plaid retrieval:

```bash
python script/evaluation/eval_plaid.py \
  --username ali \
  --dataset clerc-med-multi-flat
```

Evaluate Plaid results against local ground truth:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --method plaid
```

### Option B: one wrapper

```bash
bash script/flat_multivector/run_plaid_flat.sh ali clerc-med-multi-flat
```

## 9. Step 4: Run Dessert

Important:

- Dessert depends on Plaid artifacts
- run Plaid build first
- Dessert uses the Plaid centroids and codes

Command:

```bash
python script/evaluation/eval_dessert.py \
  --username ali \
  --dataset clerc-med-multi-flat
```

Evaluate Dessert:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --method dessert
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_dessert_flat.sh ali clerc-med-multi-flat
```

## 10. Step 5: Run MUVERA

Command:

```bash
python script/evaluation/eval_muvera.py \
  --username ali \
  --dataset_name clerc-med-multi-flat
```

Important:

- MUVERA reads `Embedding/<dataset>/base_embedding/*.npy`
- those files are created by the Plaid build path
- so if you only ran the preparation script, run Plaid build first

Evaluate MUVERA:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --method MUVERA
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_muvera_flat.sh ali clerc-med-multi-flat
```

## 11. Step 6: Run IGP

Command:

```bash
python script/evaluation/eval_igp.py \
  --username ali \
  --dataset_name clerc-med-multi-flat
```

Important:

- IGP also reads `Embedding/<dataset>/base_embedding/*.npy`
- those files are created by the Plaid build path
- so if you only ran the preparation script, run Plaid build first

Evaluate IGP:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --method IGP
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_igp_flat.sh ali clerc-med-multi-flat
```

## 12. Where each method writes results

All methods keep using the repo’s normal output directories:

- answers:
  `/data/<username>/Dataset/multi-vector-retrieval/Result/answer/`
- performance JSON:
  `/data/<username>/Dataset/multi-vector-retrieval/Result/performance/`

Examples:

- Plaid answer TSV:
  `.../Result/answer/<dataset>-plaid-top10-....tsv`
- Dessert answer TSV:
  `.../Result/answer/<dataset>-dessert-top10-....tsv`
- MUVERA answer TSV:
  `.../Result/answer/<dataset>-MUVERA-top10-....tsv`
- IGP answer TSV:
  `.../Result/answer/<dataset>-IGP-top10-....tsv`

The flat-data evaluator writes a summary CSV to:

- `/data/<username>/Dataset/multi-vector-retrieval/Result/performance/<dataset>-flat-eval-summary.csv`

## 13. How to get the final results

For each method:

1. run the method
2. run `eval_flat_groundtruth.py`
3. open the summary CSV

Example:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --method plaid
```

The CSV includes:

- `qps`
- `total_query_time_ms`
- `average_query_time_ms`
- `recall@10`
- `recall@100`
- `mrr@10`
- `mrr@100`
- `success@10`
- `success@100`
- `ndcg@10`
- `ndcg@100`

It also keeps the method-specific retrieval parameters in the same row.

## 14. How ground truth is interpreted

This workflow assumes local integer ids.

That means:

- query row `i` in the ground truth corresponds to local query id `i`
- document id `j` in the result file refers to local document id `j`

If you use `.ivecs`:

- each row is one query
- the row values are the relevant local document ids
- negative values are ignored

If you do not have `.ivecs`, you can instead prepare local ground truth from BEIR-style `qrels.tsv` plus:

- `query_ids.json`
- `corpus_ids.json`

Then pass:

```bash
--qrels-tsv ...
--query-ids-json ...
--corpus-ids-json ...
```

to the preparation script.

### If you have `qrels.tsv`, use this exact flow

There are two supported `qrels.tsv` cases.

Case 1: the `qrels.tsv` file uses external query ids and corpus ids, such as BEIR-style ids.

Use:

```bash
$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username <username> \
  --dataset <dataset_name> \
  --doc-embeddings <doc_embeddings.npy> \
  --doc-lens <doc_lens.npy> \
  --query-embeddings <query_embeddings.npy> \
  --query-lens <query_lens.npy> \
  --qrels-tsv <qrels.tsv> \
  --query-ids-json <query_ids.json> \
  --corpus-ids-json <corpus_ids.json> \
  --force
```

In this case:

- `query_ids.json` must be a JSON list whose order matches the flat query embedding order
- `corpus_ids.json` must be a JSON list whose order matches the flat document embedding order
- the preparation script maps those external ids into local integer ids before writing `queries.gnd.jsonl`

Case 2: the `qrels.tsv` file already uses local integer ids.

Use:

```bash
$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username <username> \
  --dataset <dataset_name> \
  --doc-embeddings <doc_embeddings.npy> \
  --doc-lens <doc_lens.npy> \
  --query-embeddings <query_embeddings.npy> \
  --query-lens <query_lens.npy> \
  --local-qrels-tsv <qrels.tsv> \
  --force
```

After that, the run order is the same for both cases:

```bash
$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/build_plaid_from_flat_dataset.py \
  --username <username> \
  --dataset <dataset_name>

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_plaid.py \
  --username <username> \
  --dataset <dataset_name>

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_dessert.py \
  --username <username> \
  --dataset <dataset_name>

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_muvera.py \
  --username <username> \
  --dataset_name <dataset_name>

$PYTHON_BIN /home/ali/plaid-index/script/evaluation/eval_igp.py \
  --username <username> \
  --dataset_name <dataset_name>

$PYTHON_BIN /home/ali/plaid-index/script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset_name>
```

## 15. Recommended test on `scidocs-large-multi`

Before running a full dataset, test preparation on a same-format dataset:

```bash
python script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username ali \
  --dataset scidocs-large-multi-flat-test \
  --doc-embeddings /data/ali/scidocs-large-multi/full_multi_embeddings_scidocs-large.npy \
  --doc-lens /data/ali/scidocs-large-multi/full_multi_chunk_num_scidocs-large.npy \
  --query-embeddings /data/ali/scidocs-large-multi/full_multi_embeddings_scidocs-large_query.npy \
  --query-lens /data/ali/scidocs-large-multi/full_multi_chunk_num_scidocs-large_query.npy \
  --max-queries 1000 \
  --force
```

This verifies:

- the flat format is parsed correctly
- document shards are written correctly
- query length metadata is written correctly
- the runtime metadata tree is created correctly

If you also have a ground-truth file for that dataset, include `--groundtruth-ivecs ...`.

## 16. Important limitations

This workflow deliberately does **not** change model code.

Because of that:

- Plaid still uses the repo’s current search settings from `script/evaluation/eval_plaid.py`
- Dessert still uses the repo’s current sweep from `script/evaluation/eval_dessert.py`
- MUVERA still uses the repo’s current config from `script/evaluation/eval_muvera.py`
- IGP still uses the repo’s current config from `script/evaluation/eval_igp.py`

If you want different sweeps or fewer settings, change only the outer run scripts or add new wrappers. Do not edit model internals unless you intentionally want to change the benchmark definition.

## 17. Common failure cases

### `sum(lens) != embedding rows`

Cause:

- the length array does not match the flat embedding matrix

Fix:

- verify the correct `.npy` pair was passed

### Dessert fails because Plaid files do not exist

Cause:

- Plaid build was not run first

Fix:

- run Plaid build before Dessert

### MUVERA or IGP fails during compilation

Cause:

- missing native dependencies

Fix:

- install the C++ dependencies listed above
- verify the repo build path used by the existing scripts is valid

### Evaluation finds no answer files

Cause:

- the retrieval step did not finish
- the dataset name in the evaluator does not match the retrieval output prefix

Fix:

- check `Result/answer`
- use the exact dataset name used at preparation time

## 18. Minimal end-to-end example

```bash
python script/flat_multivector/prepare_flat_multivector_dataset.py \
  --username ali \
  --dataset clerc-med-multi-flat \
  --doc-embeddings /data/ali/clerc-med-multi/full_multi_embeddings_clerc_med.npy \
  --doc-lens /data/ali/clerc-med-multi/full_multi_chunk_num_clerc_med.npy \
  --query-embeddings /data/ali/clerc-med-multi/full_multi_embeddings_clerc_med_query.npy \
  --query-lens /data/ali/clerc-med-multi/full_multi_chunk_num_clerc_med_query.npy \
  --groundtruth-ivecs /data/ali/clerc-med-multi/clerc-med-multi_groundtruth.ivecs \
  --force

bash script/flat_multivector/run_plaid_flat.sh ali clerc-med-multi-flat
bash script/flat_multivector/run_dessert_flat.sh ali clerc-med-multi-flat
python script/evaluation/eval_muvera.py --username ali --dataset_name clerc-med-multi-flat
python script/evaluation/eval_igp.py --username ali --dataset_name clerc-med-multi-flat

python script/flat_multivector/eval_flat_groundtruth.py --username ali --dataset clerc-med-multi-flat --method plaid
python script/flat_multivector/eval_flat_groundtruth.py --username ali --dataset clerc-med-multi-flat --method dessert
python script/flat_multivector/eval_flat_groundtruth.py --username ali --dataset clerc-med-multi-flat --method MUVERA
python script/flat_multivector/eval_flat_groundtruth.py --username ali --dataset clerc-med-multi-flat --method IGP
```

## 19. Which files an external user must know

If someone new wants to run this pipeline, the only things they must identify are:

1. the full document embedding `.npy`
2. the document length `.npy`
3. the full query embedding `.npy`
4. the query length `.npy`
5. the ground truth, either:
   - `.ivecs`, or
   - `qrels.tsv` + `query_ids.json` + `corpus_ids.json`

Once those are known, they can follow this README exactly.
