# Baselines


### flat multivector

##### environment

```bash
conda create -n flatmulti python=3.10.18 pip setuptools wheel -y
conda activate flatmulti

pip install -r requirements.txt
```

```bash
mkdir -p /data1/chenyifeng/libs
cd /data1/chenyifeng/libs

git clone https://github.com/cmuparlay/parlaylib.git
cd parlaylib

mkdir build && cd build

cmake -DCMAKE_INSTALL_PREFIX=/data1/chenyifeng/libs/parlay_install ..
make -j
make install
```

```bash
rm -rf /data1/chenyifeng/multi-vector-retrieval/build/*
rm /data1/chenyifeng/MultiVector-Backup/multivector_baselines/script/evaluation/IGP*.so

export CMAKE_PREFIX_PATH=/data1/chenyifeng/libs/parlay_install:$CMAKE_PREFIX_PATH
export Parlay_DIR=/data1/chenyifeng/libs/parlay_install/share/parlay/cmake
```



##### run

1. Prepare a flat dataset: script/flat_multivector/prepare_flat_multivector_dataset.py

    ```bash
    python script/flat_multivector/prepare_flat_multivector_dataset.py \
        --username <username> \
        --dataset <new-dataset-name> \
        --doc-embeddings <doc-embeddings.npy> \
        --doc-lens <doc-lens.npy> \
        --query-embeddings <query-embeddings.npy> \
        --query-lens <query-lens.npy> \
        --groundtruth-ivecs <groundtruth.ivecs> \
        --force
    ```

    Example for ```scidocs-large-multi``` :

    ```bash
    python script/flat_multivector/prepare_flat_multivector_dataset.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test \
        --doc-embeddings ./data/scidocs-large-multi/full_multi_embeddings_scidocs-large.npy \
        --doc-lens ./data/scidocs-large-multi/full_multi_chunk_num_scidocs-large.npy \
        --query-embeddings ./data/scidocs-large-multi/full_multi_embeddings_scidocs-large_query.npy \
        --query-lens ./data/scidocs-large-multi/full_multi_chunk_num_scidocs-large_query.npy \
        --max-queries 1000 \
        --force
    ```

2. Build and run Plaid

There are two ways to do this.

- Option A: explicit commands

    Build the Plaid index:

    ```bash
    python script/flat_multivector/build_plaid_from_flat_dataset.py \
        --username <username> \
        --dataset <dataset name> \
        --manifest <manifest.json path>
    ```

    Run Plaid retrieval:

    ```bash
    python script/evaluation/eval_plaid.py \
        --username <username> \
        --dataset <dataset name>
    ```

    Evaluate Plaid results against local ground truth:

    ```bash
    python script/flat_multivector/eval_flat_groundtruth.py \
        --username <username> \
        --dataset <dataset name> \
        --method plaid
    ```

    Example for ```scidocs-large-multi``` :

    ```
    python script/flat_multivector/build_plaid_from_flat_dataset.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test \
        --manifest /data1/chenyifeng/Dataset/multi-vector-retrieval/FlatData/scidocs-large-multi-flat-test/manifest.json
    ```

    ```bash
    python script/evaluation/eval_plaid.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test
    ```

    ```bash
    python script/flat_multivector/eval_flat_groundtruth.py \
        --username chenyifeng \
        --dataset scidocs-large-multi-flat-test \
        --method plaid
    ```

- Option B: one wrapper

    ```bash
    bash script/flat_multivector/run_plaid_flat.sh <username> <dataset name>
    ```

    Example for ```scidocs-large-multi``` :

    ```
    bash script/flat_multivector/run_plaid_flat.sh chenyifeng scidocs-large-multi-flat-test
    ```

3. Run Dessert

Important:

- Dessert depends on Plaid artifacts
- run Plaid build first
- Dessert uses the Plaid centroids and codes

Command:

```bash
python script/evaluation/eval_dessert.py \
  --username <username> \
  --dataset <dataset name>
```

Evaluate Dessert:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset name> \
  --method dessert
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_dessert_flat.sh <username> <dataset name>
```


Example for ```scidocs-large-multi``` :

```bash
python script/evaluation/eval_dessert.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test
```

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test \
  --method dessert
```

Or

```bash
bash script/flat_multivector/run_dessert_flat.sh chenyifeng scidocs-large-multi-flat-test
```


4. Run MUVERA

Build environment:

```bash
mkdir -p local

# fmt
git clone https://github.com/fmtlib/fmt.git
cd fmt
git checkout 9.1.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DFMT_TEST=OFF \
    -DBUILD_SHARED_LIBS=ON   
make -j $(nproc)
make install
cd ../..

# spdlog
git clone https://github.com/gabime/spdlog.git
cd spdlog
git checkout v1.11.0
mkdir build && cd build
cmake .. \
    -DCMAKE_INSTALL_PREFIX=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPDLOG_FMT_EXTERNAL=ON \
    -DSPDLOG_BUILD_SHARED=ON \
    -Dfmt_DIR=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local/lib/cmake/fmt
make -j $(nproc)
make install

export LD_LIBRARY_PATH=/data1/chenyifeng/MultiVector-Backup/multivector_baselines/local/lib:$LD_LIBRARY_PATH
```

Command:

```bash
python script/evaluation/eval_muvera.py \
  --username <username> \
  --dataset_name <dataset name>
```

Important:

- MUVERA reads `Embedding/<dataset>/base_embedding/*.npy`
- those files are created by the Plaid build path
- so if you only ran the preparation script, run Plaid build first

Evaluate MUVERA:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset name> \
  --method MUVERA
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_muvera_flat.sh <username> <dataset name>
```

Example for ```scidocs-large-multi``` :

```bash
python script/evaluation/eval_muvera.py \
  --username chenyifeng \
  --dataset_name scidocs-large-multi-flat-test
```

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test \
  --method MUVERA
```

Or

```bash
bash script/flat_multivector/run_muvera_flat.sh chenyifeng scidocs-large-multi-flat-test
```



5. Run IGP

Command:

```bash
python script/evaluation/eval_igp.py \
  --username <username> \
  --dataset_name <dataset name>
```

Important:

- IGP also reads `Embedding/<dataset>/base_embedding/*.npy`
- those files are created by the Plaid build path
- so if you only ran the preparation script, run Plaid build first

Evaluate IGP:

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username <username> \
  --dataset <dataset name> \
  --method IGP
```

Or use the wrapper:

```bash
bash script/flat_multivector/run_igp_flat.sh <username> <dataset name>
```


Example for ```scidocs-large-multi``` :

```bash
python script/evaluation/eval_igp.py \
  --username chenyifeng \
  --dataset_name scidocs-large-multi-flat-test
```

```bash
python script/flat_multivector/eval_flat_groundtruth.py \
  --username chenyifeng \
  --dataset scidocs-large-multi-flat-test \
  --method IGP
```

Or

```bash
bash script/flat_multivector/run_igp_flat.sh chenyifeng scidocs-large-multi-flat-test
```


