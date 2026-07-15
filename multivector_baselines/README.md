# Baselines


### flat multivector

##### environment

```
conda create -n flatmulti python=3.10.18 pip setuptools wheel -y
conda activate flatmulti

pip install -r requirements.txt
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
        --username Ah \
        --dataset scidocs-large-multi-flat-test \
        --doc-embeddings ./data/scidocs-large-multi/full_multi_embeddings_scidocs-large.npy \
        --doc-lens ./data/scidocs-large-multi/full_multi_chunk_num_scidocs-large.npy \
        --query-embeddings ./data/scidocs-large-multi/full_multi_embeddings_scidocs-large_query.npy \
        --query-lens ./data/scidocs-large-multi/full_multi_chunk_num_scidocs-large_query.npy \
        --runtime-root ./Dataset/multi-vector-retrieval \
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
        --method <method name>
    ```

- Option B: one wrapper

    ```bash
    bash script/flat_multivector/run_plaid_flat.sh ali clerc-med-multi-flat
    ```




3. Run Dessert


4. Run MUVERA


5. Run IGP