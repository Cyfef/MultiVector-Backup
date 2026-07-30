# Running HCNNG, ELPIS, and hnswlib on `clerc-large-single`

This document explains exactly how to run the baselines in this repo on:

- raw dataset directory: `/data/ali/clerc-large-single`
- generated data / indexes: `/data/ali/baseline-data/clerc-large-single`
- results / logs only: `/home/ali/SVR-baselines/runs/clerc-large-single`

It is written so a new person can reproduce the run without needing extra context.

## 1. What is in the raw dataset directory

The dataset directory used here is:

- `/data/ali/clerc-large-single`

The files present there are:

- `/data/ali/clerc-large-single/clerc-large-single_base.fvecs`
- `/data/ali/clerc-large-single/clerc-large-single_query.fvecs`
- `/data/ali/clerc-large-single/clerc-large-single_groundtruth.ivecs`
- `/data/ali/clerc-large-single/clerc-gt.npy`
- `/data/ali/clerc-large-single/full_single_embeddings_clerc_large.npy`
- `/data/ali/clerc-large-single/full_single_embeddings_clerc_large_query.npy`

Only these three files are required for the runs described here:

- `clerc-large-single_base.fvecs`
- `clerc-large-single_query.fvecs`
- `clerc-large-single_groundtruth.ivecs`

The `.npy` files are not used in the commands below.

## 2. Dataset structure and formats

### Raw input files

- `*_base.fvecs`: database vectors
- `*_query.fvecs`: query vectors
- `*_groundtruth.ivecs`: ground-truth nearest-neighbor IDs

### File formats

- `.fvecs`
  - binary format
  - each record is: `int32 dimension`, followed by `dimension` float32 values
  - HCNNG reads this format directly

- `.ivecs`
  - binary format
  - each record is: `int32 k`, followed by `k` int32 IDs
  - used for ground truth

- `.bin` used by ELPIS in this repo
  - raw float32 only
  - no per-record dimension header
  - vectors are stored back-to-back
  - ELPIS expects this format for dataset/query binary loading in the commands below

### Result TSV format

The current runners write retrieval results as:

```text
query_id<TAB>doc_id<TAB>rank<TAB>score
```

The score is `-L2^2`, the negative squared L2 distance. Higher scores are better, which is the convention expected by TREC/BEIR-style evaluators.

### Verified dataset facts for `clerc-large-single`

- base count: `530426`
- query count: `327414`
- dimension: `1024`
- ground-truth count: `327414`
- ground-truth neighbors per query: `1`

Important:

- The provided ground truth is only `1-NN`.
- Therefore `Recall@10` and `Recall@100` in this README mean:
  - does the single true NN appear in the returned top-10?
  - does the single true NN appear in the returned top-100?

This is a hit-rate interpretation against `1-NN` ground truth, not full recall against 10 or 100 ground-truth neighbors.

## 3. Directory layout used for this benchmark

### Source code

- HCNNG source: `/home/ali/SVR-baselines/hcnng`
- ELPIS source: `/home/ali/SVR-baselines/ELPIS`
- hnswlib source: `/home/ali/SVR-baselines/hnswlib-master/hnswlib-master`
- hnswlib runner used here: `/home/ali/SVR-baselines/runs/hnswlib_clerc_runner.cpp`

### Generated data and indexes

Everything heavy is stored under `/data/ali`:

- prep output root: `/data/ali/baseline-data/clerc-large-single`
- prep output dir: `/data/ali/baseline-data/clerc-large-single/inputs`
- HCNNG index dir: `/data/ali/baseline-data/clerc-large-single/hcnng`
- ELPIS index dir: `/data/ali/baseline-data/clerc-large-single/elpis/index`
- hnswlib index dir: `/data/ali/baseline-data/clerc-large-single/hnswlib`

### Results and logs

Only result-like outputs are stored under `/home/ali`:

- run root: `/home/ali/SVR-baselines/runs/clerc-large-single`
- HCNNG results: `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng`
- ELPIS results: `/home/ali/SVR-baselines/runs/clerc-large-single/elpis`
- hnswlib results: `/home/ali/SVR-baselines/runs/clerc-large-single/hnswlib`
- logs: `/home/ali/SVR-baselines/runs/clerc-large-single/logs`

## 4. Files produced by the preparation step

The helper script is:

- `/home/ali/SVR-baselines/runs/prepare_clerc_large_single.py`

For this dataset it produces:

- `/data/ali/baseline-data/clerc-large-single/inputs/base.bin`
- `/data/ali/baseline-data/clerc-large-single/inputs/query_1000.fvecs`
- `/data/ali/baseline-data/clerc-large-single/inputs/query_1000.bin`
- `/data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs`
- `/data/ali/baseline-data/clerc-large-single/inputs/query_full.bin`

Why both `.fvecs` and `.bin` exist:

- HCNNG needs `.fvecs`
- ELPIS needs raw `.bin`

## 5. One-time build steps

## System packages

On Ubuntu-like systems:

```bash
sudo apt-get update
sudo apt-get install -y g++ cmake libboost-dev
```

## Build HCNNG

```bash
cd /home/ali/SVR-baselines/hcnng
g++ hcnng.cpp -o hcnng -std=c++11 -fopenmp -O3
g++ search.cpp -o search -std=c++11 -fopenmp -O3
```

Expected binaries:

- `/home/ali/SVR-baselines/hcnng/hcnng`
- `/home/ali/SVR-baselines/hcnng/search`

## Build ELPIS

```bash
cmake -S /home/ali/SVR-baselines/ELPIS/code -B /home/ali/SVR-baselines/ELPIS/code/build
cmake --build /home/ali/SVR-baselines/ELPIS/code/build -j 8
```

Expected binary:

- `/home/ali/SVR-baselines/ELPIS/code/build/ELPIS`

Note:

- The ELPIS tree in this repo was locally patched during debugging.
- Query results are written as TSV in the form `query_id<TAB>vector_id<TAB>rank<TAB>score`.
- Global dataset IDs are preserved through the leaf graphs.
- Query-side bugs around incomplete top-k output, multi-probe pruning, heap merging, and `--nworker` handling were fixed.
- Binary record counting was fixed in both the query loader and the index build path.
- If you replace this tree with a fresh upstream checkout, re-verify output correctness before trusting IDs or recall numbers.

## 6. Prepare data for the 1000-query benchmark

Create target directories:

```bash
mkdir -p /data/ali/baseline-data/clerc-large-single/inputs
mkdir -p /data/ali/baseline-data/clerc-large-single/hcnng
mkdir -p /data/ali/baseline-data/clerc-large-single/elpis/index
mkdir -p /data/ali/baseline-data/clerc-large-single/hnswlib
mkdir -p /home/ali/SVR-baselines/runs/clerc-large-single/hcnng
mkdir -p /home/ali/SVR-baselines/runs/clerc-large-single/elpis
mkdir -p /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib
mkdir -p /home/ali/SVR-baselines/runs/clerc-large-single/logs
```

Run the prep script:

```bash
python3 /home/ali/SVR-baselines/runs/prepare_clerc_large_single.py \
  --dataset-dir /data/ali/clerc-large-single \
  --output-dir /data/ali/baseline-data/clerc-large-single/inputs \
  --num-queries 1000 \
  --dataset-prefix clerc-large-single
```

What this does:

- copies the first `1000` queries into `query_1000.fvecs`
- copies the first `1000` GT rows into `groundtruth_1000.ivecs`
- converts the full base set into `base.bin` for ELPIS
- converts the first `1000` queries into `query_1000.bin` for ELPIS
- converts the full query set into `query_full.bin`

## 7. Build indexes

## HCNNG index build

This is the command pattern used for HCNNG index construction:

```bash
/home/ali/SVR-baselines/hcnng/hcnng \
  /data/ali/clerc-large-single/clerc-large-single_base.fvecs \
  1000 \
  20 \
  /data/ali/baseline-data/clerc-large-single/hcnng/index.ivecs
```

Meaning of the parameters:

- `1000`: target cluster size
- `20`: number of clustering executions

Output:

- `/data/ali/baseline-data/clerc-large-single/hcnng/index.ivecs`

## ELPIS index build

If the ELPIS index does not already exist, build it like this:

```bash
/home/ali/SVR-baselines/ELPIS/code/build/ELPIS \
  --mode 0 \
  --dataset /data/ali/baseline-data/clerc-large-single/inputs/base.bin \
  --dataset-size 530426 \
  --timeseries-size 1024 \
  --index-path /data/ali/baseline-data/clerc-large-single/elpis/index/ \
  --kb 16 \
  --Lb 200 \
  --leaf-size 100 \
  --buffer-size 3
```

Meaning of the important parameters:

- `--dataset-size 530426`: number of base vectors
- `--timeseries-size 1024`: vector dimension
- `--kb 16`: graph outdegree control during leaf-graph build
- `--Lb 200`: build beamwidth
- `--leaf-size 100`: maximum vectors per leaf
- `--buffer-size 3`: buffer size in GB

Expected output:

- `/data/ali/baseline-data/clerc-large-single/elpis/index/root.idx`
- many `*.gl` files in the same directory

## 8. Run the 1000-query benchmark

## HCNNG query run

Command used:

```bash
/home/ali/SVR-baselines/hcnng/search \
  /data/ali/clerc-large-single/clerc-large-single_base.fvecs \
  /data/ali/baseline-data/clerc-large-single/inputs/query_1000.fvecs \
  /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  /data/ali/baseline-data/clerc-large-single/hcnng/index.ivecs \
  100 \
  2000 \
  /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv \
  /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/qps_k100_scored.tsv \
  > /home/ali/SVR-baselines/runs/clerc-large-single/logs/hcnng_k100_scored.log 2>&1
```

Important parameter:

- `max_calc=2000`
  - this is HCNNG's search budget
  - it limits how many vectors are explored / distance calculations are spent per query
  - larger `max_calc` usually improves recall and lowers QPS

Outputs:

- result rows: `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv`
- QPS log: `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/qps_k100_scored.tsv`
- full run log: `/home/ali/SVR-baselines/runs/clerc-large-single/logs/hcnng_k100_scored.log`

## hnswlib build and query run

### Compile the C++ runner

```bash
g++ /home/ali/SVR-baselines/runs/hnswlib_clerc_runner.cpp \
  -O3 -std=c++14 -fopenmp \
  -o /home/ali/SVR-baselines/runs/hnswlib_clerc_runner
```

### Run hnswlib on the 1000-query subset

Command used:

```bash
/home/ali/SVR-baselines/runs/hnswlib_clerc_runner \
  /data/ali/clerc-large-single/clerc-large-single_base.fvecs \
  /data/ali/baseline-data/clerc-large-single/inputs/query_1000.fvecs \
  /data/ali/baseline-data/clerc-large-single/hnswlib/index.bin \
  /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  1000 \
  100 \
  16 \
  200 \
  100 \
  96 \
  1 \
  > /home/ali/SVR-baselines/runs/clerc-large-single/logs/hnswlib_k100_scored.log 2>&1
```

Argument meaning:

- `1000`: number of queries to run
- `100`: return top-100 neighbors
- `16`: `M`
- `200`: `ef_construction`
- `100`: `ef_search`
- `96`: build threads
- `1`: search threads

Important:

- this runner is C++ only
- index construction may use multiple threads
- search is intentionally one query at a time with one search thread
- the logged `[Search Time]` excludes result-file writing

Outputs:

- index: `/data/ali/baseline-data/clerc-large-single/hnswlib/index.bin`
- results: `/home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv`
- log: `/home/ali/SVR-baselines/runs/clerc-large-single/logs/hnswlib_k100_scored.log`

## ELPIS query run

### Single run command

Command pattern:

```bash
/home/ali/SVR-baselines/ELPIS/code/build/ELPIS \
  --queries /data/ali/baseline-data/clerc-large-single/inputs/query_1000.bin \
  --queries-size 1000 \
  --k 100 \
  --index-path /data/ali/baseline-data/clerc-large-single/elpis/index/ \
  --L 100 \
  --nprobes 1000 \
  --parallel 1 \
  --nworker 1 \
  --mode 1 \
  --output-path /home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv \
  > /home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1000_scored.log 2>&1
```

Important parameters:

- `--queries-size 1000`: run only the prepared 1000-query subset
- `--k 100`: write top-100 IDs per query
- `--L 100`: search beamwidth
- `--nprobes N`: number of routed leaves to probe
- `--parallel 1`: use ELPIS parallel query mode
- `--nworker 1`: force one worker for reproducibility

Outputs:

- result rows: `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv`
- full run log: `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1000_scored.log`

### Full `nprobes` sweep used in this investigation

```bash
for NP in 1 5 10 20 50 100 200 300 500 1000; do
  /home/ali/SVR-baselines/ELPIS/code/build/ELPIS \
    --queries /data/ali/baseline-data/clerc-large-single/inputs/query_1000.bin \
    --queries-size 1000 \
    --k 100 \
    --index-path /data/ali/baseline-data/clerc-large-single/elpis/index/ \
    --L 100 \
    --nprobes "$NP" \
    --parallel 1 \
    --nworker 1 \
    --mode 1 \
    --output-path "/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np${NP}_clean.tsv" \
    > "/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np${NP}_clean.log" 2>&1
done
```

## 9. Evaluate Recall@10 and Recall@100

Evaluator script:

- `/home/ali/SVR-baselines/runs/eval_recall.py`

## HCNNG evaluation

```bash
python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv \
  --k 10

python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv \
  --k 100
```

## ELPIS evaluation

```bash
python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv \
  --k 10

python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv \
  --k 100
```

Optional `Recall@1` check:

```bash
python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv \
  --k 1

python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv \
  --k 1
```

## hnswlib evaluation

```bash
python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  --k 1

python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  --k 10

python3 /home/ali/SVR-baselines/runs/eval_recall.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  --k 100
```

## BEIR-style evaluation

BEIR is not installed in this environment, so this repo includes a local BEIR-style evaluator:

- `/home/ali/SVR-baselines/runs/eval_beir_metrics.py`

It reports:

- `ndcg`
- `map`
- `recall`
- `precision`
- `mrr`
- `recall_cap`
- `hole`
- `accuracy`

Use `--ignore-identical-ids` only for datasets where query IDs and document IDs share the same namespace and self-matches should be removed. Do not use it for `clerc-large-single`.

Example commands:

```bash
python3 /home/ali/SVR-baselines/runs/eval_beir_metrics.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv \
  --k-values 1 3 5 10 100 \
  --output-json /home/ali/SVR-baselines/runs/clerc-large-single/hcnng/beir_metrics_k100_scored.json

python3 /home/ali/SVR-baselines/runs/eval_beir_metrics.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  --k-values 1 3 5 10 100 \
  --output-json /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/beir_metrics_k100_scored.json

python3 /home/ali/SVR-baselines/runs/eval_beir_metrics.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv \
  --k-values 1 3 5 10 100 \
  --output-json /home/ali/SVR-baselines/runs/clerc-large-single/elpis/beir_metrics_k100_np1000_scored.json
```

## 10. How to compute QPS

## HCNNG QPS

HCNNG already writes QPS here:

- `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/qps_k100_scored.tsv`

Current file contents:

```text
K	max_calc	Recall	QPS
100	2000	3.14	6944.44
```

Use the `QPS` column.

## ELPIS QPS

ELPIS prints query time to the log for each run.

Example:

- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1000_scored.log`

Measured line for `nprobes=1000`:

```text
[Querying Time] 109.699(sec)
```

For `1000` queries:

- `QPS = 1000 / 109.699 = 9.12`

General command to compute ELPIS QPS from a log:

```bash
python3 - <<'PY'
import pathlib
import re

num_queries = 1000
log_path = pathlib.Path("/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1000_scored.log")
text = log_path.read_text()
sec = float(re.search(r"\[Querying Time\]\s+([0-9.]+)\(sec\)", text).group(1))
print(f"QPS={num_queries / sec:.2f}")
PY
```

## hnswlib QPS

hnswlib logs pure search time like this:

```text
[Search Time] 0.818641(sec)
[QPS] 1221.54
```

For this runner:

- the search loop is one query at a time
- `search_threads=1`
- TSV writing happens after the timed region

So the logged `QPS` is already the search-only QPS to use.

## 11. Results from the verified 1000-query run

The verified HCNNG result file was:

- HCNNG: `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv`

The verified hnswlib result file was:

- hnswlib: `/home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv`

The verified ELPIS result file used for the final BEIR-style metrics was:

- ELPIS: `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv`

The trustworthy post-fix ELPIS result files are:

- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np5_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np10_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np20_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np50_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np100_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np200_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np300_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np500_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_clean.tsv`

Structural validation:

- `1000` predicted queries
- query IDs `0..999`
- no out-of-range IDs in the clean sweep files
- `100000` rows for every clean sweep file except `nprobes=1`
- `nprobes=1` produced `69671` rows because many queries had fewer than `100` valid candidates, which is now represented honestly instead of being filled with garbage IDs

### Final recall and QPS

| Method | Recall@1 | Recall@10 | Recall@100 | QPS |
|---|---:|---:|---:|---:|
| HCNNG | 0.055 | 0.263 | 0.595 | 6944.44 |
| hnswlib | 0.054 | 0.261 | 0.592 | 1221.54 |
| ELPIS, nprobes=1000 | 0.048 | 0.197 | 0.348 | 9.12 |

Notes:

- HCNNG's own internal printed `Recall@100(2000): 2.869` in its log is not the metric to use for the requested `Recall@10` and `Recall@100` on this dataset.
- Because the GT has only one neighbor, the external evaluator in `/home/ali/SVR-baselines/runs/eval_recall.py` is the authoritative metric here.
- hnswlib here was intentionally run with sequential search, one query at a time and one search thread, so its QPS is not directly comparable to a multithreaded batch-search configuration.

### BEIR-style metric summary

These values were produced by `/home/ali/SVR-baselines/runs/eval_beir_metrics.py` on the scored TSV files.

| Method | NDCG@10 | MAP@10 | MRR@10 | Recall@10 | P@10 | Accuracy@10 | NDCG@100 | MAP@100 | MRR@100 | Recall@100 | Accuracy@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HCNNG | 0.14652 | 0.11087 | 0.11087 | 0.263 | 0.0263 | 0.263 | 0.21289 | 0.12263 | 0.12263 | 0.595 | 0.595 |
| hnswlib | 0.14495 | 0.10947 | 0.10947 | 0.261 | 0.0261 | 0.261 | 0.21118 | 0.12123 | 0.12123 | 0.592 | 0.592 |
| ELPIS, nprobes=1000 | 0.11292 | 0.08722 | 0.08722 | 0.197 | 0.0197 | 0.197 | 0.14326 | 0.09265 | 0.09265 | 0.348 | 0.348 |

Metric JSON files:

- `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/beir_metrics_k100_scored.json`
- `/home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/beir_metrics_k100_scored.json`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/beir_metrics_k100_np1000_scored.json`

### ELPIS correctness and tuning conclusion

The original ELPIS output was wrong for real correctness reasons:

- earlier files included invalid or incomplete outputs
- that came from query-side bugs, not just low recall

Those bugs were fixed in the local source tree. After the fixes, the outputs are structurally valid, but recall is still much lower than HCNNG on the current index and search settings.

### ELPIS `nprobes` sweep on the current index

All runs below used:

- the same ELPIS index under `/data/ali/baseline-data/clerc-large-single/elpis/index/`
- the same first `1000` queries
- `k=100`
- `L=100`
- `parallel=1`
- `nworker=1`

| nprobes | Recall@1 | Recall@10 | Recall@100 | Query time (sec) | QPS |
|---|---:|---:|---:|---:|---:|
| 1 | 0.002 | 0.003 | 0.003 | 0.161404 | 6195.63 |
| 5 | 0.006 | 0.010 | 0.010 | 76.9390 | 13.00 |
| 10 | 0.009 | 0.015 | 0.016 | 75.7849 | 13.20 |
| 20 | 0.008 | 0.021 | 0.025 | 87.9117 | 11.38 |
| 50 | 0.020 | 0.054 | 0.060 | 87.6843 | 11.40 |
| 100 | 0.022 | 0.065 | 0.085 | 77.3856 | 12.92 |
| 200 | 0.025 | 0.089 | 0.135 | 85.7654 | 11.66 |
| 300 | 0.032 | 0.112 | 0.181 | 92.1555 | 10.85 |
| 500 | 0.043 | 0.154 | 0.259 | 94.6170 | 10.57 |
| 1000 | 0.048 | 0.197 | 0.348 | 109.6990 | 9.12 |

Conclusion:

- increasing `nprobes` definitely improves ELPIS recall
- but `nprobes` alone is not enough to match HCNNG on this dataset
- the best measured point here is `nprobes=1000`
- even there, ELPIS remains below HCNNG:
- `Recall@1`: `0.048` vs `0.054`
- `Recall@10`: `0.197` vs `0.262`
- `Recall@100`: `0.348` vs `0.593`

If you want stronger ELPIS recall on this dataset, tune at least:

- `--nprobes`
- `--L`
- `--kb`
- `--Lb`
- `--leaf-size`

## 12. Fast checklist for a new user

If you want the shortest reproducible path:

1. Build the binaries.
2. Run the prep script.
3. Build or reuse the HCNNG index.
4. Build or reuse the hnswlib index.
5. Build or reuse the ELPIS index.
6. Run the HCNNG query command.
7. Run the hnswlib query command.
8. Run the ELPIS query command with the `nprobes` value you want to test.
9. Run the evaluator for `--k 1`, `--k 10`, and `--k 100`.
10. Run the BEIR-style evaluator if you need the full BEIR metric set.
11. Read HCNNG QPS from `qps_k100_scored.tsv`.
12. Read hnswlib QPS from `hnswlib_k100_scored.log`.
13. Read ELPIS query time from the matching `elpis_k100_np*_scored.log` and compute `QPS = nq / seconds`.
13. If you want to reproduce the ELPIS investigation in this README, run the full `nprobes` sweep instead of only `nprobes=1`.

## 13. Running on another dataset with the same structure

If another dataset follows the same three-file layout:

- `<prefix>_base.fvecs`
- `<prefix>_query.fvecs`
- `<prefix>_groundtruth.ivecs`

then you can reuse the same prep script by changing only:

- `--dataset-dir`
- `--output-dir`
- `--dataset-prefix`

Example:

```bash
python3 /home/ali/SVR-baselines/runs/prepare_clerc_large_single.py \
  --dataset-dir /data/ali/your-dataset \
  --output-dir /data/ali/baseline-data/your-dataset/inputs \
  --num-queries 1000 \
  --dataset-prefix your-dataset
```

If filenames do not follow that prefix pattern, pass them explicitly:

```bash
python3 /home/ali/SVR-baselines/runs/prepare_clerc_large_single.py \
  --dataset-dir /data/ali/your-dataset \
  --output-dir /data/ali/baseline-data/your-dataset/inputs \
  --num-queries 1000 \
  --base-fvecs /data/ali/your-dataset/custom_base.fvecs \
  --query-fvecs /data/ali/your-dataset/custom_query.fvecs \
  --groundtruth-ivecs /data/ali/your-dataset/custom_groundtruth.ivecs
```

Then update the run commands with:

- the correct dataset counts
- the correct vector dimension
- the new generated paths

## 14. Exact paths referenced in this run

### Raw dataset

- `/data/ali/clerc-large-single/clerc-large-single_base.fvecs`
- `/data/ali/clerc-large-single/clerc-large-single_query.fvecs`
- `/data/ali/clerc-large-single/clerc-large-single_groundtruth.ivecs`
- `/data/ali/clerc-large-single/clerc-gt.npy`
- `/data/ali/clerc-large-single/full_single_embeddings_clerc_large.npy`
- `/data/ali/clerc-large-single/full_single_embeddings_clerc_large_query.npy`

### Generated inputs under `/data/ali`

- `/data/ali/baseline-data/clerc-large-single/inputs/base.bin`
- `/data/ali/baseline-data/clerc-large-single/inputs/query_1000.fvecs`
- `/data/ali/baseline-data/clerc-large-single/inputs/query_1000.bin`
- `/data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs`
- `/data/ali/baseline-data/clerc-large-single/inputs/query_full.bin`

### Generated indexes under `/data/ali`

- `/data/ali/baseline-data/clerc-large-single/hcnng/index.ivecs`
- `/data/ali/baseline-data/clerc-large-single/elpis/index/root.idx`
- `/data/ali/baseline-data/clerc-large-single/elpis/index/*.gl`
- `/data/ali/baseline-data/clerc-large-single/hnswlib/index.bin`

### Results under `/home/ali`

- `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/ans_k100_scored.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/qps_k100_scored.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/hcnng/beir_metrics_k100_scored.json`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/hcnng_k100_scored.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/beir_metrics_k100_scored.json`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/hnswlib_k100_scored.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_scored.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/beir_metrics_k100_np1000_scored.json`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1000_scored.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np5_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np10_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np20_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np50_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np100_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np200_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np300_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np500_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/elpis/ans_k100_np1000_clean.tsv`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/hcnng_k100.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/hnswlib_k100.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np5_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np10_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np20_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np50_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np100_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np200_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np300_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np500_clean.log`
- `/home/ali/SVR-baselines/runs/clerc-large-single/logs/elpis_k100_np1000_clean.log`
- `/home/ali/SVR-baselines/runs/eval_recall.py`
- `/home/ali/SVR-baselines/runs/prepare_clerc_large_single.py`
- `/home/ali/SVR-baselines/runs/hnswlib_clerc_runner.cpp`
