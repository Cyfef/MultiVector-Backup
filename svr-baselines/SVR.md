# SVR Baselines

This repository contains the local benchmark setup used to run `hcnng`, `hnswlib`, and `ELPIS` on the single-vector datasets stored under `/data/ali`.

The main code root is:

- `/home/ali/SVR-baselines`

A mirrored copy for handoff is kept at:

- `/data/ali/svr-baselines`

Heavy data and indexes stay under `/data/ali`. Only results and logs are written under:

- `/home/ali/SVR-baselines/results`

## Directory Layout

Code:

- `/home/ali/SVR-baselines/hcnng`
- `/home/ali/SVR-baselines/hnswlib-master/hnswlib-master`
- `/home/ali/SVR-baselines/ELPIS`
- `/home/ali/SVR-baselines/runs`

Prepared benchmark data and indexes:

- `/data/ali/svr-baseline-data/<dataset>/inputs`
- `/data/ali/svr-baseline-data/<dataset>/hcnng`
- `/data/ali/svr-baseline-data/<dataset>/hnswlib`
- `/data/ali/svr-baseline-data/<dataset>/elpis`

Results:

- `/home/ali/SVR-baselines/results/<dataset>`
- `/home/ali/SVR-baselines/results/by_dataset_method/<dataset>/<method>`

Source qrels and external IDs:

- `/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef`
- `/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc`

The pipeline reads those source qrels, but does not modify them.

## Supported Datasets

Standardized datasets in the batch pipeline:

- `/data/ali/clef-small-single`
- `/data/ali/clef-med-single`
- `/data/ali/clerc-small-single`
- `/data/ali/clerc-med-single`

Legacy large run:

- `/data/ali/clerc-large-single`

For CLERC datasets in the standardized pipeline, only the first `1000` queries are evaluated.

## Raw Dataset Structure

Each dataset directory is expected to contain:

- `<dataset>_base.fvecs`
- `<dataset>_query.fvecs`
- `<dataset>_groundtruth.ivecs`

Binary formats:

- `.fvecs`
  - each vector is `int32 dim` followed by `dim` float32 values
- `.ivecs`
  - each row is `int32 k` followed by `k` int32 IDs
- `.bin`
  - raw float32 values only, no per-vector header
  - generated during preparation for methods that need it

Prepared input files created under `/data/ali/svr-baseline-data/<dataset>/inputs`:

- `base.bin`
- `query.bin`
- `query.fvecs`
- `groundtruth.ivecs`
- `query_ids.txt`
- `doc_ids.txt`
- `qrels_filtered.tsv`

`query_ids.txt` and `doc_ids.txt` are built from the JSONL files in the source qrels tree and are used to convert row-based ANN output into BEIR-style `qid docid rank score` TSV.

## Build

HCNNG:

```bash
g++ /home/ali/SVR-baselines/hcnng/hcnng.cpp -o /home/ali/SVR-baselines/hcnng/hcnng -std=c++11 -fopenmp -O3
g++ /home/ali/SVR-baselines/hcnng/search.cpp -o /home/ali/SVR-baselines/hcnng/search -std=c++11 -fopenmp -O3
```

hnswlib C++ runner:

```bash
g++ /home/ali/SVR-baselines/runs/hnswlib_clerc_runner.cpp -O3 -std=c++14 -fopenmp -o /home/ali/SVR-baselines/runs/hnswlib_clerc_runner
```

ELPIS:

```bash
cmake -S /home/ali/SVR-baselines/ELPIS/code -B /home/ali/SVR-baselines/ELPIS/code/build
cmake --build /home/ali/SVR-baselines/ELPIS/code/build -j 8
```

## Standardized End-to-End Run

The standard batch runner is:

- `/home/ali/SVR-baselines/runs/svr_benchmark_batch.py`

It uses:

- data root: `/data/ali/svr-baseline-data`
- results root: `/home/ali/SVR-baselines/results`
- qrels root: `/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt`

Prepare data and run the predefined sweeps:

```bash
python3 /home/ali/SVR-baselines/runs/svr_benchmark_batch.py
```

That script:

- prepares data under `/data/ali/svr-baseline-data/<dataset>/inputs`
- filters source qrels to the query subset in use
- builds indexes under `/data/ali/svr-baseline-data/<dataset>/<method>`
- writes raw ANN TSV outputs under `/home/ali/SVR-baselines/results/<dataset>/<method>`
- converts them to BEIR-style TSV
- evaluates `@10` and `@100`
- records `Recall`, `MRR`, `NDCG`, `MAP`, `P`, `R_cap`, `Hole`, `Accuracy`, and `QPS`

## Evaluation Files

Per dataset:

- summary: `/home/ali/SVR-baselines/results/<dataset>/summary.csv`
- metrics JSON: `/home/ali/SVR-baselines/results/<dataset>/eval/*.json`
- method results: `/home/ali/SVR-baselines/results/<dataset>/<method>/*.tsv`
- logs: `/home/ali/SVR-baselines/results/<dataset>/logs/*.log`

Global summary:

- `/home/ali/SVR-baselines/results/all_datasets_summary.csv`

Organized export:

- `/home/ali/SVR-baselines/results/by_dataset_method/<dataset>/<method>`

The export tree contains per-method results, logs, metrics, and summaries.

## QPS Measurement

QPS is measured from search time only.

It does not include:

- result TSV writing
- BEIR TSV conversion
- metric evaluation

For `hcnng` and `hnswlib`, search is run one query at a time. Build can still use multiple threads.

## CLERC Small Diagnostic Check

For official CLERC comparison, `clerc-small-single` must be evaluated the same way as `clerc-med-single` and `clerc-large-single`:

- use the CLERC source qrels from `/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc`
- filter to the first `1000` CLERC queries
- evaluate the generated BEIR-style TSV against those qrels

Those official source-qrels evaluations are the ones used in:

- `/home/ali/SVR-baselines/results/clerc-small-single/summary.csv`
- `/home/ali/SVR-baselines/results/all_datasets_summary.csv`
- `/home/ali/SVR-baselines/results/clerc_source_qrels_recheck.csv`

The near-zero `clerc-small-single` result is therefore the official comparable result under the shared CLERC qrels protocol.

Separately, I added a diagnostic-only check because `clerc-small-single` has an unusual vector property:

- for the first `1000` queries, query row `i` is identical to base row `i`
- meanwhile the CLERC source qrels map those same queries to external corpus IDs used by `clerc-med-single` and `clerc-large-single`

That diagnostic check is not the official benchmark and should not replace the source-qrels evaluation for cross-embedding comparison.

Diagnostic-only artifacts:

- script: `/home/ali/SVR-baselines/runs/recheck_clerc_small_corrected_qrels.py`
- local diagnostic qrels: `/data/ali/svr-baseline-data/clerc-small-single/inputs/qrels_corrected_identity_first1000.tsv`
- validation report: `/home/ali/SVR-baselines/results/clerc-small-single/eval/clerc_small_corrected_qrels_validation.json`
- diagnostic summary: `/home/ali/SVR-baselines/results/clerc-small-single/summary_corrected_qrels.csv`
- source-qrels vs diagnostic comparison: `/home/ali/SVR-baselines/results/clerc-small-single/summary_qrels_variant_comparison.csv`

Run the diagnostic check:

```bash
python3 /home/ali/SVR-baselines/runs/recheck_clerc_small_corrected_qrels.py
```

This diagnostic pass does not rebuild indexes and does not rerun search. It only:

- verifies the first `1000` query/base rows are identical
- builds a local identity-based qrels file under `/data/ali`
- re-evaluates the existing `hcnng`, `hnswlib`, and `ELPIS` BEIR TSV outputs against that diagnostic target

Diagnostic metrics JSON files are written beside the official metrics using names like:

- `hcnng_<setting>_corrected_qrels_beir_metrics.json`
- `hnswlib_<setting>_corrected_qrels_beir_metrics.json`
- `elpis_<setting>_corrected_qrels_beir_metrics.json`

## Rebuild Summaries and Export Tree

To rebuild the standard summary CSV files from the metrics JSON:

```bash
python3 /home/ali/SVR-baselines/runs/rebuild_result_summaries.py
```

To rebuild the organized export tree:

```bash
python3 /home/ali/SVR-baselines/runs/organize_svr_results.py
```

After the `clerc-small-single` repair run, the export tree also includes:

- `summary_corrected_qrels.csv` inside each `clerc-small-single/<method>` directory

## Legacy CLERC Large

The separate legacy large-dataset instructions remain in:

- `/home/ali/SVR-baselines/README_clerc_large_single.md`

The mirrored copy is:

- `/data/ali/svr-baselines/README_clerc_large_single.md`
