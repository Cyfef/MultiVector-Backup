#!/usr/bin/env bash
set -euo pipefail

EMVB_ROOT=${EMVB_ROOT:-/data/ali/EMVB}
PREP_PY=${PREP_PY:-/data/ali/gem-baseline/bin/python}
EVAL_PY=${EVAL_PY:-/data/ali/env/bin/python}

SCIDOCS_EMB_DIR=${SCIDOCS_EMB_DIR:-/data1/liuyaoyang/Papers/ACFDE/output/scidocs/colbert}
SCIDOCS_DATASET_DIR=${SCIDOCS_DATASET_DIR:-/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs}
INDEX_DIR=${INDEX_DIR:-$EMVB_ROOT/work/indexes/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32}
RESULT_DIR=${RESULT_DIR:-$EMVB_ROOT/work/results/scidocs_colbert_simple/emvb_ivfpq_l2_nlist4096_m32}
LOG_DIR=${LOG_DIR:-$EMVB_ROOT/logs}

mkdir -p "$LOG_DIR" "$RESULT_DIR"

cd "$EMVB_ROOT"

if [[ ! -x "$EMVB_ROOT/build/perf_emvb" ]]; then
  mkdir -p "$EMVB_ROOT/build"
  cmake -S "$EMVB_ROOT" -B "$EMVB_ROOT/build" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF 2>&1 | tee "$LOG_DIR/build_cmake.log"
  cmake --build "$EMVB_ROOT/build" -j"$(nproc)" 2>&1 | tee "$LOG_DIR/build_make.log"
fi

if [[ ! -f "$INDEX_DIR/metadata.json" ]]; then
  "$PREP_PY" -u "$EMVB_ROOT/prepare_emvb_data.py" \
    --base-embeddings "$SCIDOCS_EMB_DIR/corpus_points.npy" \
    --base-offsets "$SCIDOCS_EMB_DIR/corpus_offsets.npy" \
    --query-embeddings "$SCIDOCS_EMB_DIR/query_points.npy" \
    --query-offsets "$SCIDOCS_EMB_DIR/query_offsets.npy" \
    --output-dir "$INDEX_DIR" \
    --nlist 4096 \
    --pq-m 32 \
    --add-batch-size 200000 \
    --faiss-threads 48 \
    2>&1 | tee "$LOG_DIR/prepare_scidocs_colbert_simple.log"
fi

PYTHON_BIN="$EVAL_PY" "$EVAL_PY" "$EMVB_ROOT/run_emvb_ratio_sweep.py" \
  --dataset scidocs \
  --dataset-dir "$SCIDOCS_DATASET_DIR" \
  --split test \
  --index-dir "$INDEX_DIR" \
  --results-dir "$RESULT_DIR" \
  --ratios 0.25 0.5 1 2 \
  --k-values 10 100 \
  --query-id-mode positional \
  2>&1 | tee "$LOG_DIR/sweep_scidocs_colbert_simple.log"

echo "Summary written to $RESULT_DIR/metrics_summary.csv"
