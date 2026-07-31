# Run hnswlib on the 1000-query subset

/home/ali/SVR-baselines/runs/hnswlib_clerc_runner \
  /data/ali/clerc-large-single/clerc-large-single_base.fvecs \
  /data/ali/baseline-data/clerc-large-single/inputs/query_1000.fvecs \
  /data/ali/baseline-data/clerc-large-single/hnswlib/index.bin \
  /data1/chenyifeng/MultiVector-Backup/svr-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  1000 \
  100 \
  16 \
  200 \
  100 \
  96 \
  1 \
  > /data1/chenyifeng/MultiVector-Backup/svr-baselines/runs/clerc-large-single/logs/hnswlib_k100_scored.log 2>&1


# Argument meaning:

# - `1000`: number of queries to run
# - `100`: return top-100 neighbors
# - `16`: `M`
# - `200`: `ef_construction`
# - `100`: `ef_search`
# - `96`: build threads
# - `1`: search threads

# Important:

# - search is intentionally one query at a time with one search thread
# - the logged `[Search Time]` excludes result-file writing



# BEIR-style evaluation

python3 /home/ali/SVR-baselines/runs/eval_beir_metrics.py \
  --groundtruth /data/ali/baseline-data/clerc-large-single/inputs/groundtruth_1000.ivecs \
  --results /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/ans_k100_scored.tsv \
  --k-values 1 3 5 10 100 \
  --output-json /home/ali/SVR-baselines/runs/clerc-large-single/hnswlib/beir_metrics_k100_scored.json
