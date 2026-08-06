DATASET=scidocs 

DATASET_BASE=/data1/chenyifeng/Data_backup/${DATASET}          
EMBEDDING_DIR=$DATASET_BASE/colbert
BEIR_DIR=$DATASET_BASE/beir

BASE_DIR=/data1/chenyifeng/MultiVector-Backup/WARP
INDEX_ROOT=$BASE_DIR/indexes
INDEX_NAME=beir-${DATASET}.split=test.precomputed=colbert.nbits=2

THRESHOLD=0.45
NDOCS=1024

export CXX=/usr/bin/g++-9
export CUDA_VISIBLE_DEVICES=3

# 构建索引
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

# 运行 WARP 搜索
export TORCH_NUM_THREADS=1
export OMP_NUM_THREADS=1

python utility/sweep_ncells.py \
  --dataset "$DATASET" \
  --dataset-dir "$BEIR_DIR" \
  --embedding-dir "$EMBEDDING_DIR" \
  --index-root "$INDEX_ROOT" \
  --index "$INDEX_NAME" \
  --split test \
  --nbits 2 \
  --k 100 \
  --baseline warp_precomputed_colbert_nbits2_thr${THRESHOLD}_ndocs${NDOCS} \
  --centroid-score-threshold $THRESHOLD \
  --ndocs $NDOCS \
  --ncells 1 2 4 8 16 32 64 128 256 512 \
  --output-csv "$BASE_DIR/my_runs/${DATASET}_colbert_thr${THRESHOLD}_ndocs${NDOCS}.csv"