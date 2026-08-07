# SSR

## Environment

```bash
conda create -n ssr python=3.10 -y
conda activate ssr

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[data,eval]"
```


## Run


### Train

MS MARCO Passage data

```bash
wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives-bm25_1k.jsonl.gz

scp "C:\Users\Lenovo\Desktop\msmarco-hard-negatives-bm25_1k.jsonl.gz" chenyifeng@222.29.136.239:/data1/chenyifeng/MultiVector-Backup/SSR/data/raw/msmarco/passage/
```

```bash
python prepare_msmarco.py \
  --subset passage \
  --raw-dir ./data/raw/msmarco \
  --processed-dir ./data/processed/msmarco
```

1. SSR-tok

```bash
python -m ssr.train \
  --dataset msmarco-passage \
  --data-dir ./data/processed/msmarco/passage \
  --sae-token-scope non-cls \
  --sample-format triplet \
  --output-dir ./output/ssr-token
```

2. SSR-CLS

```bash
python -m ssr.train \
  --dataset msmarco-passage \
  --data-dir ./data/processed/msmarco/passage \
  --sae-token-scope cls \
  --sample-format triplet \
  --output-dir ./output/ssr-cls
```


### Retrieval

```bash
export MTEB_CACHE=/data1/chenyifeng/tmp/mteb
export HF_XET_DISABLE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
```

BEIR-style dataset:

```bash
python prepare_mteb_eval.py \
  --datasets scidocs \
  --split test \
  --processed-dir ./data/processed/mteb
```

#### Build Index

1. SSR-tok

```bash
python -m ssr.retrieval.eval_mteb \
  --task index \
  --variant ssr \
  --model-path ./output/ssr-token/final \
  --dataset scidocs \
  --data-dir ./data/processed/mteb \
  --index-cache-dir ./data/cache/mteb_index_e2e/scidocs \
  --encode-device cuda:0
```

2. SSR-CLS

```bash
python -m ssr.retrieval.eval_mteb \
  --task index \
  --variant ssr-cls \
  --model-path ./output/ssr-token/final \
  --cls-sae-path ./output/ssr-cls/final \
  --dataset scidocs \
  --data-dir ./data/processed/mteb \
  --index-cache-dir data/cache/mteb_index_e2e/scidocs_cls \
  --encode-device cuda:0
```

#### Retrieval

cpu single-thread

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

1. SSR++

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend e2e-index \
  --variant ssr++ \
  --model-path ./output/ssr-token/final \
  --dataset scidocs \
  --data-dir ./data/processed/mteb \
  --index-cache-dir ./data/cache/mteb_index_e2e/scidocs \
  --score-device index \
  --index-accum-device cpu \
  --top-k 100 \                     
  --mrr-k 10 100 \                      
  --ndcg-k 10 100 \                    
  --recall-k 10 100 \         
  --map-k 10 100 
```


2. SSR-CLS++

```bash
python -m ssr.retrieval.eval_mteb \
  --task retrieval \
  --corpus-backend e2e-index \
  --variant ssr-cls++ \
  --model-path ./output/ssr-token/final \
  --cls-sae-path ./output/ssr-cls/final \
  --dataset scidocs \
  --data-dir ./data/processed/mteb \
  --index-cache-dir ./data/cache/mteb_index_e2e/scidocs_cls \
  --score-device index \
  --index-accum-device cpu
  --top-k 100 \                     
  --mrr-k 10 100 \                      
  --ndcg-k 10 100 \                    
  --recall-k 10 100 \         
  --map-k 10 100 
```
