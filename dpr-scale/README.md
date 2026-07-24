# CITADEL

## I. Environment

```
conda create -n CITADEL python=3.8 -y
conda activate CITADEL

conda install -c conda-forge faiss-gpu
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
pip install -r requirements.txt
```

download model checkpoint:

```bash
mkdir ckpt
cd ckpt

# CITADEL
wget https://dl.fbaipublicfiles.com/citadel/checkpoints/citadel/citadel/checkpoint_best.ckpt

# CITADEL+
wget https://dl.fbaipublicfiles.com/citadel/checkpoints/citadel/citadel_plus/checkpoint_best.ckpt
```


## II. Evaluation

### BEIR-style

1. download and convert beir-style dataset into dpr format

```bash
DATASET=(arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scifact scidocs trec-covid webis-touche2020)
for dataset in ${DATASET[*]}
do
    echo $dataset
    python dpr_scale/citadel_scripts/convert_beir_to_dpr_format.py $dataset <output path>
done
```

2. generate embeddings

We then encode the corpus of dataset. For datasets with large corpus, we split it into multiple shards:

```bash
CHECKPOINT_PATH=<your_path_to_ckpt>

DATASET=(arguana nfcorpus fiqa quora scidocs scifact trec-covid webis-touche2020)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    CTX_EMBEDDINGS_DIR=<your_path_to_output_embedding_dir>
    DATA_PATH=</data_path/beir/datasets/${dataset}/dpr-scale/corpus.tsv>

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH nohup python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m --config-name msmarco_aws.yaml \
    datamodule=generate \
    datamodule.test_path=$DATA_PATH \
    task=multivec task/model=citadel_model \
    task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
    task.shared_model=True \
    +task.add_cls=True \
    +task.query_topk=1 +task.context_topk=5 \
    +task.weight_threshold=0.0 \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.add_context_id=False > nohup_${dataset}.log 2>&1&
done
```

if shard:

```bash
DATASET=(climate-fever dbpedia-entity fever hotpotqa nq)
SHARD=(0 1 2)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    for shard in ${SHARD[*]}
    do
    CTX_EMBEDDINGS_DIR=<your_path_to_output_shard_embeddings>
    DATA_PATH=</data_path/beir/datasets/${dataset}/dpr-scale/corpus.00$shard.tsv>

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH nohup python dpr_scale/citadel_scripts/generate_multivec_embeddings.py -m --config-name msmarco_aws.yaml \
    datamodule=generate \
    datamodule.test_path=$DATA_PATH \
    task=multivec task/model=citadel_model \
    task.model.tok_projection_dim=32 task.model.cls_projection_dim=128 \
    task.shared_model=True +task.cross_batch=False +task.in_batch=True \
    +task.add_cls=True \
    +task.query_topk=1 +task.context_topk=5 \
    +task.weight_threshold=0.0 \
    +task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
    +task.checkpoint_path=$CHECKPOINT_PATH \
    +task.add_context_id=False > nohup_${dataset}_${shard}.log 2>&1&
    done
```

3. Merge embeddings

```bash
DATASET=(arguana nfcorpus fiqa quora scidocs scifact trec-covid webis-touche2020 climate-fever dbpedia-entity fever hotpotqa nq)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=<your_path_to_merged_embeddings>
    CTX_EMBEDDINGS_DIR=<your_path_to_embedding_dirs>
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done
```
You could further compress the index size using product quantization and pruning. We skip the compression step for simplicity.


4. Retrieval

```bash
dataset=$1
OUTPUT_DIR=<path_to_retrieval_output_dir>
CTX_EMBEDDINGS_DIR=<path_to_merged_embeddings/expert>
CHECKPOINT_PATH=<path_to_your_ckpt>

I2D_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/index2docid.tsv
DATA_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/corpus.tsv
PATH_TO_QUERIES_TSV=/data_path/beir/datasets/${dataset}/dpr-scale/queries.tsv

PORTION=0.001 # how much portion of the index should be moved to GPU before retrieval

HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_citadel_retrieval.py \
--config-name msmarco_aws.yaml \
datamodule=generate_multivec_query_emb \
datamodule.test_path=$PATH_TO_QUERIES_TSV \
datamodule.test_batch_size=1 \
+datamodule.trec_format=True \
task=multivec_retrieval task/model=citadel_model \
task.model.tok_projection_dim=32 \
task.model.cls_projection_dim=128 +task.add_cls=True task.shared_model=True \
+task.query_topk=1 +task.context_topk=5 \
+task.output_path=$OUTPUT_DIR \
+task.ctx_embeddings_dir=$CTX_EMBEDDINGS_DIR \
+task.checkpoint_path=$CHECKPOINT_PATH \
+task.index2docid_path=$I2D_PATH \
+task.passages=$DATA_PATH \
+task.portion="$PORTION" \
+task.topk=1000 +task.cuda=True +task.quantizer=None +task.sub_vec_dim=4 trainer.precision=16 +task.expert_parallel=True \
trainer=gpu_1_host trainer.gpus=1
```

5. Evaluation

You could run evaluation on beir retrieval results using:

```bash
DATASET=(arguana climate-fever dbpedia-entity fever fiqa hotpotqa nfcorpus nq quora scifact scidocs trec-covid webis-touche2020)
for dataset in ${DATASET[*]} 
do
    echo $dataset
    QRELS_PATH=/data_path/beir/datasets/${dataset}/dpr-scale/test.tsv
    TREC_PATH=path_to_retrieval_output_dir/retrieval.trec
    python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > /data_path/results/beir/${dataset}/eval_results.txt
done
```


### Other sytle