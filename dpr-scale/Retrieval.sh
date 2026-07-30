DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=1000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased


# retrieval

for dataset in ${DATASET[*]} 
do
    MERGED_BASE=$BASE_DIR/merged_embeddings/${dataset}

    for w in ${pruning_weights[*]}
    do
        echo "Retrieving dataset=$dataset, weight=$w"

        OUTPUT_DIR=$BASE_DIR/retrieval/${dataset}/pruned_${w}
        mkdir -p $OUTPUT_DIR

        CTX_EMBEDDINGS_DIR=$MERGED_BASE/expert_pruned${w}_pq_nbits2
        I2D_PATH=$BASE_DIR/datasets/${dataset}/dpr-scale/index2docid.tsv
        DATA_PATH=$BASE_DIR/datasets/${dataset}/dpr-scale/corpus.tsv
        QUERY_PATH=$BASE_DIR/datasets/${dataset}/dpr-scale/queries.tsv

        PORTION=0.001 # how much portion of the index should be moved to GPU before retrieval

        export pruning_weight=$w

        HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_citadel_retrieval.py \
        --config-name msmarco_aws.yaml \
        datamodule=generate_multivec_query_emb \
        datamodule.test_path=$QUERY_PATH \
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
        +task.topk=100 +task.cuda=True \
        +task.quantizer=pq \
        +task.sub_vec_dim=4 \
        trainer.precision=16 +task.expert_parallel=True \
        trainer=gpu_1_host trainer.gpus=1
    done
done