DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=1000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased



# generate embeddings
export CUDA_VISIBLE_DEVICES=3

for dataset in ${DATASET[*]} 
do
    echo $dataset
    CTX_EMBEDDINGS_DIR=$BASE_DIR/embeddings/${dataset}
    mkdir -p $CTX_EMBEDDINGS_DIR

    DATA_PATH=$BASE_DIR/datasets/${dataset}/dpr-scale/corpus.tsv

    HYDRA_FULL_ERROR=1 PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/generate_multivec_embeddings.py --config-name msmarco_aws.yaml \
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
    +task.add_context_id=False \
    task.model.model_path="${LOCAL_MODEL_PATH}" \
    task.transform.model_path="${LOCAL_MODEL_PATH}" \
    trainer.max_steps=-1 \
    hydra.sweep.dir="${BASE_DIR}/hydra_outputs/${dataset}" \
    > nohup_${dataset}.log 2>&1
done