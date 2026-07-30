DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=1000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased


# merge embeddings
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=$BASE_DIR/merged_embeddings/${dataset}
    CTX_EMBEDDINGS_DIR=$BASE_DIR/embeddings/${dataset}
    mkdir -p $OUTPUT_DIR
    
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done