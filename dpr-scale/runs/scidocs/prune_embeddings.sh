DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=10000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased


# prune embeddings

for dataset in ${DATASET[*]}
do
    MERGED_BASE=$BASE_DIR/merged_embeddings/${dataset}

    for w in ${pruning_weights[*]}
    do
        echo "Processing dataset=${dataset}, pruning_weight=${w}"
        
        PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/prune_experts.py \
        $MERGED_BASE/expert \
        $MERGED_BASE \
        $w \
        "0-31000" # index range

        # The output is at ${MERGED_BASE}/expert_pruned${w}
    done
done