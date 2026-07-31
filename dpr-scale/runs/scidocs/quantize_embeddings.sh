DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=10000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased



# quantize embeddings

for dataset in ${DATASET[*]}
do
    MERGED_BASE=$BASE_DIR/merged_embeddings/${dataset}

    for w in ${pruning_weights[*]}
    do
        echo "Quantizing dataset=$dataset, weight=$w"

        PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/run_quantization.py -m \
        --config-name msmarco_aws.yaml \
        task=multivec task/model=citadel_model \
        +task.ctx_embeddings_dir=$MERGED_BASE/expert_pruned${w} \
        +task.output_dir=$MERGED_BASE/expert_pruned${w}_pq_nbits2 \
        +cls_dim=128 +dim=32 \
        +sub_vec_dim=4 +num_centroids=256 +iter=5 \
        +cuda=True \
        +threshold=$threshold \
        trainer=gpu_1_host trainer.gpus=1
    done
done