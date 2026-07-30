DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=1000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased


# dataset prepare
for dataset in ${DATASET[*]}
do
    echo "Preparing dataset=${dataset}"
    python dpr_scale/citadel_scripts/convert_beir_to_dpr_format.py $dataset $BASE_DIR
done