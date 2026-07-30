DATASET=(scidocs)
pruning_weights=(0.5 0.7 0.9 1.1 1.3)
threshold=1000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased


# Evaluation

for dataset in ${DATASET[*]} 
do
    for w in ${pruning_weights[*]}
    do
        echo "Evaluating dataset=$dataset, weight=$w"

        RESULTS_DIR=$BASE_DIR/results/${dataset}/pruned_${w}
        mkdir -p $RESULTS_DIR

        QRELS_PATH=$BASE_DIR/datasets/${dataset}/dpr-scale/test.tsv
        TREC_PATH=$BASE_DIR/retrieval/${dataset}/pruned_${w}/retrieval.trec

        python dpr_scale/citadel_scripts/run_beir_eval.py $QRELS_PATH $TREC_PATH > $RESULTS_DIR/eval_results.txt
    done
done