DATASET=(fiqa)
pruning_weights=(0.7 1.3 1.5 1.7 1.9)
threshold=10000
BASE_DIR=/data1/chenyifeng/MultiVector-Backup/dpr-scale
CHECKPOINT_PATH=$BASE_DIR/ckpt/checkpoint_best.ckpt
LOCAL_MODEL_PATH=$BASE_DIR/model/hug/bert-base-uncased


# dataset prepare
for dataset in ${DATASET[*]}
do
    echo "Preparing dataset=${dataset}"
    python dpr_scale/citadel_scripts/convert_beir_to_dpr_format.py $dataset $BASE_DIR
done


# generate embeddings
export CUDA_VISIBLE_DEVICES=2

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


# merge embeddings
for dataset in ${DATASET[*]} 
do
    echo $dataset
    OUTPUT_DIR=$BASE_DIR/merged_embeddings/${dataset}
    CTX_EMBEDDINGS_DIR=$BASE_DIR/embeddings/${dataset}
    mkdir -p $OUTPUT_DIR
    
    PYTHONPATH=.:$PYTHONPATH python dpr_scale/citadel_scripts/merge_experts.py $OUTPUT_DIR "$CTX_EMBEDDINGS_DIR" "0-31000"
done


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
        task.model.model_path="${LOCAL_MODEL_PATH}" \
        task.transform.model_path="${LOCAL_MODEL_PATH}" \
        trainer.precision=16 +task.expert_parallel=True \
        trainer=gpu_1_host trainer.gpus=1
    done
done


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

echo "All stages completed."