import json

import os
import csv

import resource

# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# export CUDA_VISIBLE_DEVICES=""
# in bash, setting CUDA_VISIBLE_DEVICES=1 to enable
# export CUDA_VISIBLE_DEVICES="0"
from os import listdir
from os.path import isfile, join
import re
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher, IndexerGenerate
from colbert.data import Queries
import numpy as np
import torch
import time
import sys
import copy
import pandas as pd
from pathlib import Path

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric


def get_peak_memory_kb():
    """返回当前进程的峰值内存占用（KB），若不可用则返回 None"""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except AttributeError:
        return None

def get_directory_size(path):
    """递归计算目录总大小（字节）"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                # 处理权限或文件被删除等情况
                pass
    return total

def resolve_checkpoint_path(raw_data_path: str = None):
    candidates = []
    env_checkpoint = os.environ.get("COLBERT_CHECKPOINT_PATH")
    if env_checkpoint:
        candidates.append(env_checkpoint)
    if raw_data_path is not None:
        candidates.append(os.path.join(raw_data_path, 'colbert-pretrain', 'colbertv2.0'))
    candidates.extend([
        "/data1/lijunlin/colbert_new/model/colbertv2.0/",
        "/data1/wuyinjun/colbert_new/model/colbertv2.0/",
        "/data/ali/colbertv2.0",
        "/home/ali/emb-Models/GTE-ModernColBERT-v1",
    ])

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    expected = os.path.join(raw_data_path, 'colbert-pretrain', 'colbertv2.0') if raw_data_path else 'a known checkpoint path'
    raise FileNotFoundError(
        f"Could not find a ColBERT checkpoint. Set COLBERT_CHECKPOINT_PATH or place a checkpoint under {expected}."
    )


def load_precomputed_query_embeddings(input_query_embedding_file: str, query_embedding_len_file: str = None):
    if input_query_embedding_file.endswith(".npy"):
        query_embeddings = np.load(input_query_embedding_file, mmap_mode="r")
        if query_embeddings.ndim == 3:
            return query_embeddings.astype("float32", copy=False)

        if query_embeddings.ndim != 2:
            raise ValueError(
                f"Expected query embeddings in 2D or 3D .npy format, got shape {query_embeddings.shape} "
                f"from {input_query_embedding_file}"
            )

        if query_embedding_len_file is None:
            raise ValueError(
                "A query length file is required when loading flat 2D query embeddings from .npy"
            )

        query_embedding_len = np.load(query_embedding_len_file)
        n_query = len(query_embedding_len)
        if n_query == 0:
            return np.zeros((0, 0, query_embeddings.shape[1]), dtype="float32")

        max_query_len = int(np.max(query_embedding_len))
        min_query_len = int(np.min(query_embedding_len))
        expected_rows = int(np.sum(query_embedding_len))
        if expected_rows > query_embeddings.shape[0]:
            raise ValueError(
                f"Flat query embeddings row count {query_embeddings.shape[0]} is smaller than "
                f"the summed query lengths {expected_rows}"
            )
        query_embeddings = query_embeddings[:expected_rows]

        if min_query_len == max_query_len:
            return query_embeddings.reshape(n_query, max_query_len, query_embeddings.shape[1]).astype("float32", copy=False)

        padded_query_embeddings = np.zeros((n_query, max_query_len, query_embeddings.shape[1]), dtype="float32")
        offset = 0
        for query_id, query_len in enumerate(query_embedding_len):
            query_len = int(query_len)
            padded_query_embeddings[query_id, :query_len, :] = query_embeddings[offset: offset + query_len]
            offset += query_len
        return padded_query_embeddings

    input_query_embedding = torch.load(input_query_embedding_file, map_location="cpu")
    return input_query_embedding.cpu().numpy().astype("float32")


def delete_file_if_exist(dire):
    if os.path.exists(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def get_n_chunk(base_dir: str):
    filename_l = [f for f in listdir(base_dir) if isfile(join(base_dir, f))]

    doclen_patten = r'doclens(.*).npy'
    embedding_patten = r'encoding(.*)_float32.npy'

    match_obj_l = [re.match(embedding_patten, filename) for filename in filename_l]
    match_chunkID_l = np.array([int(_.group(1)) if _ else None for _ in match_obj_l])
    match_chunkID_l = match_chunkID_l[match_chunkID_l != np.array(None)]
    assert len(match_chunkID_l) == np.sort(match_chunkID_l)[-1] + 1
    return len(match_chunkID_l)


def build_index_official(username: str, 
                         dataset: str,

                         num_partitions_override:int,          # 直接指定中心数
                         num_partitions_multiplier:int,           # 将 16 替换为 24
                         kmeans_sample_multiplier:int,
                         typical_doclen:int,

                         subdir: str = "",

                         embedding_folder: str = None, 
                         input_query_embedding_file: str = None, 
                         gt_file:str=None, 
                         doc_count_file:str=None, 
                         datasets_with_embeddings:list = ["openai", "mscoco","clerc","clip-multi-clustering"], 
                         dataset_dim_mappings={"openai":768, "mscoco":768, "clerc":768, "clip-multi-clustering":768}, 
                         query_embedding_len_file= None):
    
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}')
    os.makedirs(document_data_path, exist_ok=True)
    document_data_path = os.path.join(document_data_path, f'document')
    os.makedirs(document_data_path, exist_ok=True)
    dim=128
    if dataset not in datasets_with_embeddings:
        has_embedding=False
        embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}' 
        os.makedirs(embedding_path, exist_ok=True)
    else:
        has_embedding = True
        embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
        os.makedirs(embedding_path, exist_ok=True)
        dim=dataset_dim_mappings[dataset]
        source = Path(embedding_folder)
        link_name = Path(os.path.join(document_data_path,'transformed_embeddings'))
        if not link_name.exists():
            link_name.symlink_to(source)
        source_gt_file = Path(gt_file)
        link_gt_file_name = Path(os.path.join(document_data_path, 'queries.gnd.jsonl'))
        if not link_gt_file_name.exists():
            link_gt_file_name.symlink_to(source_gt_file)
            
        doc_count = torch.load(doc_count_file)
        doc_ids_ls = [[idx] for idx in range(doc_count)]
        with open(os.path.join(document_data_path, "collection.tsv"), 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(doc_ids_ls)
            
    if query_embedding_len_file is not None:
        print("save query embedding length")
        query_embedding_len = np.load(query_embedding_len_file)
        np.save(os.path.join(embedding_path, "query_n_vec_length.npy"), query_embedding_len)
        
        # os.makedirs(, exist_ok=True)

    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')

    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    index_path = os.path.join(index_path, 'plaid',subdir)
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'

    # n_gpu = torch.cuda.device_count()
    # # torch.set_num_threads(12)
    # print(f'# gpu {n_gpu}')
    
    is_gpu_avail = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count() if is_gpu_avail else 0
    nranks = n_gpu if n_gpu > 0 else 1 # 在CPU环境下，确保至少有1个进程

    print(f'# gpu {n_gpu}, nranks set to {nranks}')

    delete_file_if_exist(embedding_path)
    os.makedirs(base_embedding_path, exist_ok=False)
    with Run().context(
            RunConfig(nranks=nranks, experiment=dataset, root=os.path.join(colbert_project_path, 'experiments'))):
        config = ColBERTConfig(
            nbits=2,
            root=colbert_project_path,
            dim = dim,

            # new !!!
            num_partitions_override=num_partitions_override,          # 直接指定中心数
            num_partitions_multiplier=num_partitions_multiplier,           # 将 16 替换为 24
            kmeans_sample_multiplier = kmeans_sample_multiplier,
            typical_doclen= typical_doclen
            # new !!!
        )
        print(config)
        indexer = Indexer(checkpoint=pretrain_index_path, config=config)
        build_index_time, encode_passage_time = indexer.index(name=dataset,
                                                              collection=os.path.join(document_data_path,
                                                                                      'collection.tsv') if not has_embedding else os.path.join(document_data_path,
                                                                                      'transformed_embeddings'),
                                                              embedding_filename=base_embedding_path,
                                                              overwrite=True)
        
    index_origin_path = os.path.join(colbert_project_path, f'experiments/{dataset}/indexes/{dataset}')
    delete_file_if_exist(index_path)
    os.makedirs(index_path, exist_ok=False)
    # os.makedirs(index_path, exist_ok=True)
    index_new_path = index_path
    os.system(f'mv {index_origin_path}/* {index_new_path}/')
    # os.system(f'mv {index_origin_path} {index_new_path}')    

    n_chunk = get_n_chunk(base_embedding_path)
    total_doclens = []
    for chunkID in range(n_chunk):
        doclens = np.load(os.path.join(base_embedding_path, f'doclens{chunkID}.npy'))
        total_doclens = np.append(total_doclens, doclens)
    np.save(os.path.join(embedding_path, 'doclens.npy'), total_doclens)

    print("finish indexing, start searching")

    build_index_json = {'build_index_time (s)': build_index_time, 
                        'encode_passage_time (s)': encode_passage_time}

    # ---- 新增：收集性能指标 ----
    # 峰值内存（整个进程构建期间）
    peak_mem_kb = get_peak_memory_kb()
    
    # 索引大小（移动后的最终索引目录）
    index_size_bytes = get_directory_size(index_new_path)
    
    # 更新 JSON 内容
    build_index_json.update({
        'index_size (B)': index_size_bytes,
        'peak_build_mem (KB)': peak_mem_kb if peak_mem_kb is not None else -1
    })

    performance_save_dir = os.path.join(result_performance_path, dataset,'plaid',subdir)
    os.makedirs(performance_save_dir, exist_ok=True)

    # with open(os.path.join(result_performance_path, f'{dataset}-build_index-plaid-.json'), 'w') as f:
    with open(os.path.join(performance_save_dir,'build_index.json'), 'w') as f:
        json.dump(build_index_json, f)

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            # collection=os.path.join(document_data_path, 'collection.tsv')
            collection = os.path.join(document_data_path,'collection.tsv') 
            if not has_embedding else os.path.join(document_data_path,
                'transformed_embeddings')
        )
        
        # build and write queries.dev.tsv (第一列是query id，后面会用到)
        query_tsv_path=os.path.join(document_data_path, 'queries.dev.tsv')
        
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        if input_query_embedding_file is None:
            queries = Queries(
                path=os.path.join(document_data_path, 'queries.dev.tsv'))
            total_encode_time_ms, n_encode_query = searcher.save_query_embedding(queries, query_embedding_filename)
        else:
            total_encode_time_ms = 0
            numpy_32 = load_precomputed_query_embeddings(
                input_query_embedding_file=input_query_embedding_file,
                query_embedding_len_file=query_embedding_len_file,
            )
            print("before save")
            np.save(query_embedding_filename, numpy_32)
            n_encode_query = len(numpy_32)
        # topk = 100

        # ranking = searcher.search_all_embedding(query_embedding_filename, k=topk)
        # ranking.save(f"{dataset}_self_search_method_top{topk}.tsv")

        # ranking = searcher.search_all(queries, k=topk)
        # ranking.save(f"{dataset}_official_search_method_top{topk}.tsv")

    encode_info = {'total_encode_time_ms': total_encode_time_ms, 'n_encode_query': n_encode_query,
                   'average_encode_time_ms': total_encode_time_ms / n_encode_query}


    # with open(os.path.join(result_performance_path, f'{dataset}-encode_query.json'), 'w') as f:
    with open(os.path.join(performance_save_dir, 'encode_query.json'), 'w') as f:
        json.dump(encode_info, f)


    if query_embedding_len_file is not None:
        print("save query embedding length")
        query_embedding_len = np.load(query_embedding_len_file)
        np.save(os.path.join(embedding_path, "query_n_vec_length.npy"), query_embedding_len)



def encode_query_cpu(username: str, dataset: str):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')
    print("finish indexing, start searching")

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        total_encode_time_ms, n_encode_query = searcher.save_query_embedding_cpu(queries, query_embedding_filename)
        # topk = 100

        # ranking = searcher.search_all_embedding(query_embedding_filename, k=topk)
        # ranking.save(f"{dataset}_self_search_method_top{topk}.tsv")

        # ranking = searcher.search_all(queries, k=topk)
        # ranking.save(f"{dataset}_official_search_method_top{topk}.tsv")

    encode_info = {'total_encode_time_ms': total_encode_time_ms, 'n_encode_query': n_encode_query,
                   "cpu": True,
                   'average_encode_time_ms': total_encode_time_ms / n_encode_query}
    with open(os.path.join(result_performance_path, f'{dataset}-encode_query.json'), 'w') as f:
        json.dump(encode_info, f)


def build_index_generate(username: str, dataset: str):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'

    n_gpu = torch.cuda.device_count()
    # torch.set_num_threads(12)
    print(f'# gpu {n_gpu}')

    indexer = IndexerGenerate()
    build_index_time, encode_passage_time = indexer.index(username=username, dataset=dataset)

    build_index_json = {'build_index_time (s)': build_index_time, 'encode_passage_time (s)': encode_passage_time}
    with open(os.path.join(result_performance_path, f'{dataset}-build_index-plaid-.json'), 'w') as f:
        json.dump(build_index_json, f)



def load_training_query(username: str, dataset: str, n_sample_query: int):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'

    n_gpu = torch.cuda.device_count()
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.train.tsv'))
        train_query = searcher.get_query_embedding(queries)
        del searcher, queries
    return train_query


def load_dev_query(username: str, dataset: str, n_sample_query: int):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    base_embedding_path = os.path.join(embedding_path, 'base_embedding')
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'

    n_gpu = torch.cuda.device_count()
    print(f'# gpu {n_gpu}')

    index_new_path = os.path.join(index_path, 'plaid')

    with Run().context(
            RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
        config = ColBERTConfig(
            root=colbert_project_path,
            collection=os.path.join(document_data_path, 'collection.tsv')
        )
        searcher = Searcher(checkpoint=pretrain_index_path,
                            index=index_new_path,
                            config=config)
        queries = Queries(
            path=os.path.join(document_data_path, 'queries.dev.tsv'))
        train_query = searcher.get_query_embedding(queries)
        del searcher, queries
    return train_query


def  retrieval_official(username: str, 
                        dataset: str, 
                        topk: int, 
                        search_config_l: list,
                        subdir: str = ""):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'

    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')

    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'

    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'
    result_answer_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/answer'

    answer_dir = os.path.join(result_answer_path, dataset, 'plaid', subdir) 
    perf_dir = os.path.join(result_performance_path, dataset, 'plaid', subdir) 
    os.makedirs(answer_dir, exist_ok=True)
    os.makedirs(perf_dir, exist_ok=True)

    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')
    nranks = n_gpu if n_gpu > 0 else 1

    # mrr_gnd, success_gnd, recall_gnd_id_m = performance_metric.load_groundtruth(username=username, dataset=dataset,
                                                                                # topk=topk)

    index_new_path = os.path.join(index_path, 'plaid', subdir)
    module_name = 'plaid' 

    query_emb = np.load(query_embedding_filename)
    n_query = len(query_emb)
    final_result_l = []
    for search_config in search_config_l:
        print(f"plaid topk {topk}, search config {search_config}")
        torch.set_num_threads(search_config['n_thread'])
        with Run().context(
                RunConfig(nranks=nranks, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
                **colbert_retrieval_config
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_new_path,
                                config=config)

            
            with open(query_text_filename, 'w') as f:
                for i in range(n_query):
                    f.write(f"{i}\n")
                    qid_l = []
                    
            with open(query_text_filename, 'r') as f:
                for line in f:
                    query_text_l = line.split('\t')
                    qid_l.append(int(query_text_l[0]))
                assert len(qid_l) == n_query

            ranking, retrieval_time_l, time_ivf_l, time_filter_l, time_refine_l, n_refine_ivf_l, n_refine_filter_l, n_vec_score_refine_l = searcher.search_all_embedding_by_vector(
                query_emb=query_emb,
                query_embd_filename=query_embedding_filename,
                qid_l=qid_l,
                k=topk)

        time_ms_l = np.around(retrieval_time_l, 3)

        build_index_suffix = ''
        para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
        retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                           f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
        # ranking.save_absolute_path(
            # os.path.join(result_answer_path,
              #           f'{dataset}-plaid-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'))
        ranking.save_absolute_path(
            os.path.join(answer_dir, f'{dataset}-plaid-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv')
        )

        search_time_m = {
            'total_query_time_ms': '{:.3f}'.format(sum(retrieval_time_l)),
            "retrieval_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
            "retrieval_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
            "retrieval_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
            'average_query_time_ms': '{:.3f}'.format(1.0 * sum(retrieval_time_l) / n_query),
            'average_ivf_time_ms': '{:.3f}'.format(1.0 * sum(time_ivf_l) / n_query),
            'average_filter_time_ms': '{:.3f}'.format(1.0 * sum(time_filter_l) / n_query),
            'average_refine_time_ms': '{:.3f}'.format(1.0 * sum(time_refine_l) / n_query),
            'average_n_refine_ivf': '{:.3f}'.format(np.average(n_refine_ivf_l)),
            'average_n_refine_filter': '{:.3f}'.format(np.average(n_refine_filter_l)),
            'average_n_vec_score_refine': '{:.3f}'.format(np.average(n_vec_score_refine_l)),
        }
        retrieval_config = {
            'ndocs': searcher.config.ndocs,
            'ncells': searcher.config.ncells,
            'centroid_score_threshold': searcher.config.centroid_score_threshold,
            'n_thread': search_config['n_thread']
        }
        # recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
        #     username=username, dataset=dataset, topk=topk,
        #     method_name=module_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
        #     mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)
        retrieval_info_m = {
            'n_query': n_query, 'topk': topk, 'build_index': {},
            'retrieval': retrieval_config,
            'search_time': search_time_m, 
            # 'search_accuracy': search_accuracy_m
        }

        method_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
        result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'
        # performance_filename = os.path.join(result_performance_path, method_performance_name)
        performance_filename = os.path.join(perf_dir, method_performance_name)
        with open(performance_filename, "w") as f:
            json.dump(retrieval_info_m, f)

        # df = pd.DataFrame({'time(ms)': time_ms_l, 'recall': recall_l})
        # df.index.name = 'local_queryID'
        # if mrr_l:
        #     df['mrr'] = mrr_l
        # if success_l:
        #     df['success'] = success_l
        single_query_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.csv'
        result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/single_query_performance'
        single_query_performance_filename = os.path.join(result_performance_path, single_query_performance_name)
        # df.to_csv(single_query_performance_filename, index=True)

        print("#############final result###############")
        print("filename", method_performance_name)
        print("search time", retrieval_info_m['search_time'])
        # print("search accuracy", retrieval_info_m['search_accuracy'])
        print("########################################")

        final_result_l.append({'filename': method_performance_name, 'search_time': retrieval_info_m['search_time'],
                            #    'search_accuracy': retrieval_info_m['search_accuracy']
                               })

    # for final_result in final_result_l:
    #     print("#############final result###############")
    #     print("filename", final_result['filename'])
    #     print("search time", final_result['search_time'])
    #     print("search accuracy", final_result['search_accuracy'])
    #     print("########################################")


def retrieval_end2end_single(username: str, dataset: str, topk: int, search_config_l: list):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
    result_answer_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/answer'
    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    index_new_path = os.path.join(index_path, 'plaid')
    retrieval_suffix_l = []

    n_query = len(np.load(query_embedding_filename))
    for search_config in search_config_l:
        print(f"plaid topk {topk}, search config {search_config}")
        torch.set_num_threads(search_config['n_thread'])
        with Run().context(
                RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
                **colbert_retrieval_config
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_new_path,
                                config=config)
            queries = Queries(
                path=os.path.join(document_data_path, 'queries.dev.tsv'))
            ranking_l, encode_time_l, retrieval_time_l = searcher.search_all_single(
                queries, k=topk)

            search_result_m = {
                'time': {
                    "average_retrieval_time_ms": '{:.3f}'.format(
                        np.average(encode_time_l) + np.average(retrieval_time_l)),
                    "average_encode_time_ms": '{:.3f}'.format(np.average(encode_time_l)),
                    "search_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
                    "search_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
                    "search_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
                    'average_search_time_ms': '{:.3f}'.format(np.average(retrieval_time_l)),
                }, 'config': {
                    'ndocs': searcher.config.ndocs,
                    'ncells': searcher.config.ncells,
                    'centroid_score_threshold': searcher.config.centroid_score_threshold,
                    'n_thread': search_config['n_thread']
                }
            }
            retrieval_info_m = {'n_query': n_query, 'topk': topk, 'search_result': search_result_m}
            para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
            build_index_suffix = ''
            retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                               f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
            output_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
            output_filename = os.path.join(output_path,
                                           f'{dataset}-retrieval-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}-time.json')
            with open(output_filename, 'w') as f:
                json.dump(retrieval_info_m, f)

            answer_str_l = []
            for ranking in ranking_l:
                str_l = ranking.tolist()
                for string in str_l:
                    answer_str_l.append(string)
            index_filename = os.path.join(result_answer_path,
                                          f'{dataset}-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv')

            with open(index_filename, 'w') as f:
                for items in answer_str_l:
                    line = '\t'.join(
                        map(lambda x: str(int(x) if type(x) is torch.Tensor or type(x) is bool else x), items)) + '\n'
                    f.write(line)
                print(f"#> Saved ranking of {n_query} queries and {len(answer_str_l)} lines to {f.name}")

            retrieval_suffix_l.append(retrieval_suffix)

    return retrieval_suffix_l


def retrieval_end2end_batch(username: str, dataset: str, topk: int, search_config_l: list):
    colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
    raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
    pretrain_index_path = resolve_checkpoint_path(raw_data_path)
    document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    query_embedding_filename = os.path.join(embedding_path, 'query_embedding.npy')
    index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}'
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
    result_answer_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/answer'
    query_text_filename = os.path.join(document_data_path, 'queries.dev.tsv')

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    index_new_path = os.path.join(index_path, 'plaid')
    retrieval_suffix_l = []

    n_query = len(np.load(query_embedding_filename))
    for search_config in search_config_l:
        print(f"plaid topk {topk}, search config {search_config}")
        torch.set_num_threads(search_config['n_thread'])
        with Run().context(
                RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):
            colbert_retrieval_config = copy.deepcopy(search_config)
            del colbert_retrieval_config['n_thread']

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
                **colbert_retrieval_config
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_new_path,
                                config=config)
            queries = Queries(
                path=os.path.join(document_data_path, 'queries.dev.tsv'))
            ranking, encode_time, retrieval_time_l = searcher.search_all_batch(
                queries, k=topk)

            search_result_m = {
                'time': {
                    "average_retrieval_time_ms": '{:.3f}'.format(
                        1.0 * encode_time / n_query + np.average(retrieval_time_l)),
                    "average_encode_time_ms": '{:.3f}'.format(1.0 * encode_time / n_query),
                    "search_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 5)),
                    "search_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 50)),
                    "search_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_time_l, 95)),
                    'average_search_time_ms': '{:.3f}'.format(np.average(retrieval_time_l)),
                }, 'config': {
                    'ndocs': searcher.config.ndocs,
                    'ncells': searcher.config.ncells,
                    'centroid_score_threshold': searcher.config.centroid_score_threshold,
                    'n_thread': search_config['n_thread']
                }
            }
            retrieval_info_m = {'n_query': n_query, 'topk': topk, 'search_result': search_result_m}
            para_score_thres = "{:.2f}".format(searcher.config.centroid_score_threshold)
            build_index_suffix = ''
            retrieval_suffix = f'ndocs_{searcher.config.ndocs}-ncells_{searcher.config.ncells}-' \
                               f'centroid_score_threshold_{para_score_thres}-n_thread_{search_config["n_thread"]}'
            output_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
            output_filename = os.path.join(output_path,
                                           f'{dataset}-retrieval-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}-time.json')
            with open(output_filename, 'w') as f:
                json.dump(retrieval_info_m, f)

            ranking.save_absolute_path(
                os.path.join(result_answer_path,
                             f'{dataset}-Plaid-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv'))
            retrieval_suffix_l.append(retrieval_suffix)

    return retrieval_suffix_l


if __name__ == '__main__':
    username = 'username1'
    dataset = 'lotte-small'
    build_index_official(username=username, dataset=dataset)
