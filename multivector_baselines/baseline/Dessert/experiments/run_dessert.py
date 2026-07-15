import dessert_py
import numpy as np
import time

import tqdm

import torch
import os
import json
import sys
import pandas as pd

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric
from script.data import util

ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, os.pardir, 'ColBERT')
sys.path.append(ROOT_PATH)

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer, Searcher
from colbert.data import Queries


def build_index(
        username: str,
        dataset: str,

        num_tables: int,
        hashes_per_table: int = -1,
):
    # ---------------------------------- Parameters --------------------------------
    dessert_index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/dessert'
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    rawdata_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}'
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'

    index_filename = (
        os.path.join(dessert_index_path,
                     f'dessert-{dataset}-n_table_{num_tables}.index')
    )

    if hashes_per_table == -1:
        doclen_l = np.load(os.path.join(embedding_path, f'doclens.npy'))
        hashes_per_table = int(np.ceil(np.log2(np.average(doclen_l))))

    print(
        f"""
        EXPERIMENT: 
            num_tables = {num_tables}
            hashes_per_table = {hashes_per_table}
        """
    )

    # ----------------------- Loading Data and Building Index ----------------------

    centroids = np.load(os.path.join(dessert_index_path, 'centroids.npy'))
    # centroids = np.load(f"{FOLDER}/centroids.npy")

    build_index_start_time = time.time()
    vec_dim = np.load(os.path.join(embedding_path, 'base_embedding', f'encoding0_float32.npy')).shape[1]

    index = dessert_py.DocRetrieval(
        dense_input_dimension=vec_dim,
        num_tables=num_tables,
        hashes_per_table=hashes_per_table,
        centroids=centroids,
    )

    def get_doc_starts_and_ends(doc_lens):
        doc_starts = [0] * len(doc_lens)
        for i in range(1, len(doc_lens)):
            doc_starts[i] = doc_starts[i - 1] + doc_lens[i - 1]
        return [
            (int(start), int(start + length)) for (start, length) in zip(doc_starts, doc_lens)
        ]

    def load(i):
        return np.load(os.path.join(embedding_path, 'base_embedding', f'encoding{i}_float32.npy'))
        # return np.load(f"{FOLDER}/encodings{i}_float32.npy")

    all_centroid_ids = np.load(os.path.join(dessert_index_path, 'all_centroid_ids.npy'))
    # all_centroid_ids = np.load(f"{FOLDER}/all_centroid_ids.npy")
    doclens = np.load(os.path.join(embedding_path, 'doclens.npy'))
    # doclens = np.load(f"{FOLDER}/doclens.npy")
    current_data_file_id = 0
    current_data_file = load(current_data_file_id)
    current_pos_in_array = 0
    doc_offsets = get_doc_starts_and_ends(doclens)

    # if not os.path.exists(index_filename):
    for doc_id, doc_len in tqdm.tqdm(enumerate(doclens), total=len(doclens)):

        end_in_array = min(current_pos_in_array + doc_len, len(current_data_file))
        end_in_array = int(end_in_array)
        embeddings = current_data_file[current_pos_in_array:end_in_array]
        doc_len_left = int(doc_len - end_in_array + current_pos_in_array)
        current_pos_in_array = end_in_array

        # Assumes doc is split over max of 2 files
        if doc_len_left > 0:
            current_data_file_id += 1
            current_data_file = load(current_data_file_id)
            end_in_array = min(
                current_pos_in_array + doc_len, len(current_data_file)
            )
            if len(embeddings) > 0:
                embeddings = np.concatenate(
                    embeddings, current_data_file[:doc_len_left]
                )
            else:
                embeddings = current_data_file[:doc_len_left]
            current_pos_in_array = doc_len_left

        index.add_doc(
            doc_id='ID' + str(doc_id),
            doc_embeddings=embeddings,
            doc_centroid_ids=all_centroid_ids[
                             doc_offsets[doc_id][0]: doc_offsets[doc_id][1]
                             ],
        )
    build_index_time_except_centroid = time.time() - build_index_start_time

    build_index_info_m = {'build_index_time_except_centroid(s)': build_index_time_except_centroid,
                          'hashes_per_table': hashes_per_table}
    index.serialize_to_file(index_filename)

    build_index_filename = os.path.join(result_performance_path,
                                        f"{dataset}-build_index-dessert-n_table_{num_tables}-time.json")

    with open(build_index_filename, "w") as f:
        json.dump(build_index_info_m, f)

    return build_index_info_m


def retrieval(
        username: str,
        dataset: str,

        topk: int,
        num_tables: int,
        retrieval_config_l: list
):
    dessert_index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/dessert'
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    rawdata_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}'
    result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'
    result_answer_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/answer'

    index_filename = (
        os.path.join(dessert_index_path,
                     f'dessert-{dataset}-n_table_{num_tables}.index')
    )

    print(
        f"""
        Index parameters: 
            num_tables = {num_tables}
        """
    )
    # initial_filter_k = {initial_filter_k}
    # nprobe_query = {nprobe_query}
    # remove_centroid_dupes = {remove_centroid_dupes},

    # with open(os.path.join(rawdata_path, 'document', 'queries.dev.tsv')) as f:
    #     # with open(f"{FOLDER}/queries.dev.small.tsv") as f:
    #     qid_map = [int(line.split()[0]) for line in f.readlines()]

    module_name = 'dessert'

    query_embeddings = np.load(os.path.join(embedding_path, 'query_embedding.npy')).astype(np.float32)
    # query_embeddings = np.load(f"{FOLDER}/small_queries_embeddings.npy")

    qid_map = []
    query_text_filename = os.path.join(rawdata_path, f'document/queries.dev.tsv')
    if os.path.exists(query_text_filename):
        with open(query_text_filename, 'r') as f:
            for line in f:
                query_text_l = line.split('\t')
                qid_map.append(int(query_text_l[0]))
            assert len(qid_map) == len(query_embeddings)
    else:
        qid_map = list(range(len(query_embeddings)))
    # qid_map = np.arange(len(query_embeddings))

    centroids = np.load(os.path.join(dessert_index_path, 'centroids.npy'))

    centroids = torch.from_numpy(centroids.transpose()).float()
    final_result_l = []
    for retrieval_config in retrieval_config_l:
        print(f"dessert topk {topk}, search config {retrieval_config}")
        initial_filter_k = 8192 if 'initial_filter_k' not in retrieval_config \
            else retrieval_config['initial_filter_k']
        nprobe_query = 4 if 'nprobe_query' not in retrieval_config \
            else retrieval_config['nprobe_query']
        remove_centroid_dupes = False if 'remove_centroid_dupes' not in retrieval_config \
            else retrieval_config['remove_centroid_dupes']
        n_thread = 1 if 'n_thread' not in retrieval_config \
            else retrieval_config['n_thread']
        torch.set_num_threads(n_thread)

        print(
            f"""
            Search parameters:
                topk = {topk} 
                initial_filter_k = {initial_filter_k}
                nprobe_query = {nprobe_query}
                remove_centroid_dupes = {remove_centroid_dupes},
            """
        )

        assert topk <= initial_filter_k
        all_pids = []
        retrieval_time_l = []
        retrieval_filter_time_l = []
        retrieval_rerank_time_l = []
        n_refine_l = []
        n_score_refine_l = []
        index = dessert_py.DocRetrieval.deserialize_from_file(index_filename)
        for query_id in range(len(query_embeddings)):
            embeddings = query_embeddings[query_id]
            torch_embeddings = torch.from_numpy(embeddings)

            start = time.time_ns()

            start_filter_python_time = time.time_ns()
            if nprobe_query == 1:
                centroid_ids = torch.argmax(
                    torch_embeddings @ centroids, dim=1
                ).tolist()
            else:
                centroid_ids = (
                    torch.topk(torch_embeddings @ centroids, nprobe_query, dim=1)
                    .indices.flatten()
                    .tolist()
                )

            if remove_centroid_dupes:
                centroid_ids = list(set(centroid_ids))
            end_filter_python_time = time.time_ns()
            filter_python_time = (end_filter_python_time - start_filter_python_time) * 1e-9

            # print(time.time() - start)
            results, filter_cpp_time, rerank_time, n_refine, n_score_refine = index.query(
                embeddings,
                num_to_rerank=initial_filter_k,
                top_k=topk,
                query_centroid_ids=centroid_ids,
                n_thread=n_thread
            )
            # print(time.time() - start)

            tmp_topk_time = (time.time_ns() - start) * 1e-9
            all_pids.append([int(r[2:]) for r in results])
            retrieval_time_l.append(tmp_topk_time)
            retrieval_filter_time_l.append(filter_cpp_time + filter_python_time)
            retrieval_rerank_time_l.append(rerank_time)
            n_refine_l.append(n_refine)
            n_score_refine_l.append(n_score_refine)

        print("#####################")
        print(
            f"Query time metrics top{topk}-n_table_{num_tables}-rerank_k_{initial_filter_k}:")
        print(
            f"P50: {format(1.0 * np.percentile(retrieval_time_l, 50) * 1e3, '.1f')}ms, "
            f"P95: {format(1.0 * np.percentile(retrieval_time_l, 95) * 1e3, '.1f')}ms, "
            f"P99: {format(1.0 * np.percentile(retrieval_time_l, 99) * 1e3, '.1f')}ms"
        )
        print(f"Mean: {format(1.0 * np.average(retrieval_time_l) * 1e3, '.1f')}ms", )
        print("#####################")

        retrieval_suffix = f'initial_filter_k_{initial_filter_k}-nprobe_query_{nprobe_query}-' \
                           f'remove_centroid_dupes_{remove_centroid_dupes}-n_thread_{n_thread}'
        build_index_suffix = f'n_table_{num_tables}'

        result_filename = os.path.join(result_answer_path,
                                       f"{dataset}-dessert-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv")

        with open(result_filename, "w") as f:
            for qid_index, r in enumerate(all_pids):
                qid = qid_map[qid_index]
                # score_list = n_score_refine_l[qid_index]
                for rank, pid in enumerate(r):
                    qid = qid_map[qid_index]
                    # score = score_list[rank]
                    f.write(f"{qid}\t{pid}\t{rank + 1}\t1\n")

        build_index_config = {'n_table': num_tables}
        if 'item_n_vec_l' in build_index_config:
            del build_index_config['item_n_vec_l']
        if 'item_n_vecs_l' in build_index_config:
            del build_index_config['item_n_vecs_l']
        retrieval_config = {
            'initial_filter_k': initial_filter_k,
            'nprobe_query': nprobe_query,
            'remove_centroid_dupes': remove_centroid_dupes,
            'n_thread': n_thread,
        }
        search_time_m = {
            'total_query_time_ms': sum(retrieval_time_l) * 1e3,
            "retrieval_time_p5(ms)": 1.0 * np.percentile(retrieval_time_l, 5) * 1e3,
            "retrieval_time_p50(ms)": 1.0 * np.percentile(retrieval_time_l, 50) * 1e3,
            "retrieval_time_p95(ms)": 1.0 * np.percentile(retrieval_time_l, 95) * 1e3,
            "retrieval_filter_time_average(ms)": 1.0 * np.average(retrieval_filter_time_l) * 1e3,
            "retrieval_rerank_time_average(ms)": 1.0 * np.average(retrieval_rerank_time_l) * 1e3,
            'average_query_time_ms': 1.0 * np.average(retrieval_time_l) * 1e3,
            'average_n_refine': 1.0 * np.average(n_refine_l),
            'average_n_score_refine': 1.0 * np.average(n_score_refine_l),
        }
        # recall_l, mrr_l, success_l, search_accuracy_m = performance_metric.count_accuracy(
        #     username=username, dataset=dataset, topk=topk,
        #     method_name=module_name, build_index_suffix=build_index_suffix, retrieval_suffix=retrieval_suffix,
        #     mrr_gnd=mrr_gnd, success_gnd=success_gnd, recall_gnd_id_m=recall_gnd_id_m)
        retrieval_info_m = {'n_query': len(query_embeddings), 'topk': topk,
                            'build_index': build_index_config, 'retrieval': retrieval_config,
                            'search_time': search_time_m, 
                            # 'search_accuracy': search_accuracy_m
                            }
        method_performance_name = f'{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json'
        result_performance_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Result/performance'
        performance_filename = os.path.join(result_performance_path, method_performance_name)
        with open(performance_filename, "w") as f:
            json.dump(retrieval_info_m, f)

        time_ms_l = np.around(np.array(retrieval_time_l) * 1e3, 3)
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
                               }
                              )

    for final_result in final_result_l:
        print("#############final result###############")
        print("filename", final_result['filename'])
        print("search time", final_result['search_time'])
        # print("search accuracy", final_result['search_accuracy'])
        print("########################################")


def retrieval_end_to_end(
        username: str,
        dataset: str,

        topk: int,
        num_tables: int,
        retrieval_config_l: list
):
    dessert_index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/dessert'
    embedding_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}'
    rawdata_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}'
    result_answer_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/answer'

    index_filename = (
        os.path.join(dessert_index_path,
                     f'dessert-{dataset}-n_table_{num_tables}.index')
    )

    print(
        f"""
        Index parameters: 
            num_tables = {num_tables}
        """
    )
    # initial_filter_k = {initial_filter_k}
    # nprobe_query = {nprobe_query}
    # remove_centroid_dupes = {remove_centroid_dupes},

    # with open(os.path.join(rawdata_path, 'document', 'queries.dev.tsv')) as f:
    #     # with open(f"{FOLDER}/queries.dev.small.tsv") as f:
    #     qid_map = [int(line.split()[0]) for line in f.readlines()]

    module_name = 'dessert'

    n_gpu = torch.cuda.device_count()
    is_avail = torch.cuda.is_available()
    print(f'# gpu {n_gpu}, is_avail {is_avail}')

    qid_map = []
    query_str_l = []
    query_text_filename = os.path.join(rawdata_path, f'document/queries.dev.tsv')
    with open(query_text_filename, 'r') as f:
        for line in f:
            query_text_l = line.split('\t')
            qid_map.append(int(query_text_l[0]))
            query_str_l.append(query_text_l[1])

    centroids = np.load(os.path.join(dessert_index_path, 'centroids.npy'))

    centroids = torch.from_numpy(centroids.transpose())
    retrieval_suffix_l = []
    for retrieval_config in retrieval_config_l:
        print(f"dessert topk {topk}, search config {retrieval_config}")
        initial_filter_k = 8192 if 'initial_filter_k' not in retrieval_config \
            else retrieval_config['initial_filter_k']
        nprobe_query = 4 if 'nprobe_query' not in retrieval_config \
            else retrieval_config['nprobe_query']
        remove_centroid_dupes = False if 'remove_centroid_dupes' not in retrieval_config \
            else retrieval_config['remove_centroid_dupes']
        n_thread = 1 if 'n_thread' not in retrieval_config \
            else retrieval_config['n_thread']
        torch.set_num_threads(n_thread)

        print(
            f"""
            Search parameters:
                topk = {topk} 
                initial_filter_k = {initial_filter_k}
                nprobe_query = {nprobe_query}
                remove_centroid_dupes = {remove_centroid_dupes},
            """
        )

        assert topk <= initial_filter_k
        all_pids = []
        retrieval_time_l = []
        retrieval_encode_time_l = []
        retrieval_search_time_l = []
        n_refine_l = []
        index = dessert_py.DocRetrieval.deserialize_from_file(index_filename)

        raw_data_path = f'/data1/{username}/Dataset/multi-vector-retrieval/RawData'
        colbert_project_path = f'/data1/{username}/multi-vector-retrieval/baseline/ColBERT'
        document_data_path = os.path.join(raw_data_path, f'{dataset}/document')
        pretrain_index_path = os.path.join(raw_data_path, 'colbert-pretrain/colbertv2.0')
        index_path = f'/data1/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/plaid'
        with Run().context(
                RunConfig(nranks=n_gpu, experiment=dataset, root=os.path.join(colbert_project_path, 'result'))):

            config = ColBERTConfig(
                root=colbert_project_path,
                collection=os.path.join(document_data_path, 'collection.tsv'),
            )
            searcher = Searcher(checkpoint=pretrain_index_path,
                                index=index_path,
                                config=config)

            for query_id, query_str in enumerate(query_str_l):

                start = time.time_ns()

                start_encode_time = time.time_ns()
                query_emb = np.array(searcher.encode(query_str)[0])
                torch_embeddings = torch.from_numpy(query_emb)
                end_encode_time = time.time_ns()
                encode_time = (end_encode_time - start_encode_time) * 1e-9

                start_filter_python_time = time.time_ns()
                if nprobe_query == 1:
                    centroid_ids = torch.argmax(
                        torch_embeddings @ centroids, dim=1
                    ).tolist()
                else:
                    centroid_ids = (
                        torch.topk(torch_embeddings @ centroids, nprobe_query, dim=1)
                        .indices.flatten()
                        .tolist()
                    )

                if remove_centroid_dupes:
                    centroid_ids = list(set(centroid_ids))
                end_filter_python_time = time.time_ns()
                filter_python_time = (end_filter_python_time - start_filter_python_time) * 1e-9

                # print(time.time() - start)
                results, filter_cpp_time, rerank_time, n_refine = index.query(
                    query_emb,
                    num_to_rerank=initial_filter_k,
                    top_k=topk,
                    query_centroid_ids=centroid_ids,
                    n_thread=n_thread
                )
                # print(time.time() - start)

                tmp_topk_time = (time.time_ns() - start) * 1e-9
                all_pids.append([int(r[2:]) for r in results])
                retrieval_time_l.append(tmp_topk_time)
                retrieval_encode_time_l.append(encode_time)
                retrieval_search_time_l.append(filter_cpp_time + filter_python_time + rerank_time)
                n_refine_l.append(n_refine)

        print("#####################")
        print(
            f"Query time metrics top{topk}-n_table_{num_tables}-rerank_k_{initial_filter_k}:")
        print(
            f"P50: {format(1.0 * np.percentile(retrieval_time_l, 50) * 1e3, '.1f')}ms, "
            f"P95: {format(1.0 * np.percentile(retrieval_time_l, 95) * 1e3, '.1f')}ms, "
            f"P99: {format(1.0 * np.percentile(retrieval_time_l, 99) * 1e3, '.1f')}ms"
        )
        print(f"Mean: {format(1.0 * np.average(retrieval_time_l) * 1e3, '.1f')}ms", )
        print("#####################")

        retrieval_suffix = f'initial_filter_k_{initial_filter_k}-nprobe_query_{nprobe_query}-' \
                           f'remove_centroid_dupes_{remove_centroid_dupes}-n_thread_{n_thread}'
        retrieval_suffix_l.append(retrieval_suffix)
        build_index_suffix = f'n_table_{num_tables}'
        result_filename = os.path.join(result_answer_path,
                                       f"{dataset}-dessert-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv")

        with open(result_filename, "w") as f:
            for qid_index, r in enumerate(all_pids):
                for rank, pid in enumerate(r):
                    qid = qid_map[qid_index]
                    f.write(f"{qid}\t{pid}\t{rank + 1}\n")

        build_index_config = {'n_table': num_tables}
        if 'item_n_vec_l' in build_index_config:
            del build_index_config['item_n_vec_l']
        if 'item_n_vecs_l' in build_index_config:
            del build_index_config['item_n_vecs_l']

        search_result_m = {
            'time': {
                "average_retrieval_time_ms": '{:.3f}'.format(np.average(retrieval_time_l) * 1e3),
                "average_encode_time_ms": '{:.3f}'.format(np.average(retrieval_encode_time_l) * 1e3),
                "search_time_p5(ms)": '{:.3f}'.format(np.percentile(retrieval_search_time_l, 5) * 1e3),
                "search_time_p50(ms)": '{:.3f}'.format(np.percentile(retrieval_search_time_l, 50) * 1e3),
                "search_time_p95(ms)": '{:.3f}'.format(np.percentile(retrieval_search_time_l, 95) * 1e3),
                'average_search_time_ms': '{:.3f}'.format(np.average(retrieval_search_time_l) * 1e3),
            }, 'config': {
                'initial_filter_k': initial_filter_k,
                'nprobe_query': nprobe_query,
                'remove_centroid_dupes': remove_centroid_dupes,
                'n_thread': n_thread,
            }
        }
        retrieval_info_m = {'n_query': len(query_str_l), 'topk': topk, 'search_result': search_result_m}
        output_path = f'/data1/{username}/Dataset/multi-vector-retrieval/end2end/Result/performance'
        output_filename = os.path.join(output_path,
                                       f'{dataset}-retrieval-dessert-end2end-top{topk}-{build_index_suffix}-{retrieval_suffix}-time.json')
        with open(output_filename, 'w') as f:
            json.dump(retrieval_info_m, f)
    return retrieval_suffix_l


if __name__ == "__main__":
    pass
    # do_hyperparameter_search = False
    #
    # if do_hyperparameter_search:
    #     for hashes_per_table in [4, 5, 6, 7, 8]:
    #         for num_tables in [16, 32, 64, 128]:
    #             if (2 ** hashes_per_table) * num_tables > 2048:
    #                 continue
    #             for top_k_return in [1000]:
    #                 for initial_filter_k in [1000, 2048, 4096, 8192, 16384]:
    #                     if top_k_return > initial_filter_k:
    #                         continue
    #                     for nprobe_query in [1, 2, 4]:
    #                         run_experiment(
    #                             top_k_return=top_k_return,
    #                             initial_filter_k=initial_filter_k,
    #                             nprobe_query=nprobe_query,
    #                             remove_centroid_dupes=False,
    #                             hashes_per_table=hashes_per_table,
    #                             num_tables=num_tables,
    #                             use_scann=True,
    #                         )
    # else:
    #
    #     for _ in range(3):
    #         run_experiment(
    #             top_k_return=10,
    #             initial_filter_k=128,
    #             nprobe_query=4,
    #             remove_centroid_dupes=False,
    #             hashes_per_table=6,
    #             num_tables=32,
    #             use_scann=True,
    #         )
