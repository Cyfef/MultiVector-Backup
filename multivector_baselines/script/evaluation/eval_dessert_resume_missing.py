import argparse
import os
import sys


FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'Dessert')
sys.path.append(ROOT_PATH)

from baseline.Dessert import run as dessert_run


class bcolors:
    HEADER = '\033[95m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for n_table in grid_search_para['n_table']:
        for initial_filter_k in grid_search_para['initial_filter_k']:
            for nprobe_query in grid_search_para['nprobe_query']:
                for remove_centroid_dupes in grid_search_para['remove_centroid_dupes']:
                    for n_thread in grid_search_para['n_thread']:
                        parameter_l.append({
                            'n_table': n_table,
                            'initial_filter_k': initial_filter_k,
                            'nprobe_query': nprobe_query,
                            'remove_centroid_dupes': remove_centroid_dupes,
                            'n_thread': n_thread,
                        })
    return parameter_l


def build_index_if_missing(username: str, dataset: str, n_table: int):
    performance_path = f'/data/{username}/Dataset/multi-vector-retrieval/Result/performance'
    dessert_index_file = (
        f'/data/{username}/Dataset/multi-vector-retrieval/Index/{dataset}/dessert/'
        f'dessert-{dataset}-n_table_{n_table}.index'
    )
    build_index_file = os.path.join(
        performance_path,
        f'{dataset}-build_index-dessert-n_table_{n_table}-time.json',
    )
    if os.path.exists(dessert_index_file) and os.path.exists(build_index_file):
        print(
            bcolors.OKGREEN
            + f"dessert build index already exists {dataset} {{'n_table': {n_table}}}, skipping"
            + bcolors.ENDC
        )
        return

    print(
        bcolors.OKGREEN + f"dessert build index start {dataset} {{'n_table': {n_table}}}" + bcolors.ENDC
    )
    dessert_run.build_index(username=username, dataset=dataset, n_table=n_table)
    print(
        bcolors.OKGREEN + f"dessert build index finish {dataset} {{'n_table': {n_table}}}" + bcolors.ENDC
    )


def performance_filename(username: str, dataset: str, topk: int, n_table: int, retrieval_config: dict):
    retrieval_suffix = (
        f"initial_filter_k_{retrieval_config['initial_filter_k']}-"
        f"nprobe_query_{retrieval_config['nprobe_query']}-"
        f"remove_centroid_dupes_{retrieval_config['remove_centroid_dupes']}-"
        f"n_thread_{retrieval_config['n_thread']}"
    )
    return os.path.join(
        f'/data/{username}/Dataset/multi-vector-retrieval/Result/performance',
        f'{dataset}-retrieval-dessert-top{topk}-n_table_{n_table}-{retrieval_suffix}.json',
    )


def retrieval_missing_only(username: str, dataset: str, n_table: int, topk: int, retrieval_config_l: list):
    missing_config_l = []
    for retrieval_config in retrieval_config_l:
        if os.path.exists(performance_filename(username, dataset, topk, n_table, retrieval_config)):
            continue
        missing_config_l.append(retrieval_config)

    if not missing_config_l:
        print(
            bcolors.OKGREEN
            + f"dessert retrieval already complete {dataset} n_table={n_table} topk={topk}, skipping"
            + bcolors.ENDC
        )
        return

    print(
        bcolors.OKGREEN
        + f"dessert retrieval resume start {dataset} n_table={n_table} topk={topk} missing={len(missing_config_l)}"
        + bcolors.ENDC
    )
    dessert_run.retrieval(
        username=username,
        dataset=dataset,
        topk=topk,
        num_tables=n_table,
        retrieval_config_l=missing_config_l,
    )
    print(
        bcolors.OKGREEN
        + f"dessert retrieval resume finish {dataset} n_table={n_table} topk={topk}"
        + bcolors.ENDC
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resume only missing Dessert retrieval configs')
    parser.add_argument('--username', type=str, default='ali')
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    username = args.username
    dataset = args.dataset

    grid_search_para = {
        10: {
            'n_table': [64],
            'initial_filter_k': [10, 50, 100, 500, 1000],
            'nprobe_query': [2],
            'remove_centroid_dupes': [True],
            'n_thread': [1],
        },
        100: {
            'n_table': [64],
            'initial_filter_k': [100, 200, 500, 1000, 2000, 5000, 8000, 10000],
            'nprobe_query': [2, 4],
            'remove_centroid_dupes': [True],
            'n_thread': [1],
        },
    }

    n_table_retrieval_params = {}
    for topk, grid_config in grid_search_para.items():
        for param in grid_retrieval_parameter(grid_config):
            n_table = param['n_table']
            if n_table not in n_table_retrieval_params:
                n_table_retrieval_params[n_table] = {}
            if topk not in n_table_retrieval_params[n_table]:
                n_table_retrieval_params[n_table][topk] = []
            retrieval_param = {k: v for k, v in param.items() if k != 'n_table'}
            n_table_retrieval_params[n_table][topk].append(retrieval_param)

    for n_table in sorted(n_table_retrieval_params.keys()):
        print(bcolors.HEADER + f"\n========== Processing n_table = {n_table} ==========" + bcolors.ENDC)
        build_index_if_missing(username=username, dataset=dataset, n_table=n_table)
        for topk in sorted(n_table_retrieval_params[n_table].keys()):
            retrieval_missing_only(
                username=username,
                dataset=dataset,
                n_table=n_table,
                topk=topk,
                retrieval_config_l=n_table_retrieval_params[n_table][topk],
            )
