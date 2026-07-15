import numpy as np
import os
import sys
import argparse

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
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def build_basic_index(username: str, build_index_config: dict, dataset: str):
    print(
        bcolors.OKGREEN + f"dessert build index start {dataset} {build_index_config}" + bcolors.ENDC)
    dessert_run.build_index(username=username, dataset=dataset, **build_index_config)
    print(
        bcolors.OKGREEN + f"dessert build index finish {dataset} {build_index_config}" + bcolors.ENDC)


def retrieval(username: str, dataset: str, build_index_config: dict, retrieval_config_l: list, topk_l: list):
    print(bcolors.OKGREEN + f" dessert retrieval start {dataset} {build_index_config}" + bcolors.ENDC)
    for topk in topk_l:
        dessert_run.retrieval(username=username, dataset=dataset,
                              topk=topk,
                              retrieval_config_l=retrieval_config_l,
                              **build_index_config)
    print(bcolors.OKGREEN + f" dessert retrieval end {dataset} {build_index_config}" + bcolors.ENDC)


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
                            "nprobe_query": nprobe_query,
                            'remove_centroid_dupes': remove_centroid_dupes, 
                            "n_thread": n_thread
                        })
    return parameter_l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--host_name', type=str, default='local')
    parser.add_argument('--username', type=str, default='ali')
    parser.add_argument('--dataset', type=str, default='msmarco-large') # openai_gte_small openai_gpt2_large clerc scidocs-large msmarco
    args = parser.parse_args()

    config_l = {
        'dbg': {
            'username': 'username',
            'topk_l': [10, 100, 1000],
            'retrieval_parameter_l': [
                {'initial_filter_k': 32, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 64, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 128, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 256, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 512, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 1024, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 2048, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 4096, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 8192, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 16384, "nprobe_query": 1, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 16384, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 32768, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 65536, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 131072, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
                {'initial_filter_k': 262144, "nprobe_query": 2, 'remove_centroid_dupes': True, "n_thread": 1},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    'n_table': [32, 64, 128, 256, 512, 1024, 4096, 8192],
                    'initial_filter_k': [32, 64, 128, 256, 512, 1024, 2048],
                    'nprobe_query': [1, 2, 4, 8],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
                100: {
                    'n_table': [32, 64, 128, 256, 512, 1024, 4096, 8192],
                    'initial_filter_k': [128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                    'nprobe_query': [1, 2, 4, 8],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
                1000: {
                    'n_table': [32, 64, 128, 256, 512, 1024, 4096, 8192],
                    'initial_filter_k': [1024, 2048, 4096, 8192, 16384],
                    'nprobe_query': [1, 2, 4, 8],
                    'remove_centroid_dupes': [True],
                    'n_thread': [1],
                },
            }
        },
        'local': {
            'username': args.username,
            'topk_l': [10, 100],
            'retrieval_parameter_l': [],
            'grid_search': True,
            'grid_search_para': {
                # 10: {
                #     'n_table': [64, 128, 512, 2048, 8192],
                #     'initial_filter_k': [128, 256, 512, 1024, 4096],
                #     'nprobe_query': [1, 2, 4],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [16],
                # },
                # 100: {
                #     'n_table': [64, 128, 512, 2048, 8192],
                #     'initial_filter_k': [128, 256, 512, 1024, 4096],
                #     'nprobe_query': [1, 2, 4],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [16],
                # },
                
                # --- small test ---
                # 10: {
                #     'n_table': [128],
                #     'initial_filter_k': [128],
                #     'nprobe_query': [2],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [1],
                # },
                # 100: {
                #     'n_table': [128],
                #     'initial_filter_k': [128],
                #     'nprobe_query': [2],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [1],
                # },
                
                # --- final test ---
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
                # 10: {
                #     'n_table': [64],
                #     'initial_filter_k': [10,50,100,500,1000],
                #     'nprobe_query': [2],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [1],
                # },
                # 100: {
                #     'n_table': [128, 2048],
                #     'initial_filter_k': [512, 2048, 4096],
                #     'nprobe_query': [2, 4],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [1],
                # },
                
                # 10: {
                #     'n_table': [64, 128, 512, 2048, 4096, 8192, 16384],
                #     'initial_filter_k': [128, 256, 512, 1024, 4096, 2048, 8192, 16384, 32768, 65536],
                #     'nprobe_query': [1, 2, 4],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [1],
                # },
                # 100: {
                #     'n_table': [64, 128, 512, 2048, 4096, 8192, 16384, 32768],
                #     'initial_filter_k': [512, 1024, 4096,2048, 8192, 16384, 32768, 65536, 131072, 262144],
                #     'nprobe_query': [1, 2, 4],
                #     'remove_centroid_dupes': [True],
                #     'n_thread': [1],
                # },
            }
        }
    }
    
    host_name = args.host_name
    dataset = args.dataset

    config = config_l[host_name]
    username = config['username']
    topk_l = config['topk_l']

    # 创建一个字典来存储每个n_table对应的检索参数
    n_table_retrieval_params = {}
    
    for topk in topk_l:
        grid_search = config['grid_search']
        if grid_search:
            all_params = grid_retrieval_parameter(config['grid_search_para'][topk])
            # 按n_table分组参数
            for param in all_params:
                n_table = param['n_table']
                if n_table not in n_table_retrieval_params:
                    n_table_retrieval_params[n_table] = {}
                if topk not in n_table_retrieval_params[n_table]:
                    n_table_retrieval_params[n_table][topk] = []
                # 创建不包含n_table的检索参数
                retrieval_param = {k: v for k, v in param.items() if k != 'n_table'}
                n_table_retrieval_params[n_table][topk].append(retrieval_param)
    
    # 遍历每个n_table值
    for n_table in sorted(n_table_retrieval_params.keys()):
        print(bcolors.HEADER + f"\n========== Processing n_table = {n_table} ==========" + bcolors.ENDC)
        
        # 为当前n_table构建索引
        build_basic_index(username=username, build_index_config={'n_table': n_table}, dataset=dataset)
        
        # 对每个topk进行检索
        for topk in topk_l:
            if topk in n_table_retrieval_params[n_table]:
                retrieval_parameter_l = n_table_retrieval_params[n_table][topk]
                retrieval(username=username, dataset=dataset, 
                         build_index_config={'n_table': n_table},
                         retrieval_config_l=retrieval_parameter_l, 
                         topk_l=[topk])
