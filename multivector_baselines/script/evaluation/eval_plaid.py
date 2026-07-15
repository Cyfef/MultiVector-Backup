import logging
import numpy as np
import os
import sys
import json

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
from script.evaluation import performance_metric
from baseline.ColBERT import run as colbert_run
# from script import util
import argparse

def retrieval(username: str, dataset: str, retrieval_parameter_l: list, topk: int):
    logging.info(f"plaid retrieval start {dataset}")
    colbert_run.retrieval_official(username=username, dataset=dataset,
                                   topk=topk, search_config_l=retrieval_parameter_l)
    logging.info(f"plaid retrieval end {dataset}")


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for ndocs in grid_search_para['ndocs']:
        for ncells in grid_search_para['ncells']:
            for centroid_score_threshold in grid_search_para['centroid_score_threshold']:
                for n_thread in grid_search_para['n_thread']:
                    parameter_l.append(
                        {"ndocs": ndocs, "ncells": ncells,
                         "centroid_score_threshold": centroid_score_threshold,
                         "n_thread": n_thread})
    return parameter_l


if __name__ == '__main__':
    # 'ndocs': searcher.config.ndocs,
    # 'ncells': searcher.config.ncells,
    # 'centroid_score_threshold': searcher.config.centroid_score_threshold
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--username', type=str, default="ali")
    parser.add_argument('--dataset', type=str, default="msmarco-large") # clip-multi-clustering clerc_128  scidocs-large msmarco-large
    args = parser.parse_args()
    config_l = {
        'dbg': {
            'username': 'username2',
            # 'dataset_l': ['lotte', 'msmacro'],
            # 'dataset_l': ['lotte-lifestyle', 'lotte', 'msmacro'],
            # 'dataset_l': ['quora'],
            'dataset_l': ['msmacro', 'lotte'],
            'topk_l': [10],
            'retrieval_parameter_l': [
                {"ndocs": 4 * 2000, "ncells": 3, "centroid_score_threshold": 0.5, "n_thread": 1},
            ],
            'grid_search': True,
            'grid_search_para': {
                10: {
                    # 'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400, 4 * 500, 4 * 600, 4 * 700, 4 * 800, 4 * 900, 4 * 1000],
                    # 'ndocs': [4 * 15, 4 * 30, 4 * 60, 4 * 120, 4 * 250, 4 * 500, 4 * 1000],
                    # 'ncells': [1, 2],
                    # 'centroid_score_threshold': [0.5],
                    # 'n_thread': [1]

                    'ndocs': [256],
                    'ncells': [20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000],
                    'centroid_score_threshold': [0.5],
                    'n_thread': [1]
                },
                100: {
                    # 'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400, 4 * 500, 4 * 600, 4 * 700, 4 * 800, 4 * 900, 4 * 1000],
                    'ndocs': [20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000],
                    'ncells': [1, 2, 4],
                    'centroid_score_threshold': [0.45],
                    'n_thread': [1]
                },
                1000: {
                    # 'ndocs': [4 * 100, 4 * 200, 4 * 300, 4 * 400, 4 * 500, 4 * 600, 4 * 700, 4 * 800, 4 * 900, 4 * 1000],
                    'ndocs': [4 * 1000, 4 * 1500, 4 * 2000, 4 * 3500],
                    'ncells': [1, 4, 8],
                    'centroid_score_threshold': [0.4],
                    'n_thread': [1]
                }
            }
        },
        'local': {
            'username': args.username,
            # 'dataset_l': ['lotte-500-gnd'],
            'dataset_l': [args.dataset],
            'topk_l': [10, 100],
            'retrieval_parameter_l': [
                # {'ndocs': 32, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                # {'ndocs': 128, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1},
                # {'ndocs': 512, 'ncells': 64, 'centroid_score_threshold': 0.5, "n_thread": 1}
            ],
            'grid_search': True,
            'grid_search_para': {
                # 10: {
                #     'ndocs': [200,400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000],
                #     'ncells': [2, 4, 8],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [16]
                # },
                # 100: {
                #     'ndocs': [200,400,800,1200,1600,2000,2400,2800,3200,3600,4000,4400,4800,5200,5600,6000,6400,6800,7200,7600,8000],
                #     'ncells': [2, 4, 8],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [16]
                # },
                # 10: {
                #     'ndocs': [20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000,3400,3800,4200,4600,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,24000,26000,28000,30000,34000,38000,42000,46000,50000,55000,60000,65000,70000,75000,80000],
                #     'ncells': [4],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [16]
                # },
                # 100: {
                #     'ndocs': [20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000,3400,3800,4200,4600,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,24000,26000,28000,30000,34000,38000,42000,46000,50000,55000,60000,65000,70000,75000,80000],
                #     'ncells': [4],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [16]
                # },


                # --- final exp test ---
                10: {
                    'ndocs': [10,50,100,200,500,1000,2000,5000,8000,10000],
                    'ncells': [64],
                    'centroid_score_threshold': [0.5],
                    'n_thread': [1]
                },
                100: {
                    'ndocs': [10,50,100,200,500,1000,2000,5000,8000,10000],
                    'ncells': [64],
                    'centroid_score_threshold': [0.5],
                    'n_thread': [1]
                },
                
                # --- small test ---
                # 10: {
                #     'ndocs': [10],
                #     'ncells': [64],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [1]
                # },
                # 10: {
                #     'ndocs': [10000,20000,20],
                #     'ncells': [64],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [1]
                # },
                # 10: {
                #     'ndocs': [50000,80000,100000,150000,200000,250000,300000,350000,400000,1,2,5,8],
                #     'ncells': [2],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [1]
                # },
                # 100: {
                #     'ndocs': [4 * 50, 4 * 100, 4 * 200, 4 * 300, 4 * 500, 4 * 1000, 4 * 2000, 4 * 5000],
                #     'ncells': [4],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [1]
                # },


                # --- final test ---
                # 10: {
                #     'ndocs': [20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5400,5800,6200,6600,7000,7400,7800,8200,8600,9000,9400,9800,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,22000,24000,26000,28000,30000,34000,38000,42000,46000,50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000],
                #     'ncells': [4],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [1]
                # },
                # 100: {
                #     'ndocs': [20,30,40,50,60,70,80,90,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2200,2400,2600,2800,3000,3200,3400,3600,3800,4000,4200,4400,4600,4800,5000,5400,5800,6200,6600,7000,7400,7800,8200,8600,9000,9400,9800,10000,11000,12000,13000,14000,15000,16000,17000,18000,19000,20000,22000,24000,26000,28000,30000,34000,38000,42000,46000,50000,55000,60000,65000,70000,75000,80000,85000,90000,95000,100000],
                #     'ncells': [4],
                #     'centroid_score_threshold': [0.5],
                #     'n_thread': [1]
                # },

                # 10: {
                #     'ndocs': [4 * 10, 4 * 50, 4 * 100, 4 * 200, 4 * 300, 4 * 500],
                #     'ncells': [1, 2, 4, 8],
                #     'centroid_score_threshold': [0.5, 0.4, 0.6],
                #     'n_thread': [1, 8, 16, 32]
                # },
                # 100: {
                #     'ndocs': [4 * 50, 4 * 100, 4 * 200, 4 * 300, 4 * 500, 4 * 1000, 4 * 1500, 4 * 2000, 4 * 5000],
                #     'ncells': [1, 2, 4, 8],
                #     'centroid_score_threshold': [0.5, 0.4, 0.6],
                #     'n_thread': [1, 8, 16, 32]
                # }
            }
        }
    }
    host_name = 'local'

    config = config_l[host_name]
    username = config['username']
    dataset_l = config['dataset_l']
    topk_l = config['topk_l']

    for dataset in dataset_l:
        for topk in topk_l:
            grid_search = config['grid_search']
            if grid_search:
                retrieval_parameter_l = grid_retrieval_parameter(config['grid_search_para'][topk])
            else:
                retrieval_parameter_l = config['retrieval_parameter_l']

            retrieval(username=username, dataset=dataset, retrieval_parameter_l=retrieval_parameter_l, topk=topk)
