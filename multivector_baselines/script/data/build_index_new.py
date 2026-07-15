import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'ColBERT')
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, 'baseline', 'Dessert')
sys.path.append(ROOT_PATH)

from baseline.ColBERT import run as colbert_run
from script.data import groundtruth
import util
import argparse

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


if __name__ == '__main__':
    username = 'lijunlin'
    topk_l = [10, 100]
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--dataset', type=str, default="openai_gte_base")
    args = parser.parse_args()
    dataset_l = [
        # 'lotte-500-gnd',
        # 'mscoco',
        # 'openai',
        # 'openai_gte_base',
        # args.dataset,
        # 'openai_gte_large',
        # 'lotte',
        # 'msmacro',
        # 'wikipedia',
        # 'lotte-lifestyle',
        # 'quora',
        # 'hotpotqa',
        # 'wiki-nq',
        
        "clerc"
        # "gte-small-multi"
        # "clip-multi-clustering"
    ]
    
    base_data_dir_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/", \
        "openai":"/data1/wuyinjun/openai1Mresults/", \
        "openai_gte_small":"/data1/ali/openai1Mresult_gte_small/",\
        "openai_gte_base":"/data1/ali/openai1Mresults_gte_base/",\
        "openai_gte_large":"/data1/ali/openai1Mresults_gte_large/",\
        "openai_gpt2_large":"/data1/ali/openai1Mresults_gpt2_large/",\
        "clerc":"",\
        "gte-small-multi":"",\
        "clip-multi-clustering":""
        }
    
    embedding_folder_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/", \
        "openai":"/data1/wuyinjun/openai1Mresults/transformed_embeddings/", \
        "openai_gte_small":os.path.join(base_data_dir_mappings["openai_gte_small"],"docs_embeddings/transformed_embeddings/"),\
        "openai_gte_base":os.path.join(base_data_dir_mappings["openai_gte_base"],"docs_embeddings/transformed_embeddings/"),\
        "openai_gte_large":os.path.join(base_data_dir_mappings["openai_gte_large"],"docs_embeddings/transformed_embeddings/"),\
        "openai_gpt2_large":os.path.join(base_data_dir_mappings["openai_gpt2_large"],"docs_embeddings/transformed_embeddings/"),\
        "clerc":"/data/lijunlin/Dataset/multi-vector-retrieval/transform/clerc/transformed_embeddings",\
        "gte-small-multi":"/data/lijunlin/Dataset/multi-vector-retrieval/transform/gte-small-multi/transformed_embeddings",\
        "clip-multi-clustering":"/data/lijunlin/Dataset/multi-vector-retrieval/transform/clip-multi-clustering/transformed_embeddings"
        }
    
    query_embedding_file_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/query_embeddings.pt", \
         "openai":"/data1/ali/openaiqueryresults/transformed_embeddings/query_embeddings.pt", \
         "openai_gte_small":os.path.join(base_data_dir_mappings["openai_gte_small"],"query_embeddings/transformed_embeddings/query_embeddings.pt"),\
        "openai_gte_base":os.path.join(base_data_dir_mappings["openai_gte_base"],"query_embeddings/transformed_embeddings/query_embeddings.pt"), \
        "openai_gte_large":os.path.join(base_data_dir_mappings["openai_gte_large"],"query_embeddings/transformed_embeddings/query_embeddings.pt"), \
        "openai_gpt2_large":os.path.join(base_data_dir_mappings["openai_gpt2_large"],"query_embeddings/transformed_embeddings/query_embeddings.pt"),\
            "gte-small-multi":"/data/lijunlin/Dataset/ptdata/gte_small/query_embedding/query_embeddings.pt"
         }
    
    query_embedding_len_file_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/query_n_vec_length.npy", \
         "openai":"/data1/ali/openaiqueryresults/transformed_embeddings/query_n_vec_length.npy", \
         "openai_gte_small":os.path.join(base_data_dir_mappings["openai_gte_small"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"),\
        "openai_gte_base":os.path.join(base_data_dir_mappings["openai_gte_base"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"), \
        "openai_gte_large":os.path.join(base_data_dir_mappings["openai_gte_large"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"), \
        "openai_gpt2_large":os.path.join(base_data_dir_mappings["openai_gpt2_large"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"),\
            "gte-small-multi":"/data/lijunlin/Dataset/ptdata/gte_small/query_embedding/query_n_vec_length.npy"
         }
    
    
    dataset_dim_mappings={"mscoco":768, "openai":768, "openai_gte_small":384,"openai_gte_base":768, "openai_gte_large":1024, "openai_gpt2_large":1280, "clerc": 768, "gte-small-multi": 384, "clip-multi-clustering":768}
    
    for dataset in dataset_l:
        
        embedding_folder = None
        query_embedding_file = None
        gt_file = None
        query_embedding_len_file = None
        if dataset in embedding_folder_mappings:
            embedding_folder = embedding_folder_mappings[dataset]
            print(embedding_folder)
            # query_embedding_file = query_embedding_file_mappings[dataset]
            
        print(bcolors.OKGREEN + f"plaid start {dataset}" + bcolors.ENDC)
        colbert_run.build_index_official(username=username, dataset=dataset, embedding_folder=embedding_folder, input_query_embedding_file=query_embedding_file, gt_file=gt_file, datasets_with_embeddings=list(embedding_folder_mappings.keys()), dataset_dim_mappings=dataset_dim_mappings, query_embedding_len_file=query_embedding_len_file)
        print(bcolors.OKGREEN + f"plaid finish {dataset}" + bcolors.ENDC)

        module_name = 'BruteForceProgressive'
        print(bcolors.OKGREEN + f"groundtruth start {dataset}" + bcolors.ENDC)
        util.compile_file(username=username, module_name=module_name, is_debug=True)
        est_dist_l_l, est_id_l_l = groundtruth.gnd_cpp(username=username, dataset=dataset, topk_l=topk_l,
                                                       compile_file=False, module_name=module_name)
        for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
            groundtruth.save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset,
                                     topk=topk)
        print(bcolors.OKGREEN + f"groundtruth end {dataset}" + bcolors.ENDC)
