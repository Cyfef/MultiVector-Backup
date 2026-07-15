import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

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
    topk_l = [10, 100]
    parser = argparse.ArgumentParser(description='argparse')
    parser.add_argument('--username', type=str, default="ali")
    parser.add_argument('--dataset', type=str, default="openai_gte_base")
    parser.add_argument('--run-groundtruth', action='store_true')
    args = parser.parse_args()
    username = args.username
    dataset_l = [args.dataset]
    
    base_data_dir_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/", \
        "openai":"/data1/wuyinjun/openai1Mresults/", \
        "openai_gte_small":"/data1/ali/openai1Mresult_gte_small/",\
        "openai_gte_base":"/data1/ali/openai1Mresults_gte_base/",\
        "openai_gte_large":"/data1/ali/openai1Mresults_gte_large/",\
        "openai_gpt2_large":"/data1/ali/openai1Mresults_gpt2_large/",\
        "clerc":"",\
        "gte-small-multi":"",\
        "clip-multi-clustering":"",\
            
        "clerc_128":"",\
        "clerc_large":"",\
        "clerc_small":"",\
        "multiqa_med":"",\
        "clef_large":"",\
        "clef_small":"",\
        "clef_med":"",\
        "scidocs-large":"",\
        "scidocs-colbert":"/data/ali/scidocs-colbert",\
        "clerc-colbert":"/data/ali/clerc-colbert",\
        "clef-colbert":"/data/ali/clef-colbert",\
        "msmarco-modern-colbert":"/data/ali/msmarco-modern-colbert",\
        "fiqa-modern-colbert":"/data/ali/fiqa-modern-colbert",\
        "clerc-modern-colbert":"/data/ali/clerc-modern-colbert",\
        "clef-modern-colbert":"/data/ali/clef-modern-colbert",\
        "nq-modern-colbert":"/data/ali/nq-modern-colbert",\
        "scidocs-modern-colbert":"/data/ali/scidocs-modern-colbert",\

        }
    
    embedding_folder_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/", \
        "openai":"/data1/wuyinjun/openai1Mresults/transformed_embeddings/", \
        "openai_gte_small":os.path.join(base_data_dir_mappings["openai_gte_small"],"docs_embeddings/transformed_embeddings/"),\
        "openai_gte_base":os.path.join(base_data_dir_mappings["openai_gte_base"],"docs_embeddings/transformed_embeddings/"),\
        "openai_gte_large":os.path.join(base_data_dir_mappings["openai_gte_large"],"docs_embeddings/transformed_embeddings/"),\
        "openai_gpt2_large":os.path.join(base_data_dir_mappings["openai_gpt2_large"],"docs_embeddings/transformed_embeddings/"),\
        "clerc":"/data1/lijunlin/sigmod2025/transform/clerc/doc_embeddings/transformed_embeddings",\
        "gte-small-multi":"/data/lijunlin/Dataset/multi-vector-retrieval/transform/gte-small-multi/transformed_embeddings",\
        "clip-multi-clustering":"/data1/lijunlin/sigmod2025/transform/clip-multi-clustering/doc_embeddings/transformed_embeddings",\
        
        "clerc_large":"/data1/lijunlin/sigmod2025/transform/clerc_large/doc_embeddings/transformed_embeddings",\
        "clerc_small":"/data1/lijunlin/sigmod2025/transform/clerc_small/doc_embeddings/transformed_embeddings",\
        "multiqa_med":"/data1/lijunlin/sigmod2025/transform/multiqa_med/doc_embeddings/transformed_embeddings",\
        "clef_large":"/data1/lijunlin/sigmod2025/transform/clef_large/doc_embeddings/transformed_embeddings",\
        "clef_small":"/data1/lijunlin/sigmod2025/transform/clef_small/doc_embeddings/transformed_embeddings",\
        "clef_med":"/data1/lijunlin/sigmod2025/transform/clef_med/doc_embeddings/transformed_embeddings",\
            
        "clerc_128":"/data2/lijunlin/sigmod2026-r4/sigmod-2025-backup-47_94/transform/clerc_128/doc_embeddings/transformed_embeddings",\
        "scidocs-large":"/data1/lijunlin/raw/transform/scidocs-large/doc_embeddings/transformed_embeddings",\
        "msmarco-large":"/data1/lijunlin/raw/transform/msmarco-large/doc_embeddings/transformed_embeddings",\
        "scidocs-colbert":"/data/ali/scidocs-colbert/doc_embeddings/transformed_embeddings",\
        "clerc-colbert":"/data/ali/clerc-colbert/doc_embeddings/transformed_embeddings",\
        "clef-colbert":"/data/ali/clef-colbert/doc_embeddings/transformed_embeddings",\
        "msmarco-modern-colbert":"/data/ali/msmarco-modern-colbert/doc_embeddings/transformed_embeddings",\
        "fiqa-modern-colbert":"/data/ali/fiqa-modern-colbert/doc_embeddings/transformed_embeddings",\
        "clerc-modern-colbert":"/data/ali/clerc-modern-colbert/doc_embeddings/transformed_embeddings",\
        "clef-modern-colbert":"/data/ali/clef-modern-colbert/doc_embeddings/transformed_embeddings",\
        "nq-modern-colbert":"/data/ali/nq-modern-colbert/doc_embeddings/transformed_embeddings",\
        "scidocs-modern-colbert":"/data/ali/scidocs-modern-colbert/doc_embeddings/transformed_embeddings",\
        }
    query_embedding_file_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/query_embeddings.pt", \
         "openai":"/data1/ali/openaiqueryresults/transformed_embeddings/query_embeddings.pt", \
         "openai_gte_small":os.path.join(base_data_dir_mappings["openai_gte_small"],"query_embeddings/transformed_embeddings/query_embeddings.pt"),\
        "openai_gte_base":os.path.join(base_data_dir_mappings["openai_gte_base"],"query_embeddings/transformed_embeddings/query_embeddings.pt"), \
        "openai_gte_large":os.path.join(base_data_dir_mappings["openai_gte_large"],"query_embeddings/transformed_embeddings/query_embeddings.pt"), \
        "openai_gpt2_large":os.path.join(base_data_dir_mappings["openai_gpt2_large"],"query_embeddings/transformed_embeddings/query_embeddings.pt"),\
        "gte-small-multi":"/data/lijunlin/Dataset/ptdata/gte_small/query_embedding/query_embeddings.pt",\
        "clerc":"/data1/lijunlin/sigmod2025/transform/clerc/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "clip-multi-clustering":"/data1/lijunlin/sigmod2025/transform/clip-multi-clustering/query_embeddings/transformed_embeddings/query_embeddings.pt",\
            
        "clerc_large":"/data1/lijunlin/sigmod2025/transform/clerc_large/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "clerc_small":"/data1/lijunlin/sigmod2025/transform/clerc_small/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "multiqa_med":"/data1/lijunlin/sigmod2025/transform/multiqa_med/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "clef_large":"/data1/lijunlin/sigmod2025/transform/clef_large/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "clef_small":"/data1/lijunlin/sigmod2025/transform/clef_small/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "clef_med":"/data1/lijunlin/sigmod2025/transform/clef_med/query_embeddings/transformed_embeddings/query_embeddings.pt",\
            
        "clerc_128":"/data2/lijunlin/sigmod2026-r4/sigmod-2025-backup-47_94/transform/clerc_128/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "scidocs-large":"/data1/lijunlin/raw/transform/scidocs-large/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "msmarco-large":"/data1/lijunlin/raw/transform/msmarco-large/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "scidocs-colbert":"/data/ali/scidocs-colbert/query_embeddings/transformed_embeddings/query_embeddings.pt",\
        "clerc-colbert":"/data/ali/clerc-colbert/full_multi_embeddings_clerc_colbert_query.npy",\
        "clef-colbert":"/data/ali/clef-colbert/full_multi_embeddings_clef_colbert_query.npy",\
        "msmarco-modern-colbert":"/data/ali/msmarco-modern-colbert/query_points.npy",\
        "fiqa-modern-colbert":"/data/ali/fiqa-modern-colbert/query_points.npy",\
        "clerc-modern-colbert":"/data/ali/clerc-modern-colbert/query_points.npy",\
        "clef-modern-colbert":"/data/ali/clef-modern-colbert/query_points.npy",\
        "nq-modern-colbert":"/data/ali/nq-modern-colbert/query_points.npy",\
        "scidocs-modern-colbert":"/data/ali/scidocs-modern-colbert/query_points.npy",\
         }
    
    query_embedding_len_file_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/query_n_vec_length.npy", \
         "openai":"/data1/ali/openaiqueryresults/transformed_embeddings/query_n_vec_length.npy", \
         "openai_gte_small":os.path.join(base_data_dir_mappings["openai_gte_small"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"),\
        "openai_gte_base":os.path.join(base_data_dir_mappings["openai_gte_base"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"), \
        "openai_gte_large":os.path.join(base_data_dir_mappings["openai_gte_large"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"), \
        "openai_gpt2_large":os.path.join(base_data_dir_mappings["openai_gpt2_large"],"query_embeddings/transformed_embeddings/query_n_vec_length.npy"),\
            "gte-small-multi":"/data/lijunlin/Dataset/ptdata/gte_small/query_embedding/query_n_vec_length.npy",\
            "clerc":"/data1/lijunlin/sigmod2025/transform/clerc/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
                "clip-multi-clustering":"/data1/lijunlin/sigmod2025/transform/clip-multi-clustering/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        
        "clerc_large":"/data1/lijunlin/sigmod2025/transform/clerc_large/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "clerc_small":"/data1/lijunlin/sigmod2025/transform/clerc_small/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "multiqa_med":"/data1/lijunlin/sigmod2025/transform/multiqa_med/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "clef_large":"/data1/lijunlin/sigmod2025/transform/clef_large/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "clef_small":"/data1/lijunlin/sigmod2025/transform/clef_small/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "clef_med":"/data1/lijunlin/sigmod2025/transform/clef_med/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
            
        "clerc_128":"/data2/lijunlin/sigmod2026-r4/sigmod-2025-backup-47_94/transform/clerc_128/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "scidocs-large":"/data1/lijunlin/raw/transform/scidocs-large/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "msmarco-large":"/data1/lijunlin/raw/transform/msmarco-large/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "scidocs-colbert":"/data/ali/scidocs-colbert/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "clerc-colbert":"/data/ali/clerc-colbert/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "clef-colbert":"/data/ali/clef-colbert/query_embeddings/transformed_embeddings/query_n_vec_length.npy",\
        "msmarco-modern-colbert":"/data/ali/msmarco-modern-colbert/plaid_query_lens.npy",\
        "fiqa-modern-colbert":"/data/ali/fiqa-modern-colbert/plaid_query_lens.npy",\
        "clerc-modern-colbert":"/data/ali/clerc-modern-colbert/plaid_query_lens.npy",\
        "clef-modern-colbert":"/data/ali/clef-modern-colbert/plaid_query_lens.npy",\
        "nq-modern-colbert":"/data/ali/nq-modern-colbert/plaid_query_lens.npy",\
        "scidocs-modern-colbert":"/data/ali/scidocs-modern-colbert/plaid_query_lens.npy",\
         }
    
    gt_file_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/gnd.tsv", \
        "openai":"/data1/ali/openaiqueryresults/transformed_embeddings/query_doc_mappings.jsonl",\
        "openai_gte_small": os.path.join(base_data_dir_mappings["openai_gte_small"],"query_embeddings/transformed_embeddings/query_doc_mappings.jsonl"),\
        "openai_gte_base": os.path.join(base_data_dir_mappings["openai_gte_base"],"query_embeddings/transformed_embeddings/query_doc_mappings.jsonl"),\
        "openai_gte_large": os.path.join(base_data_dir_mappings["openai_gte_large"],"query_embeddings/transformed_embeddings/query_doc_mappings.jsonl"),\
        "openai_gpt2_large": os.path.join(base_data_dir_mappings["openai_gpt2_large"],"query_embeddings/transformed_embeddings/query_doc_mappings.jsonl"),\
            "gte-small-multi":"/data/lijunlin/Dataset/ptdata/gte_small/query_embedding/query_doc_mappings.jsonl",\
                "clerc":"/data1/lijunlin/sigmod2025/transform/clerc/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
                    "clip-multi-clustering":"/data1/lijunlin/sigmod2025/transform/clip-multi-clustering/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
            
        "clerc_large":"/data1/lijunlin/sigmod2025/transform/clerc_large/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clerc_small":"/data1/lijunlin/sigmod2025/transform/clerc_small/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "multiqa_med":"/data1/lijunlin/sigmod2025/transform/clmultiqa_mederc/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clef_large":"/data1/lijunlin/sigmod2025/transform/clef_large/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clef_small":"/data1/lijunlin/sigmod2025/transform/clef_small/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clef_med":"/data1/lijunlin/sigmod2025/transform/clef_med/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
            
        "clerc_128":"/data2/lijunlin/sigmod2026-r4/sigmod-2025-backup-47_94/transform/clerc_128/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "scidocs-large":"/data1/lijunlin/raw/transform/scidocs-large/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "msmarco-large":"/data1/lijunlin/raw/transform/msmarco-large/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "scidocs-colbert":"/data/ali/scidocs-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clerc-colbert":"/data/ali/clerc-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clef-colbert":"/data/ali/clef-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "msmarco-modern-colbert":"/data/ali/msmarco-modern-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "fiqa-modern-colbert":"/data/ali/fiqa-modern-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clerc-modern-colbert":"/data/ali/clerc-modern-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "clef-modern-colbert":"/data/ali/clef-modern-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "nq-modern-colbert":"/data/ali/nq-modern-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        "scidocs-modern-colbert":"/data/ali/scidocs-modern-colbert/query_embeddings/transformed_embeddings/query_doc_mappings.jsonl",\
        }
        
    doclen_file_mappings = \
        {"mscoco":"/data1/coco/coco2017/embeddings/transformed_embeddings/doc_count", \
        "openai":"/data1/wuyinjun/openai1Mresults/transformed_embeddings/doc_count",\
        "openai_gte_small": os.path.join(base_data_dir_mappings["openai_gte_small"],"docs_embeddings/transformed_embeddings/doc_count"),\
        "openai_gte_base": os.path.join(base_data_dir_mappings["openai_gte_base"],"docs_embeddings/transformed_embeddings/doc_count"),\
        "openai_gte_large": os.path.join(base_data_dir_mappings["openai_gte_large"],"docs_embeddings/transformed_embeddings/doc_count"),\
        "openai_gpt2_large": os.path.join(base_data_dir_mappings["openai_gpt2_large"],"docs_embeddings/transformed_embeddings/doc_count"),\
            "gte-small-multi":"/data/lijunlin/Dataset/ptdata/gte_small/doc_embedding/doc_count",\
                "clerc":"/data1/lijunlin/sigmod2025/transform/clerc/doc_embeddings/transformed_embeddings/doc_count",\
                    "clip-multi-clustering":"/data1/lijunlin/sigmod2025/transform/clip-multi-clustering/doc_embeddings/transformed_embeddings/doc_count",\
        
        "clerc_large":"/data1/lijunlin/sigmod2025/transform/clerc_large/doc_embeddings/transformed_embeddings/doc_count",\
        "clerc_small":"/data1/lijunlin/sigmod2025/transform/clerc_small/doc_embeddings/transformed_embeddings/doc_count",\
        "multiqa_med":"/data1/lijunlin/sigmod2025/transform/multiqa_med/doc_embeddings/transformed_embeddings/doc_count",\
        "clef_large":"/data1/lijunlin/sigmod2025/transform/clef_large/doc_embeddings/transformed_embeddings/doc_count",\
        "clef_small":"/data1/lijunlin/sigmod2025/transform/clef_small/doc_embeddings/transformed_embeddings/doc_count",\
        "clef_med":"/data1/lijunlin/sigmod2025/transform/clef_med/doc_embeddings/transformed_embeddings/doc_count",\
            
        "clerc_128":"/data2/lijunlin/sigmod2026-r4/sigmod-2025-backup-47_94/transform/clerc_128/doc_embeddings/transformed_embeddings/doc_count",\
        "scidocs-large":"/data1/lijunlin/raw/transform/scidocs-large/doc_embeddings/transformed_embeddings/doc_count",\
        "msmarco-large":"/data1/lijunlin/raw/transform/msmarco-large/doc_embeddings/transformed_embeddings/doc_count",\
        "scidocs-colbert":"/data/ali/scidocs-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "clerc-colbert":"/data/ali/clerc-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "clef-colbert":"/data/ali/clef-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "msmarco-modern-colbert":"/data/ali/msmarco-modern-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "fiqa-modern-colbert":"/data/ali/fiqa-modern-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "clerc-modern-colbert":"/data/ali/clerc-modern-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "clef-modern-colbert":"/data/ali/clef-modern-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "nq-modern-colbert":"/data/ali/nq-modern-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        "scidocs-modern-colbert":"/data/ali/scidocs-modern-colbert/doc_embeddings/transformed_embeddings/doc_count",\
        } 
    
    dataset_dim_mappings={"mscoco":768, "openai":768, "openai_gte_small":384,"openai_gte_base":768, "openai_gte_large":1024, "openai_gpt2_large":1280, "gte-small-multi":384, "clerc":768, "clip-multi-clustering": 768,\
        "clerc_large":1024,\
        "clerc_small":384,\
        "multiqa_med":768,\
        "clef_large":1024,\
        "clef_small":384,\
            
        "clerc_128":128,\
        "clef_med":768,\
        "scidocs-large":1024,\
        "msmarco-large":1024,\
        "scidocs-colbert":128,\
        "clerc-colbert":128,\
        "clef-colbert":128,\
        "msmarco-modern-colbert":128,\
        "fiqa-modern-colbert":128,\
        "clerc-modern-colbert":128,\
        "clef-modern-colbert":128,\
        "nq-modern-colbert":128,\
        "scidocs-modern-colbert":128,\
        }
    
    for dataset in dataset_l:
        
        embedding_folder = None
        query_embedding_file = None
        gt_file = None
        doc_count_file = None
        query_embedding_len_file = None
        if dataset in embedding_folder_mappings:
            embedding_folder = embedding_folder_mappings[dataset]
            query_embedding_file = query_embedding_file_mappings[dataset]
            gt_file = gt_file_mappings[dataset]
            doc_count_file = doclen_file_mappings[dataset]
            query_embedding_len_file = query_embedding_len_file_mappings[dataset]
            
        print(bcolors.OKGREEN + f"plaid start {dataset}" + bcolors.ENDC)
        colbert_run.build_index_official(username=username, dataset=dataset, embedding_folder=embedding_folder, input_query_embedding_file=query_embedding_file, gt_file=gt_file, doc_count_file=doc_count_file, datasets_with_embeddings=list(embedding_folder_mappings.keys()), dataset_dim_mappings=dataset_dim_mappings, query_embedding_len_file=query_embedding_len_file)
        print(bcolors.OKGREEN + f"plaid finish {dataset}" + bcolors.ENDC)

        if args.run_groundtruth:
            module_name = 'BruteForceProgressive'
            print(bcolors.OKGREEN + f"groundtruth start {dataset}" + bcolors.ENDC)
            util.compile_file(username=username, module_name=module_name, is_debug=True)
            est_dist_l_l, est_id_l_l = groundtruth.gnd_cpp(username=username, dataset=dataset, topk_l=topk_l,
                                                           compile_file=False, module_name=module_name)
            for topk, est_dist_l, est_id_l in zip(topk_l, est_dist_l_l, est_id_l_l):
                groundtruth.save_gnd_tsv(gnd_dist_l=est_dist_l, gnd_id_l=est_id_l, username=username, dataset=dataset,
                                         topk=topk)
            print(bcolors.OKGREEN + f"groundtruth end {dataset}" + bcolors.ENDC)
