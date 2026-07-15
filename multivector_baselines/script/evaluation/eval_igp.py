import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ["OPENBLAS_NUM_THREADS"] = "1"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)

from script.data import util
from script.evaluation.technique import vq_sq


def load_module(module_name: str):
    importlib.invalidate_caches()
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_retrieval_result(est_dist_l: np.ndarray, est_id_l: np.ndarray, retrieval_result_filename: str):
    assert len(est_dist_l) == len(est_id_l)
    with open(retrieval_result_filename, "w", encoding="utf-8") as f:
        for local_query_id, (single_dist_l, single_id_l) in enumerate(zip(est_dist_l, est_id_l)):
            for rank, (dist, passage_id) in enumerate(zip(single_dist_l, single_id_l), start=1):
                f.write(f"{local_query_id}\t{int(passage_id)}\t{rank}\t{float(dist)}\n")


def approximate_solution_retrieval(index: object, retrieval_config: dict, query_l: np.ndarray, topk: int):
    nprobe = retrieval_config["nprobe"]
    probe_topk = retrieval_config["probe_topk"]
    print(f"retrieval: nprobe {nprobe}, probe_topk {probe_topk}")

    (
        est_dist_l,
        est_id_l,
        retrieval_time_l,
        filter_time_l,
        decode_time_l,
        refine_time_l,
        n_sorted_ele_l,
        n_seen_item_l,
        n_refine_item_l,
        incremental_graph_n_compute_l,
        n_vq_score_refine_l,
        n_vq_score_linear_scan_l,
    ) = index.search(query_l=query_l, topk=topk, nprobe=nprobe, probe_topk=probe_topk)

    search_time_m = {
        "total_query_time_ms": "{:.3f}".format(sum(retrieval_time_l) * 1e3),
        "retrieval_time_p5(ms)": "{:.3f}".format(np.percentile(retrieval_time_l, 5) * 1e3),
        "retrieval_time_p50(ms)": "{:.3f}".format(np.percentile(retrieval_time_l, 50) * 1e3),
        "retrieval_time_p95(ms)": "{:.3f}".format(np.percentile(retrieval_time_l, 95) * 1e3),
        "average_query_time_ms": "{:.3f}".format(np.average(retrieval_time_l) * 1e3),
        "filter_time_average(ms)": "{:.3f}".format(np.average(filter_time_l) * 1e3),
        "decode_time_average(ms)": "{:.3f}".format(np.average(decode_time_l) * 1e3),
        "refine_time_average(ms)": "{:.3f}".format(np.average(refine_time_l) * 1e3),
        "n_sorted_ele_average": "{:.3f}".format(np.average(n_sorted_ele_l)),
        "n_seen_item_average": "{:.3f}".format(np.average(n_seen_item_l)),
        "n_refine_item_average": "{:.3f}".format(np.average(n_refine_item_l)),
        "incremental_graph_n_compute_average": "{:.3f}".format(np.average(incremental_graph_n_compute_l)),
        "n_vq_score_refine_average": "{:.3f}".format(np.average(n_vq_score_refine_l)),
        "n_vq_score_linear_scan_average": "{:.3f}".format(np.average(n_vq_score_linear_scan_l)),
    }
    retrieval_suffix = f"nprobe_{nprobe}-probe_topk_{probe_topk}"
    retrieval_time_ms_l = np.around(retrieval_time_l * 1e3, 3)
    return est_dist_l, est_id_l, retrieval_suffix, search_time_m, retrieval_time_ms_l


def approximate_solution_build_index(
    username: str,
    dataset: str,
    module: object,
    module_name: str,
    build_index_config: dict,
    build_index_suffix: str,
):
    embedding_dir = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}")
    item_n_vec_l = np.load(embedding_dir / "doclens.npy").astype(np.uint32)
    n_item = int(item_n_vec_l.shape[0])
    vec_dim = int(np.load(embedding_dir / "base_embedding" / "encoding0_float32.npy").shape[1])
    n_centroid = int(build_index_config["n_centroid"])
    n_bit = int(build_index_config["n_bit"])

    index = module.DocRetrieval(
        item_n_vec_l=item_n_vec_l.tolist(),
        n_item=n_item,
        vec_dim=vec_dim,
        n_centroid=n_centroid,
    )

    print("start insert item")
    start_time = time.time()
    centroid_l, vq_code_l, weight_l, residual_code_l, _residual_norm_l = vq_sq.vq_sq_ivf(
        username=username,
        dataset=dataset,
        module=module,
        n_centroid=n_centroid,
        n_bit=n_bit,
    )
    print("weight_l", weight_l)
    print(f"n_centroid {n_centroid}, total_n_vec {len(vq_code_l)}")
    index.build_index(
        centroid_l=np.asarray(centroid_l, dtype=np.float32),
        vq_code_l=np.asarray(vq_code_l, dtype=np.uint32),
        weight_l=np.asarray(weight_l, dtype=np.float32),
        residual_code_l=np.asarray(residual_code_l, dtype=np.uint8),
    )
    build_index_time_sec = time.time() - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")

    result_performance_path = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Result/performance")
    ensure_dir(result_performance_path)
    build_index_performance_filename = result_performance_path / f"{dataset}-build_index-{module_name}-{build_index_suffix}.json"
    with build_index_performance_filename.open("w", encoding="utf-8") as f:
        json.dump({"build_index_time (s)": build_index_time_sec}, f)

    return index, {"n_item": n_item, "vec_dim": vec_dim, "n_centroid": n_centroid, "n_bit": n_bit}


def grid_retrieval_parameter(grid_search_para: dict):
    parameter_l = []
    for nprobe in grid_search_para["nprobe"]:
        for probe_topk in grid_search_para["probe_topk"]:
            parameter_l.append({"nprobe": nprobe, "probe_topk": probe_topk})
    return parameter_l


def run_retrieval_experiments(
    username: str,
    dataset: str,
    index: object,
    module_name: str,
    build_index_suffix: str,
    build_index_config: dict,
    topk: int,
    retrieval_parameter_l: list,
):
    embedding_dir = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}")
    query_l = np.load(embedding_dir / "query_embedding.npy")

    result_answer_path = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Result/answer")
    result_performance_path = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Result/performance")
    ensure_dir(result_answer_path)
    ensure_dir(result_performance_path)

    final_result_l = []
    for retrieval_config in retrieval_parameter_l:
        est_dist_l, est_id_l, retrieval_suffix, search_time_m, _time_ms_l = approximate_solution_retrieval(
            index=index,
            retrieval_config=retrieval_config,
            query_l=query_l,
            topk=topk,
        )

        answer_name = f"{dataset}-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.tsv"
        save_retrieval_result(
            est_dist_l=est_dist_l,
            est_id_l=est_id_l,
            retrieval_result_filename=str(result_answer_path / answer_name),
        )

        retrieval_info_m = {
            "n_query": int(query_l.shape[0]),
            "topk": topk,
            "build_index": build_index_config,
            "retrieval": retrieval_config,
            "search_time": search_time_m,
        }
        performance_name = f"{dataset}-retrieval-{module_name}-top{topk}-{build_index_suffix}-{retrieval_suffix}.json"
        with (result_performance_path / performance_name).open("w", encoding="utf-8") as f:
            json.dump(retrieval_info_m, f)

        print("#############final result###############")
        print("filename", performance_name)
        print("search time", retrieval_info_m["search_time"])
        print("########################################")
        final_result_l.append({"filename": performance_name, "search_time": retrieval_info_m["search_time"]})

    for final_result in final_result_l:
        print("#############final result###############")
        print("filename", final_result["filename"])
        print("search time", final_result["search_time"])
        print("########################################")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IGP experiments.")
    parser.add_argument("--dataset_name", type=str, default="clerc-colbert")
    parser.add_argument("--username", type=str, default="ali")
    args = parser.parse_args()

    config_l = {
        "local": {
            "username": args.username,
            "dataset_l": [args.dataset_name],
            "topk_l": [10, 100],
            "is_debug": True,
            "build_index_parameter_l": [
                {"n_centroid": 1024, "n_bit": 2},
            ],
            "grid_search": True,
            "grid_search_para": {
                10: {
                    "nprobe": [32],
                    "probe_topk": [10, 50, 100, 500, 1000],
                },
                100: {
                    "nprobe": [32, 128],
                    "probe_topk": [100, 500, 1000, 5000],
                },
            },
        }
    }

    config = config_l["local"]
    username = config["username"]
    module_name = "IGP"
    move_path = "evaluation"

    util.compile_file(username=username, module_name=module_name, is_debug=config["is_debug"], move_path=move_path)
    module = load_module(module_name)

    for dataset in config["dataset_l"]:
        for build_index_config in config["build_index_parameter_l"]:
            build_index_suffix = (
                f"n_centroid_{build_index_config['n_centroid']}-"
                f"n_bit_{build_index_config['n_bit']}"
            )
            index, constructor_insert_item = approximate_solution_build_index(
                username=username,
                dataset=dataset,
                module=module,
                module_name=module_name,
                build_index_config=build_index_config,
                build_index_suffix=build_index_suffix,
            )

            for topk in config["topk_l"]:
                retrieval_parameter_l = grid_retrieval_parameter(config["grid_search_para"][topk])
                run_retrieval_experiments(
                    username=username,
                    dataset=dataset,
                    index=index,
                    module_name=module_name,
                    build_index_suffix=build_index_suffix,
                    build_index_config=constructor_insert_item,
                    topk=topk,
                    retrieval_parameter_l=retrieval_parameter_l,
                )
