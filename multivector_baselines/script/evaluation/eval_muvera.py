import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"

FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)

from script.data import util


def load_module(module_name: str):
    importlib.invalidate_caches()
    if module_name in sys.modules:
        del sys.modules[module_name]
    return importlib.import_module(module_name)


def save_retrieval_result(est_dist_l: np.ndarray, est_id_l: np.ndarray, retrieval_result_filename: str):
    assert len(est_dist_l) == len(est_id_l)
    n_query = len(est_dist_l)
    with open(retrieval_result_filename, "w", encoding="utf-8") as f:
        for local_query_id, (single_dist_l, single_id_l) in enumerate(zip(est_dist_l, est_id_l)):
            for rank, (dist, passage_id) in enumerate(zip(single_dist_l, single_id_l), start=1):
                f.write(f"{local_query_id}\t{int(passage_id)}\t{rank}\t{float(dist)}\n")


def approximate_solution_retrieval(index: object, retrieval_config: dict, query_l: np.ndarray, topk: int):
    n_candidate = retrieval_config["n_candidate"]
    print(f"retrieval: n_candidate {n_candidate}")

    (
        est_dist_l,
        est_id_l,
        retrieval_time_l,
        transform_time_l,
        ip_time_l,
        decode_time_l,
        refine_time_l,
        n_search_candidate_l,
        _query_mat,
    ) = index.search(query_l=query_l, topk=topk, n_candidate=n_candidate)

    search_time_m = {
        "total_query_time_ms": "{:.3f}".format(sum(retrieval_time_l) * 1e3),
        "retrieval_time_p5(ms)": "{:.3f}".format(np.percentile(retrieval_time_l, 5) * 1e3),
        "retrieval_time_p50(ms)": "{:.3f}".format(np.percentile(retrieval_time_l, 50) * 1e3),
        "retrieval_time_p95(ms)": "{:.3f}".format(np.percentile(retrieval_time_l, 95) * 1e3),
        "average_query_time_ms": "{:.3f}".format(1.0 * np.average(retrieval_time_l) * 1e3),
        "transform_time_average(ms)": "{:.3f}".format(1.0 * np.average(transform_time_l) * 1e3),
        "ip_time_average(ms)": "{:.3f}".format(1.0 * np.average(ip_time_l) * 1e3),
        "decode_time_average(ms)": "{:.3f}".format(1.0 * np.average(decode_time_l) * 1e3),
        "refine_time_average(ms)": "{:.3f}".format(1.0 * np.average(refine_time_l) * 1e3),
        "n_search_candidate_average": "{:.3f}".format(1.0 * np.average(n_search_candidate_l)),
    }
    retrieval_suffix = f"n_candidate_{n_candidate}"
    retrieval_time_ms_l = np.around(retrieval_time_l * 1e3, 3)
    return est_dist_l, est_id_l, retrieval_suffix, search_time_m, retrieval_time_ms_l


def compress_into_codes(embs: np.ndarray, centroid_l: np.ndarray):
    codes = []
    centroid_l = torch.tensor(centroid_l, device="cpu")
    bsize = max(1, (1 << 29) // centroid_l.shape[0])
    embs = torch.from_numpy(embs)
    for batch in embs.split(bsize):
        indices = (centroid_l @ batch.T.float()).max(dim=0).indices.cpu()
        codes.append(indices)
    return torch.cat(codes).numpy().astype(np.uint32)


def transform_multi_vector_to_single_vector(
    itemlen_l_chunk: np.ndarray,
    item_vecs_l_chunk: np.ndarray,
    random_matrix_l: torch.Tensor,
    partition_vec_l: torch.Tensor,
    module,
    r_reps: int,
    k_sim: int,
    vec_dim: int,
    d_proj: int,
):
    n_item_chunk = itemlen_l_chunk.shape[0]
    n_vec_chunk = item_vecs_l_chunk.shape[0]

    item_vecs_l_chunk_t = torch.from_numpy(item_vecs_l_chunk)
    partition_bit_l = torch.einsum("ij,klj->ikl", item_vecs_l_chunk_t, partition_vec_l)
    partition_bit_l = torch.sign(partition_bit_l).int()
    partition_bit_l[partition_bit_l == -1] = 0
    vec_cluster_bit_l = partition_bit_l.cpu().numpy().astype(np.uint8)
    assert vec_cluster_bit_l.shape == (n_vec_chunk, r_reps, k_sim)

    cluster_vector_l = module.assign_cluster_vector(
        vec_cluster_bit_l=vec_cluster_bit_l,
        item_vecs_l_chunk=item_vecs_l_chunk,
        item_n_vec_l_chunk=itemlen_l_chunk,
        batch_n_vec=n_vec_chunk,
        batch_n_item=n_item_chunk,
        r_reps=r_reps,
        k_sim=k_sim,
        vec_dim=vec_dim,
    )
    cluster_vector_l_t = torch.from_numpy(cluster_vector_l)
    ip_vector_l = torch.einsum("iljk,lmk->iljm", cluster_vector_l_t, random_matrix_l)
    ip_vector_l = ip_vector_l.cpu().numpy().reshape(n_item_chunk, -1).astype(np.float32)
    assert ip_vector_l.shape == (n_item_chunk, r_reps * (2 ** k_sim) * d_proj)
    return ip_vector_l


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_index(
    username: str,
    dataset: str,
    index: object,
    module: object,
    vec_dim: int,
    k_sim: int,
    d_proj: int,
    r_reps: int,
    n_centroid_per_subspace: int,
    dim_per_subspace: int,
    build_index_suffix: str,
):
    embedding_dir = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}")
    base_embedding_dir = embedding_dir / "base_embedding"
    raw_transform_dir = Path(
        f"/data1/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}/document/transformed_embeddings"
    )
    ensure_dir(raw_transform_dir)

    query_single_vec_path = raw_transform_dir / f"single_vector_query_embedding-{build_index_suffix}.npy"
    chunk_single_vec_dir = raw_transform_dir / f"transformed_single_vector_embedding-{build_index_suffix}"
    ensure_dir(chunk_single_vec_dir)
    full_single_vec_path = raw_transform_dir / f"single_vector_embedding-{build_index_suffix}.npy"
    full_item_vec_path = raw_transform_dir / "all_document_vectors.npy"
    pq_centroid_path = raw_transform_dir / f"pq_centroids-{build_index_suffix}.npy"
    pq_code_path = raw_transform_dir / f"pq_codes-{build_index_suffix}.npy"

    partition_vec_l_np = np.random.normal(loc=0, scale=1.0, size=(r_reps, k_sim, vec_dim)).astype(np.float32)
    partition_vec_l = torch.tensor(partition_vec_l_np, device="cpu")
    random_matrix_l = torch.randint(0, 2, (r_reps, d_proj, vec_dim), dtype=torch.int32)
    random_matrix_l = (random_matrix_l * 2 - 1).float()
    random_matrix_l_np = random_matrix_l.numpy().astype(np.float32)
    index.add_projection(partition_vec_l=partition_vec_l_np, random_matrix_l=random_matrix_l_np)

    item_n_vec_l = np.load(embedding_dir / "doclens.npy").astype(np.uint32)
    n_vecs = int(np.sum(item_n_vec_l))
    n_item = int(item_n_vec_l.shape[0])
    n_chunk = util.get_n_chunk(str(base_embedding_dir))
    ip_dim = r_reps * (2 ** k_sim) * d_proj

    ip_vector_l = np.lib.format.open_memmap(full_single_vec_path, mode="w+", dtype=np.float32, shape=(n_item, ip_dim))
    item_vec_l = np.lib.format.open_memmap(full_item_vec_path, mode="w+", dtype=np.float32, shape=(n_vecs, vec_dim))

    query_l = np.load(embedding_dir / "query_embedding.npy")
    query_lens = np.load(embedding_dir / "query_n_vec_length.npy").astype(np.uint32)
    query_concat_l = np.concatenate([query_l[idx][: query_lens[idx]] for idx in range(len(query_lens))], axis=0)
    query_single_vec = transform_multi_vector_to_single_vector(
        itemlen_l_chunk=query_lens,
        item_vecs_l_chunk=query_concat_l,
        random_matrix_l=random_matrix_l,
        partition_vec_l=partition_vec_l,
        module=module,
        r_reps=r_reps,
        k_sim=k_sim,
        vec_dim=vec_dim,
        d_proj=d_proj,
    )
    np.save(query_single_vec_path, query_single_vec)

    item_offset = 0
    vec_offset = 0
    for chunk_id in tqdm.trange(n_chunk, desc=f"{dataset} MUVERA transform"):
        itemlen_l_chunk = np.load(base_embedding_dir / f"doclens{chunk_id}.npy").astype(np.uint32)
        item_vecs_l_chunk = np.load(base_embedding_dir / f"encoding{chunk_id}_float32.npy").astype(np.float32)
        n_item_chunk = int(itemlen_l_chunk.shape[0])
        n_vec_chunk = int(item_vecs_l_chunk.shape[0])

        ip_vector_l_chunk = transform_multi_vector_to_single_vector(
            itemlen_l_chunk=itemlen_l_chunk,
            item_vecs_l_chunk=item_vecs_l_chunk,
            random_matrix_l=random_matrix_l,
            partition_vec_l=partition_vec_l,
            module=module,
            r_reps=r_reps,
            k_sim=k_sim,
            vec_dim=vec_dim,
            d_proj=d_proj,
        )
        np.save(chunk_single_vec_dir / f"single_vector_embedding{chunk_id}.npy", ip_vector_l_chunk)

        ip_vector_l[item_offset:item_offset + n_item_chunk] = ip_vector_l_chunk
        item_vec_l[vec_offset:vec_offset + n_vec_chunk] = item_vecs_l_chunk
        item_offset += n_item_chunk
        vec_offset += n_vec_chunk

    assert item_offset == n_item
    assert vec_offset == n_vecs
    ip_vector_l.flush()
    item_vec_l.flush()

    print("start build_graph_index")
    index.build_graph_index(ip_vector_l=ip_vector_l)
    print("start add_item_vector_l")
    index.add_item_vector_l(vec_l=item_vec_l)

    n_subspace = ip_dim // dim_per_subspace + (0 if ip_dim % dim_per_subspace == 0 else 1)
    sub_centroid_l_l = np.lib.format.open_memmap(
        pq_centroid_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_subspace, n_centroid_per_subspace, dim_per_subspace),
    )
    sub_code_l_l = np.lib.format.open_memmap(
        pq_code_path,
        mode="w+",
        dtype=np.uint32,
        shape=(n_subspace, n_item),
    )

    sample_size = min(n_item, 300_000)
    use_gpu = faiss.get_num_gpus() > 0
    for subspace_id in tqdm.trange(n_subspace, desc=f"{dataset} MUVERA PQ"):
        start_dim = subspace_id * dim_per_subspace
        end_dim = min(start_dim + dim_per_subspace, ip_dim)
        sub_ip_vector_l = np.asarray(ip_vector_l[:, start_dim:end_dim], dtype=np.float32)
        n_dim_subspace = end_dim - start_dim
        if n_dim_subspace < dim_per_subspace:
            sub_ip_vector_l = np.pad(
                sub_ip_vector_l,
                pad_width=((0, 0), (0, dim_per_subspace - n_dim_subspace)),
                mode="constant",
                constant_values=0,
            ).astype(np.float32, copy=False)

        train_vecs = np.ascontiguousarray(sub_ip_vector_l[:sample_size])
        kmeans = faiss.Kmeans(
            dim_per_subspace,
            n_centroid_per_subspace,
            niter=20,
            gpu=use_gpu,
            verbose=True,
            seed=123,
        )
        kmeans.train(train_vecs)
        sub_centroid_l = np.ascontiguousarray(kmeans.centroids.reshape(n_centroid_per_subspace, dim_per_subspace))
        _dist, sub_code_l = kmeans.assign(np.ascontiguousarray(sub_ip_vector_l))
        sub_centroid_l_l[subspace_id] = sub_centroid_l
        sub_code_l_l[subspace_id] = sub_code_l.astype(np.uint32)

    sub_centroid_l_l.flush()
    sub_code_l_l.flush()

    print("start add_pq_code_l")
    index.add_pq_code_l(sub_centroid_l_l=sub_centroid_l_l, sub_code_l_l=sub_code_l_l)


def approximate_solution_build_index(
    username: str,
    dataset: str,
    constructor_insert_item: dict,
    module: object,
    module_name: str,
    build_index_config: dict,
    build_index_suffix: str,
):
    index = module.DocRetrieval(**constructor_insert_item)
    print("start insert item")
    start_time = time.time()

    embedding_dir = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}")
    vec_dim = np.load(embedding_dir / "base_embedding" / "encoding0_float32.npy").shape[1]

    build_index(
        username=username,
        dataset=dataset,
        index=index,
        module=module,
        vec_dim=vec_dim,
        k_sim=build_index_config["k_sim"],
        d_proj=build_index_config["d_proj"],
        r_reps=build_index_config["r_reps"],
        n_centroid_per_subspace=build_index_config["n_centroid_per_subspace"],
        dim_per_subspace=build_index_config["dim_per_subspace"],
        build_index_suffix=build_index_suffix,
    )

    build_index_time_sec = time.time() - start_time
    print(f"insert time spend {build_index_time_sec:.3f}s")

    result_performance_path = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Result/performance")
    ensure_dir(result_performance_path)
    build_index_performance_filename = result_performance_path / f"{dataset}-build_index-{module_name}-{build_index_suffix}.json"
    with build_index_performance_filename.open("w", encoding="utf-8") as f:
        json.dump({"build_index_time (s)": build_index_time_sec}, f)

    return index


def grid_retrieval_parameter(grid_search_para: dict):
    return [{"n_candidate": n_candidate} for n_candidate in grid_search_para["n_candidate"]]


def run_retrieval_experiments(
    username: str,
    dataset: str,
    index: object,
    module_name: str,
    build_index_suffix: str,
    constructor_insert_item: dict,
    topk: int,
    retrieval_parameter_l: list,
):
    embedding_dir = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}")
    query_l = np.load(embedding_dir / "query_embedding.npy")

    result_answer_path = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Result/answer")
    result_performance_path = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Result/performance")
    ensure_dir(result_answer_path)
    ensure_dir(result_performance_path)

    build_index_config = dict(constructor_insert_item)
    build_index_config.pop("item_n_vec_l", None)

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
    parser = argparse.ArgumentParser(description="Run MUVERA experiments.")
    parser.add_argument("--dataset_name", type=str, default="msmarco-large")
    parser.add_argument("--username", type=str, default="ali")
    args = parser.parse_args()

    config_l = {
        "dbg": {
            "username": "username2",
            "dataset_l": ["lotte"],
            "topk_l": [10, 100],
            "is_debug": False,
            "build_index_parameter_l": [
                {
                    "k_sim": 3,
                    "d_proj": 8,
                    "r_reps": 20,
                    "R": 200,
                    "L": 600,
                    "alpha": 1.2,
                    "n_centroid_per_subspace": 256,
                    "dim_per_subspace": 16,
                },
            ],
            "grid_search": True,
            "grid_search_para": {
                10: {"n_candidate": [10, 20, 50, 100, 200, 500, 1000, 2000, 4000]},
                100: {"n_candidate": [100, 500, 1000, 2000, 4000, 8000, 16000]},
            },
        },
        "local": {
            "username": args.username,
            "dataset_l": [args.dataset_name],
            "topk_l": [10, 100],
            "is_debug": True,
            "build_index_parameter_l": [
                {
                    "k_sim": 3,
                    "d_proj": 32,
                    "r_reps": 8,
                    "R": 20,
                    "L": 50,
                    "alpha": 1.2,
                    "n_centroid_per_subspace": 256,
                    "dim_per_subspace": 16,
                },
            ],
            "grid_search": True,
            "grid_search_para": {
                10: {"n_candidate": [20, 50, 100]},
                100: {"n_candidate": [100, 500, 1000]},
            },
        },
    }

    config = config_l["local"]
    username = config["username"]
    module_name = "MUVERA"
    move_path = "evaluation"

    util.compile_file(username=username, module_name=module_name, is_debug=config["is_debug"], move_path=move_path)
    module = load_module(module_name)

    for dataset in config["dataset_l"]:
        embedding_dir = Path(f"/data1/{username}/Dataset/multi-vector-retrieval/Embedding/{dataset}")
        vec_dim = np.load(embedding_dir / "base_embedding" / "encoding0_float32.npy").shape[1]
        item_n_vec_l = np.load(embedding_dir / "doclens.npy").astype(np.uint32)
        n_item = int(item_n_vec_l.shape[0])

        for build_index_config in config["build_index_parameter_l"]:
            constructor_insert_item = {
                "item_n_vec_l": item_n_vec_l.tolist(),
                "n_item": n_item,
                "vec_dim": int(vec_dim),
                "k_sim": build_index_config["k_sim"],
                "d_proj": build_index_config["d_proj"],
                "r_reps": build_index_config["r_reps"],
                "R": build_index_config["R"],
                "L": build_index_config["L"],
                "alpha": build_index_config["alpha"],
                "n_centroid_per_subspace": build_index_config["n_centroid_per_subspace"],
                "dim_per_subspace": build_index_config["dim_per_subspace"],
            }
            build_index_suffix = (
                f"k_sim_{build_index_config['k_sim']}-"
                f"d_proj_{build_index_config['d_proj']}-"
                f"r_reps_{build_index_config['r_reps']}"
            )

            index = approximate_solution_build_index(
                username=username,
                dataset=dataset,
                constructor_insert_item=constructor_insert_item,
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
                    constructor_insert_item=constructor_insert_item,
                    topk=topk,
                    retrieval_parameter_l=retrieval_parameter_l,
                )
