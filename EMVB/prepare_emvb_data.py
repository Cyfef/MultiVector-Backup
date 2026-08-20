#!/usr/bin/env python3
"""Prepare EMVB inputs from multi-vector document/query embeddings.

This script follows the FAISS decomposition workflow in `ConvertFaissIndex.ipynb`
and adds the query-padding step required by `src/perf_emvb.cpp`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
from faiss.contrib.inspect_tools import get_invlist
from tqdm.auto import tqdm

import resource
import os

def get_dir_size(path: Path) -> int:
    """Return total size (in bytes) of all regular files under `path`."""
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-embeddings", required=True, help="Path to the document embeddings .npy file.")
    parser.add_argument("--doclens", help="Path to the per-document chunk counts .npy file.")
    parser.add_argument(
        "--base-offsets",
        help="Path to packed document offsets .npy file. Use this instead of --doclens for ColBERT-style inputs.",
    )
    parser.add_argument("--query-embeddings", required=True, help="Path to the query embeddings .npy file.")
    parser.add_argument("--query-doclens", help="Path to the per-query chunk counts .npy file.")
    parser.add_argument(
        "--query-offsets",
        help="Path to packed query offsets .npy file. Use this instead of --query-doclens for ColBERT-style inputs.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional prefix subset size for documents. Keeps only the first N documents and their vectors.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional prefix subset size for queries. Keeps only the first N queries and their vectors.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where EMVB inputs will be written.")
    parser.add_argument("--nlist", type=int, required=True, help="Number of IVF centroids.")
    parser.add_argument("--pq-m", type=int, required=True, help="Number of PQ subquantizers.")
    parser.add_argument("--nbits", type=int, default=8, help="Bits per PQ code. EMVB currently expects 8.")
    parser.add_argument(
        "--metric",
        choices=("l2", "ip"),
        default="l2",
        help="FAISS metric used when training the IVFPQ index. The notebook example uses l2.",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="How many document vectors to sample for FAISS training. Defaults to max(100000, 40 * nlist), capped by the dataset size.",
    )
    parser.add_argument(
        "--add-batch-size",
        type=int,
        default=50_000,
        help="How many document vectors to add to FAISS per batch.",
    )
    parser.add_argument(
        "--query-max-terms",
        type=int,
        default=None,
        help="Pad queries to this many terms. Defaults to the maximum query length in the input file.",
    )
    parser.add_argument(
        "--pad-value",
        type=float,
        default=0.0,
        help="Padding value used for missing query vectors.",
    )
    parser.add_argument(
        "--truncate-queries",
        action="store_true",
        help="Allow truncating queries longer than --query-max-terms.",
    )
    parser.add_argument(
        "--faiss-threads",
        type=int,
        default=None,
        help="Optional FAISS thread count. Leave unset to keep the environment default.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed for training set sampling.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use FAISS GPU resources for IVFPQ training and add(). The final export still runs on a CPU copy.",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="GPU device id to use when --use-gpu is enabled.",
    )
    parser.add_argument(
        "--gpu-use-float16",
        action="store_true",
        help="Use float16 in the FAISS GPU clone options to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--gpu-train-only",
        action="store_true",
        help="Train IVFPQ on GPU, copy the trained empty index back to CPU, then add/decompose on CPU.",
    )
    return parser.parse_args()


def load_npy(path: str, mmap_mode: str | None = None) -> np.ndarray:
    return np.load(path, mmap_mode=mmap_mode)


def ensure_1d_counts(name: str, array: np.ndarray) -> np.ndarray:
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {array.shape}")
    if np.any(array <= 0):
        raise ValueError(f"{name} must contain strictly positive counts")
    return array.astype(np.int64, copy=False)


def counts_from_offsets(name: str, array: np.ndarray) -> np.ndarray:
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array, got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} must contain at least two entries")

    array = array.astype(np.int64, copy=False)
    if int(array[0]) != 0:
        raise ValueError(f"{name} must start at 0, got {array[0]}")

    deltas = np.diff(array)
    if np.any(deltas <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    return deltas


def validate_embeddings(name: str, array: np.ndarray) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}")
    return array


def maybe_truncate_counts(name: str, counts: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None:
        return counts
    if limit <= 0:
        raise ValueError(f"{name} limit must be positive, got {limit}")
    return counts[:limit]


def sample_training_set(base_embeddings: np.ndarray, train_size: int, seed: int) -> np.ndarray:
    total = base_embeddings.shape[0]
    train_size = min(train_size, total)
    rng = np.random.default_rng(seed)
    sample_ids = np.sort(rng.choice(total, size=train_size, replace=False))
    return np.asarray(base_embeddings[sample_ids], dtype=np.float32)


def resolve_train_size(total_vectors: int, requested_train_size: int | None, nlist: int) -> int:
    recommended = min(total_vectors, max(100_000, 40 * nlist))
    if requested_train_size is None:
        return recommended
    return min(requested_train_size, total_vectors)


def pad_query_embeddings(
    query_embeddings: np.ndarray,
    query_doclens: np.ndarray,
    query_max_terms: int | None,
    pad_value: float,
    truncate_queries: bool,
) -> np.ndarray:
    n_queries = int(query_doclens.shape[0])
    dim = int(query_embeddings.shape[1])
    inferred_max_terms = int(query_doclens.max())
    target_terms = query_max_terms or inferred_max_terms

    if target_terms <= 0:
        raise ValueError("query_max_terms must be positive")

    too_long = query_doclens > target_terms
    if np.any(too_long) and not truncate_queries:
        longest = int(query_doclens.max())
        raise ValueError(
            f"At least one query has {longest} terms, but query_max_terms={target_terms}. "
            "Pass --truncate-queries to allow truncation."
        )

    padded = np.full((n_queries, target_terms, dim), pad_value, dtype=np.float32)
    offset = 0
    for qid, qlen in enumerate(query_doclens.tolist()):
        usable = min(int(qlen), target_terms)
        padded[qid, :usable] = np.asarray(query_embeddings[offset : offset + usable], dtype=np.float32)
        offset += int(qlen)

    return padded


def make_quantizer(metric: str, dim: int) -> faiss.Index:
    if metric == "l2":
        return faiss.IndexFlatL2(dim)
    return faiss.IndexFlatIP(dim)


def make_ivfpq_index(metric: str, dim: int, nlist: int, pq_m: int, nbits: int) -> faiss.IndexIVFPQ:
    quantizer = make_quantizer(metric, dim)
    if metric == "l2":
        return faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, nbits)
    return faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, nbits, faiss.METRIC_INNER_PRODUCT)


def maybe_move_index_to_gpu(
    index: faiss.IndexIVFPQ,
    use_gpu: bool,
    gpu_device: int,
    gpu_use_float16: bool,
) -> tuple[faiss.Index, faiss.StandardGpuResources | None]:
    if not use_gpu:
        return index, None

    if not hasattr(faiss, "StandardGpuResources"):
        raise RuntimeError("This FAISS build does not expose GPU APIs.")

    num_gpus = faiss.get_num_gpus()
    if num_gpus <= 0:
        raise RuntimeError("No visible FAISS GPU devices were found.")
    if gpu_device < 0 or gpu_device >= num_gpus:
        raise ValueError(f"gpu_device={gpu_device} is out of range for {num_gpus} visible GPUs")

    resources = faiss.StandardGpuResources()
    clone_options = faiss.GpuClonerOptions()
    clone_options.useFloat16 = gpu_use_float16
    gpu_index = faiss.index_cpu_to_gpu(resources, gpu_device, index, clone_options)
    return gpu_index, resources


def ensure_cpu_index(index: faiss.Index, use_gpu: bool) -> faiss.IndexIVFPQ:
    is_gpu_index = type(index).__name__.startswith("Gpu")
    cpu_index = faiss.index_gpu_to_cpu(index) if is_gpu_index else index
    return faiss.downcast_index(cpu_index)


def describe_index(index: faiss.Index, label: str) -> None:
    print(
        f"{label}: type={type(index).__name__} d={getattr(index, 'd', 'n/a')} "
        f"ntotal={getattr(index, 'ntotal', 'n/a')} trained={getattr(index, 'is_trained', 'n/a')}"
    )


def maybe_transfer_trained_index_to_cpu(index: faiss.Index, args: argparse.Namespace) -> faiss.Index:
    if not args.use_gpu or not args.gpu_train_only:
        return index

    print("Copying trained empty index from GPU to CPU before add()")
    cpu_index = make_ivfpq_index(args.metric, args.embedding_dim, args.nlist, args.pq_m, args.nbits)
    if hasattr(index, "copyTo"):
        index.copyTo(cpu_index)
    else:
        cpu_index = ensure_cpu_index(index, True)
    describe_index(cpu_index, "CPU index after GPU->CPU copy")
    print("CPU index ready for add()/export")
    return cpu_index


def build_emb2pid(doclens: np.ndarray) -> np.ndarray:
    emb2pid = np.empty(int(doclens.sum()), dtype=np.int64)
    offset = 0
    for doc_id, doc_len in enumerate(doclens.tolist()):
        next_offset = offset + int(doc_len)
        emb2pid[offset:next_offset] = doc_id
        offset = next_offset
    return emb2pid


def decompose_faiss_index(index: faiss.IndexIVFPQ, emb2pid: np.ndarray, output_dir: Path) -> None:
    residuals = np.zeros((index.ntotal, index.pq.M), dtype=np.uint8)
    all_indices = np.zeros((index.ntotal,), dtype=np.uint64)
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    centroids_to_pids: list[np.ndarray] = [np.empty((0,), dtype=np.int64) for _ in range(index.nlist)]

    for centroid_id in tqdm(range(index.nlist), desc="Decomposing IVF lists"):
        ids, codes = get_invlist(index.invlists, centroid_id)
        ids = np.asarray(ids, dtype=np.int64)
        codes = np.asarray(codes, dtype=np.uint8)
        if ids.size == 0:
            continue

        if codes.ndim == 1:
            codes = codes.reshape(-1, index.pq.M)

        residuals[ids] = codes
        all_indices[ids] = centroid_id
        centroids_to_pids[centroid_id] = emb2pid[ids]

    with (output_dir / "centroids_to_pids.txt").open("w", encoding="utf-8") as handle:
        for centroid_pids in tqdm(centroids_to_pids, desc="Writing centroid_to_pids"):
            if centroid_pids.size:
                handle.write(" ".join(str(int(pid)) for pid in centroid_pids))
            handle.write("\n")

    np.save(output_dir / "residuals.npy", residuals)
    np.save(output_dir / "centroids.npy", centroids)
    np.save(output_dir / "index_assignment.npy", all_indices)
    np.save(output_dir / "pq_centroids.npy", faiss.vector_to_array(index.pq.centroids))


def write_query_ids(path: Path, n_queries: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for qid in range(n_queries):
            handle.write(f"{qid}\n")


def main() -> None:
    args = parse_args()
    start = time.time()

    if args.nbits != 8:
        raise ValueError("EMVB hardcodes 8-bit PQ codes, so --nbits must be 8.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.faiss_threads is not None:
        faiss.omp_set_num_threads(args.faiss_threads)

    base_embeddings = validate_embeddings("base_embeddings", load_npy(args.base_embeddings, mmap_mode="r"))
    if bool(args.doclens) == bool(args.base_offsets):
        raise ValueError("Provide exactly one of --doclens or --base-offsets")
    if bool(args.query_doclens) == bool(args.query_offsets):
        raise ValueError("Provide exactly one of --query-doclens or --query-offsets")

    doclens = (
        ensure_1d_counts("doclens", load_npy(args.doclens))
        if args.doclens
        else counts_from_offsets("base_offsets", load_npy(args.base_offsets))
    )
    query_embeddings = validate_embeddings("query_embeddings", load_npy(args.query_embeddings, mmap_mode="r"))
    query_doclens = (
        ensure_1d_counts("query_doclens", load_npy(args.query_doclens))
        if args.query_doclens
        else counts_from_offsets("query_offsets", load_npy(args.query_offsets))
    )

    doclens = maybe_truncate_counts("max_docs", doclens, args.max_docs)
    query_doclens = maybe_truncate_counts("max_queries", query_doclens, args.max_queries)

    base_embeddings = base_embeddings[: int(doclens.sum())]
    query_embeddings = query_embeddings[: int(query_doclens.sum())]

    if int(doclens.sum()) != int(base_embeddings.shape[0]):
        raise ValueError(
            f"sum(doclens)={int(doclens.sum())} does not match base_embeddings rows={base_embeddings.shape[0]}"
        )
    if int(query_doclens.sum()) != int(query_embeddings.shape[0]):
        raise ValueError(
            f"sum(query_doclens)={int(query_doclens.sum())} does not match query_embeddings rows={query_embeddings.shape[0]}"
        )
    if int(base_embeddings.shape[1]) != int(query_embeddings.shape[1]):
        raise ValueError(
            f"Embedding dimensions do not match: base={base_embeddings.shape[1]}, query={query_embeddings.shape[1]}"
        )

    dim = int(base_embeddings.shape[1])
    args.embedding_dim = dim
    if dim % args.pq_m != 0:
        raise ValueError(f"Embedding dimension {dim} must be divisible by pq_m={args.pq_m}")

    print(f"Base embeddings: {base_embeddings.shape} {base_embeddings.dtype}")
    print(f"Document count: {doclens.shape[0]}")
    print(f"Query embeddings: {query_embeddings.shape} {query_embeddings.dtype}")
    print(f"Query count: {query_doclens.shape[0]}")

    padded_queries = pad_query_embeddings(
        query_embeddings=query_embeddings,
        query_doclens=query_doclens,
        query_max_terms=args.query_max_terms,
        pad_value=args.pad_value,
        truncate_queries=args.truncate_queries,
    )
    print(f"Padded queries: {padded_queries.shape}")

    np.save(output_dir / "query_embeddings.npy", padded_queries)
    np.save(output_dir / "alldoclens.npy", doclens.astype(np.int32))
    write_query_ids(output_dir / "queries_id.txt", int(query_doclens.shape[0]))

    train_size = resolve_train_size(base_embeddings.shape[0], args.train_size, args.nlist)
    recommended_train_size = min(base_embeddings.shape[0], max(100_000, 40 * args.nlist))
    if train_size < recommended_train_size:
        print(
            f"WARNING requested train_size={train_size} is smaller than the FAISS coarse-quantizer recommendation "
            f"of {recommended_train_size} for nlist={args.nlist}"
        )

    training_set = sample_training_set(base_embeddings, train_size, args.seed)
    print(f"Training IVFPQ with sample shape {training_set.shape}")

    index = make_ivfpq_index(args.metric, dim, args.nlist, args.pq_m, args.nbits)
    describe_index(index, "Initial CPU index")
    index, _gpu_resources = maybe_move_index_to_gpu(
        index=index,
        use_gpu=args.use_gpu,
        gpu_device=args.gpu_device,
        gpu_use_float16=args.gpu_use_float16,
    )
    if args.use_gpu:
        print(f"Using FAISS GPU device {args.gpu_device} for train/add")
        describe_index(index, "GPU index before train")

    index.train(training_set)
    print("Index training complete")
    describe_index(index, "Index after train")
    index = maybe_transfer_trained_index_to_cpu(index, args)

    for start_idx in tqdm(range(0, base_embeddings.shape[0], args.add_batch_size), desc="Adding embeddings"):
        stop_idx = min(start_idx + args.add_batch_size, base_embeddings.shape[0])
        batch = np.asarray(base_embeddings[start_idx:stop_idx], dtype=np.float32)
        if start_idx == 0:
            print(f"First add batch shape={batch.shape} dtype={batch.dtype} index_d={getattr(index, 'd', 'n/a')}")
        index.add(batch)

    print(f"Index add complete, ntotal={index.ntotal}")
    cpu_index = ensure_cpu_index(index, args.use_gpu)
    describe_index(cpu_index, "CPU index before write/export")
    try:
        faiss.write_index(cpu_index, str(output_dir / "faiss_ivfpq.index"))
    except RuntimeError as exc:
        print(f"WARNING failed to serialize faiss_ivfpq.index: {exc}")
        print("Continuing with EMVB decomposition because the decomposed files are the required runtime inputs.")

    emb2pid = build_emb2pid(doclens)
    decompose_faiss_index(cpu_index, emb2pid, output_dir)

    # ---------- 新增：计算索引大小 ----------
    index_size_bytes = get_dir_size(output_dir)

    # ---------- 新增：获取峰值内存（使用 resource） ----------
    # ru_maxrss 单位为 KiB（Linux/macOS），转换为字节
    peak_mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    metadata = {
        "base_embeddings": str(Path(args.base_embeddings).resolve()),
        "doclens": str(Path(args.doclens).resolve()) if args.doclens else None,
        "base_offsets": str(Path(args.base_offsets).resolve()) if args.base_offsets else None,
        "query_embeddings": str(Path(args.query_embeddings).resolve()),
        "query_doclens": str(Path(args.query_doclens).resolve()) if args.query_doclens else None,
        "query_offsets": str(Path(args.query_offsets).resolve()) if args.query_offsets else None,
        "output_dir": str(output_dir.resolve()),
        "n_docs": int(doclens.shape[0]),
        "n_queries": int(query_doclens.shape[0]),
        "n_doc_vectors": int(base_embeddings.shape[0]),
        "n_query_vectors": int(query_embeddings.shape[0]),
        "max_docs": args.max_docs,
        "max_queries": args.max_queries,
        "embedding_dim": dim,
        "query_max_terms": int(padded_queries.shape[1]),
        "metric": args.metric,
        "nlist": args.nlist,
        "pq_m": args.pq_m,
        "nbits": args.nbits,
        "train_size": int(training_set.shape[0]),
        "add_batch_size": args.add_batch_size,
        "seed": args.seed,
        "use_gpu": args.use_gpu,
        "gpu_device": args.gpu_device if args.use_gpu else None,
        "gpu_use_float16": args.gpu_use_float16,
        "gpu_train_only": args.gpu_train_only,
        "elapsed_seconds": time.time() - start,
        # ---------- 新增字段 ----------
        "index_size_bytes": index_size_bytes,
        "peak_build_mem_bytes": peak_mem_bytes,
        # -------------------------------
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
