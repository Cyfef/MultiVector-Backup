#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import faiss
import numpy as np
from tqdm import tqdm


RAW_BASE_EMBEDDINGS = ""
RAW_BASE_CHUNKS = ""
RAW_QUERY_EMBEDDINGS = ""
RAW_QUERY_CHUNKS = ""


def configure_raw_paths(dataset_dir: Path, dataset_stem: str) -> None:
    global RAW_BASE_EMBEDDINGS
    global RAW_BASE_CHUNKS
    global RAW_QUERY_EMBEDDINGS
    global RAW_QUERY_CHUNKS

    base = dataset_dir / f"full_multi_embeddings_{dataset_stem}.npy"
    base_chunks = dataset_dir / f"full_multi_chunk_num_{dataset_stem}.npy"
    query = dataset_dir / f"full_multi_embeddings_{dataset_stem}_query.npy"
    query_chunks = dataset_dir / f"full_multi_chunk_num_{dataset_stem}_query.npy"

    RAW_BASE_EMBEDDINGS = str(base)
    RAW_BASE_CHUNKS = str(base_chunks)
    RAW_QUERY_EMBEDDINGS = str(query)
    RAW_QUERY_CHUNKS = str(query_chunks)


def load_memmap(path: str):
    return np.load(path, mmap_mode="r")


def build_doc_offsets(chunk_counts: np.ndarray) -> np.ndarray:
    offsets = np.zeros(len(chunk_counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(chunk_counts, dtype=np.int64)
    return offsets


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    docdata = root / "docdata"
    cdata = root / "cdata"
    qdata = root / "qdata"
    docdata.mkdir(parents=True, exist_ok=True)
    cdata.mkdir(parents=True, exist_ok=True)
    qdata.mkdir(parents=True, exist_ok=True)
    return docdata, cdata, qdata


def load_raw_dataset():
    base_embeddings = load_memmap(RAW_BASE_EMBEDDINGS)
    base_chunks = np.asarray(load_memmap(RAW_BASE_CHUNKS), dtype=np.int64)
    query_embeddings = load_memmap(RAW_QUERY_EMBEDDINGS)
    query_chunks = np.asarray(load_memmap(RAW_QUERY_CHUNKS), dtype=np.int64)
    return base_embeddings, base_chunks, query_embeddings, query_chunks


def inspect_dataset() -> None:
    base_embeddings, base_chunks, query_embeddings, query_chunks = load_raw_dataset()
    print("base_embeddings", base_embeddings.shape, base_embeddings.dtype)
    print("base_chunks", base_chunks.shape, base_chunks.dtype, "sum", int(base_chunks.sum()))
    print("query_embeddings", query_embeddings.shape, query_embeddings.dtype)
    print("query_chunks", query_chunks.shape, query_chunks.dtype, "sum", int(query_chunks.sum()))
    print("query_chunk_counts", query_chunks.tolist())


def write_doc_shards(output_root: Path, docs_per_shard: int) -> None:
    docdata, _, _ = ensure_dirs(output_root)
    base_embeddings, base_chunks, _, _ = load_raw_dataset()
    doc_offsets = build_doc_offsets(base_chunks)
    num_docs = len(base_chunks)
    num_shards = math.ceil(num_docs / docs_per_shard)

    for shard_idx in tqdm(range(num_shards), desc="Writing doc shards"):
        start_doc = shard_idx * docs_per_shard
        end_doc = min((shard_idx + 1) * docs_per_shard, num_docs)

        start_vec = int(doc_offsets[start_doc])
        end_vec = int(doc_offsets[end_doc])

        shard_embeddings = np.asarray(base_embeddings[start_vec:end_vec], dtype=np.float16)
        shard_chunks = np.asarray(base_chunks[start_doc:end_doc], dtype=np.int64)

        np.save(docdata / f"encoding{shard_idx}_float16.npy", shard_embeddings)
        np.save(docdata / f"doclens{shard_idx}.npy", shard_chunks)


def write_query_embeddings(output_root: Path) -> None:
    _, _, qdata = ensure_dirs(output_root)
    _, _, query_embeddings, query_chunks = load_raw_dataset()

    unique_counts = np.unique(query_chunks)
    if len(unique_counts) == 1:
        query_len = int(unique_counts[0])
        num_queries = len(query_chunks)
        query_tensor = np.asarray(query_embeddings, dtype=np.float32).reshape(num_queries, query_len, query_embeddings.shape[1])
        np.save(qdata / "qembs.npy", query_tensor)
    else:
        np.save(qdata / "filterd_query.npy", np.asarray(query_embeddings, dtype=np.float32))
        np.save(qdata / "filterd_query_len.npy", np.asarray(query_chunks, dtype=np.int64))


def faiss_gpu_available() -> bool:
    return hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0


def train_kmeans(
    data: np.ndarray, k: int, niter: int, seed: int, spherical: bool, use_gpu: bool
) -> np.ndarray:
    kmeans = faiss.Kmeans(
        d=data.shape[1],
        k=k,
        niter=niter,
        verbose=True,
        spherical=spherical,
        seed=seed,
        gpu=use_gpu,
    )
    kmeans.train(np.ascontiguousarray(data, dtype=np.float32))
    return kmeans.centroids.astype(np.float32, copy=False)


def sample_base_vectors(sample_size: int, seed: int) -> np.ndarray:
    base_embeddings = load_memmap(RAW_BASE_EMBEDDINGS)
    total = base_embeddings.shape[0]
    if sample_size > total:
        sample_size = total
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=sample_size, replace=False)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(len(order), dtype=order.dtype)
    sample = np.asarray(base_embeddings[sorted_indices], dtype=np.float32)[inverse_order]
    return sample


def write_fine_centroids(
    output_root: Path, fine_k: int, sample_size: int, niter: int, seed: int, use_gpu: bool
) -> None:
    _, cdata, _ = ensure_dirs(output_root)
    sample = sample_base_vectors(sample_size=sample_size, seed=seed)
    centroids = train_kmeans(
        sample, k=fine_k, niter=niter, seed=seed, spherical=True, use_gpu=use_gpu
    )
    np.save(cdata / "centroids.npy", centroids.astype(np.float16))


def assign_codes(output_root: Path, docs_per_shard: int, batch_size: int, use_gpu: bool) -> None:
    docdata, cdata, _ = ensure_dirs(output_root)
    base_embeddings, base_chunks, _, _ = load_raw_dataset()
    doc_offsets = build_doc_offsets(base_chunks)
    centroids = np.asarray(np.load(cdata / "centroids.npy"), dtype=np.float32)

    cpu_index = faiss.IndexFlatIP(centroids.shape[1])
    if use_gpu:
        index = faiss.index_cpu_to_all_gpus(cpu_index)
    else:
        index = cpu_index
    index.add(np.ascontiguousarray(centroids))

    num_docs = len(base_chunks)
    num_shards = math.ceil(num_docs / docs_per_shard)

    for shard_idx in tqdm(range(num_shards), desc="Assigning fine codes"):
        start_doc = shard_idx * docs_per_shard
        end_doc = min((shard_idx + 1) * docs_per_shard, num_docs)
        start_vec = int(doc_offsets[start_doc])
        end_vec = int(doc_offsets[end_doc])

        codes = np.empty(end_vec - start_vec, dtype=np.int32)
        out_pos = 0
        for batch_start in range(start_vec, end_vec, batch_size):
            batch_end = min(batch_start + batch_size, end_vec)
            batch = np.asarray(base_embeddings[batch_start:batch_end], dtype=np.float32)
            _, batch_codes = index.search(np.ascontiguousarray(batch), 1)
            batch_codes = batch_codes.reshape(-1).astype(np.int32, copy=False)
            codes[out_pos:out_pos + len(batch_codes)] = batch_codes
            out_pos += len(batch_codes)

        np.save(docdata / f"doc_codes_{shard_idx}.npy", codes)


def write_coarse_centroids(
    output_root: Path, coarse_k: int, niter: int, seed: int, use_gpu: bool
) -> None:
    _, cdata, _ = ensure_dirs(output_root)
    fine_centroids = np.asarray(np.load(cdata / "centroids.npy"), dtype=np.float32)
    coarse_centroids = train_kmeans(
        fine_centroids, k=coarse_k, niter=niter, seed=seed, spherical=True, use_gpu=use_gpu
    )
    np.save(cdata / "coarse_centroids.npy", coarse_centroids.astype(np.float32))


def build_coarse_cluster_info(output_root: Path, docs_per_shard: int, top_r: int) -> None:
    docdata, cdata, _ = ensure_dirs(output_root)
    base_chunks = np.asarray(load_memmap(RAW_BASE_CHUNKS), dtype=np.int64)
    num_docs = len(base_chunks)
    num_shards = math.ceil(num_docs / docs_per_shard)

    fine_centroids = np.asarray(np.load(cdata / "centroids.npy"), dtype=np.float32)
    coarse_centroids = np.asarray(np.load(cdata / "coarse_centroids.npy"), dtype=np.float32)

    coarse_index = faiss.IndexFlatIP(coarse_centroids.shape[1])
    coarse_index.add(np.ascontiguousarray(coarse_centroids))
    _, fine_to_coarse = coarse_index.search(np.ascontiguousarray(fine_centroids), 1)
    fine_to_coarse = fine_to_coarse.reshape(-1).astype(np.int32, copy=False)

    doc_coarse_counts = np.full((num_docs, top_r), -1, dtype=np.int32)
    doc_coarse_scores = np.zeros((num_docs, top_r), dtype=np.float32)
    doc_freq = np.zeros(coarse_centroids.shape[0], dtype=np.int64)

    doc_cursor = 0
    for shard_idx in tqdm(range(num_shards), desc="Counting coarse TF"):
        shard_chunks = np.asarray(np.load(docdata / f"doclens{shard_idx}.npy"), dtype=np.int64)
        shard_codes = np.asarray(np.load(docdata / f"doc_codes_{shard_idx}.npy"), dtype=np.int32)
        local_cursor = 0
        for chunk_len in shard_chunks:
            token_codes = shard_codes[local_cursor:local_cursor + chunk_len]
            local_cursor += chunk_len

            token_coarse = fine_to_coarse[token_codes]
            coarse_ids, tf_counts = np.unique(token_coarse, return_counts=True)
            doc_freq[coarse_ids] += 1

            keep = min(top_r, len(coarse_ids))
            order = np.argsort(tf_counts)[::-1][:keep]
            doc_coarse_counts[doc_cursor, :keep] = coarse_ids[order]
            doc_coarse_scores[doc_cursor, :keep] = tf_counts[order].astype(np.float32)
            doc_cursor += 1

    idf = np.log(num_docs / (1.0 + doc_freq.astype(np.float64)))

    cluster_members = [[] for _ in range(coarse_centroids.shape[0])]
    for doc_id in tqdm(range(num_docs), desc="Building coarse cluster memberships"):
        coarse_ids = doc_coarse_counts[doc_id]
        mask = coarse_ids >= 0
        coarse_ids = coarse_ids[mask]
        if len(coarse_ids) == 0:
            continue
        tf = doc_coarse_scores[doc_id, :len(coarse_ids)]
        scores = tf * idf[coarse_ids]
        order = np.argsort(scores)[::-1]
        chosen = coarse_ids[order[: min(top_r, len(order))]]
        for coarse_id in chosen:
            cluster_members[int(coarse_id)].append(doc_id)

    coarse_info_path = cdata / "coarse_cluster_info.txt"
    with coarse_info_path.open("w", encoding="utf-8") as f:
        for members in cluster_members:
            f.write(" ".join(str(doc_id) for doc_id in members))
            f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate MSMARCO GEM input files from raw multi-vector data.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ali/gem-baseline/gem_data/msmarco"),
        help="Target GEM dataset directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("/data/ali/msmarco-large-multi"),
        help="Directory containing raw multi-vector dataset files.",
    )
    parser.add_argument(
        "--dataset-stem",
        type=str,
        default="msmarco-large",
        help="Stem used in raw filenames, e.g. msmarco-large or scidocs-large.",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "inspect",
            "docdata",
            "queries",
            "fine-centroids",
            "codes",
            "coarse-centroids",
            "coarse-info",
        ],
        help="Pipeline stage to execute.",
    )
    parser.add_argument("--docs-per-shard", type=int, default=25000)
    parser.add_argument("--fine-k", type=int, default=262144)
    parser.add_argument("--coarse-k", type=int, default=40960)
    parser.add_argument("--sample-size", type=int, default=500000)
    parser.add_argument("--niter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--top-r", type=int, default=3)
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU FAISS when available for clustering/search-heavy preprocessing stages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configure_raw_paths(args.dataset_dir, args.dataset_stem)
    use_gpu = args.use_gpu and faiss_gpu_available()
    if args.use_gpu and not use_gpu:
        print("GPU FAISS requested, but GPU support is unavailable in the current FAISS build.")
    if use_gpu:
        print(f"Using GPU FAISS across {faiss.get_num_gpus()} visible GPU(s).")
    if args.stage == "inspect":
        inspect_dataset()
    elif args.stage == "docdata":
        write_doc_shards(args.output_root, docs_per_shard=args.docs_per_shard)
    elif args.stage == "queries":
        write_query_embeddings(args.output_root)
    elif args.stage == "fine-centroids":
        write_fine_centroids(
            args.output_root,
            fine_k=args.fine_k,
            sample_size=args.sample_size,
            niter=args.niter,
            seed=args.seed,
            use_gpu=use_gpu,
        )
    elif args.stage == "codes":
        assign_codes(
            args.output_root,
            docs_per_shard=args.docs_per_shard,
            batch_size=args.batch_size,
            use_gpu=use_gpu,
        )
    elif args.stage == "coarse-centroids":
        write_coarse_centroids(
            args.output_root,
            coarse_k=args.coarse_k,
            niter=args.niter,
            seed=args.seed,
            use_gpu=use_gpu,
        )
    elif args.stage == "coarse-info":
        build_coarse_cluster_info(
            args.output_root,
            docs_per_shard=args.docs_per_shard,
            top_r=args.top_r,
        )


if __name__ == "__main__":
    main()
