import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


FILE_ABS_PATH = os.path.dirname(__file__)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir)
sys.path.append(ROOT_PATH)
ROOT_PATH = os.path.join(FILE_ABS_PATH, os.pardir, os.pardir, "baseline", "ColBERT")
sys.path.append(ROOT_PATH)

from baseline.ColBERT.run import load_precomputed_query_embeddings


def ensure_symlink(link_path: Path, source_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        return
    link_path.symlink_to(source_path)


def ensure_raw_metadata(username: str, dataset: str, dataset_dir: str, n_query: int) -> None:
    raw_document_path = Path(f"/data/{username}/Dataset/multi-vector-retrieval/RawData/{dataset}/document")
    raw_document_path.mkdir(parents=True, exist_ok=True)

    transformed_embeddings = Path(dataset_dir) / "doc_embeddings" / "transformed_embeddings"
    transformed_link = raw_document_path / "transformed_embeddings"
    ensure_symlink(transformed_link, transformed_embeddings)

    gt_file = Path(dataset_dir) / "query_embeddings" / "transformed_embeddings" / "query_doc_mappings.jsonl"
    gt_link = raw_document_path / "queries.gnd.jsonl"
    if gt_file.exists():
        ensure_symlink(gt_link, gt_file)

    collection_path = raw_document_path / "collection.tsv"
    if not collection_path.exists():
        doc_count_file = transformed_embeddings / "doc_count"
        doc_count = int(torch.load(doc_count_file))
        with collection_path.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerows([[idx] for idx in range(doc_count)])

    queries_path = raw_document_path / "queries.dev.tsv"
    with queries_path.open("w") as f:
        for idx in range(n_query):
            f.write(f"{idx}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate canonical Plaid query artifacts from existing query embeddings.")
    parser.add_argument("--username", type=str, default="ali")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--query-embedding-file", type=str, required=True)
    parser.add_argument("--query-len-file", type=str, required=True)
    args = parser.parse_args()

    embedding_path = Path(f"/data/{args.username}/Dataset/multi-vector-retrieval/Embedding/{args.dataset}")
    result_performance_path = Path(f"/data/{args.username}/Dataset/multi-vector-retrieval/Result/performance")
    embedding_path.mkdir(parents=True, exist_ok=True)
    result_performance_path.mkdir(parents=True, exist_ok=True)

    query_embedding_filename = embedding_path / "query_embedding.npy"
    query_embedding_len_filename = embedding_path / "query_n_vec_length.npy"

    if not os.path.exists(args.query_embedding_file):
        raise FileNotFoundError(f"missing query embedding file: {args.query_embedding_file}")
    if not os.path.exists(args.query_len_file):
        raise FileNotFoundError(f"missing query length file: {args.query_len_file}")

    start = time.perf_counter()
    query_embeddings = load_precomputed_query_embeddings(
        input_query_embedding_file=args.query_embedding_file,
        query_embedding_len_file=args.query_len_file,
    )
    query_lengths = np.load(args.query_len_file)

    if len(query_embeddings) != len(query_lengths):
        raise ValueError(
            f"query embedding count {len(query_embeddings)} does not match query length count {len(query_lengths)}"
        )

    np.save(query_embedding_filename, query_embeddings)
    np.save(query_embedding_len_filename, query_lengths)
    ensure_raw_metadata(
        username=args.username,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        n_query=len(query_embeddings),
    )

    total_encode_time_ms = (time.perf_counter() - start) * 1000.0
    encode_info = {
        "total_encode_time_ms": total_encode_time_ms,
        "n_encode_query": int(len(query_embeddings)),
        "average_encode_time_ms": total_encode_time_ms / max(len(query_embeddings), 1),
        "source": "precomputed_query_embeddings",
    }
    with (result_performance_path / f"{args.dataset}-encode_query.json").open("w") as f:
        json.dump(encode_info, f)

    print(f"saved query embeddings to {query_embedding_filename}")
    print(f"saved query lengths to {query_embedding_len_filename}")


if __name__ == "__main__":
    main()
