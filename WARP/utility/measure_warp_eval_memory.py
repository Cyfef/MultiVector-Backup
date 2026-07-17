#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.evaluate_precomputed_queries import (  # noqa: E402
    configure_search,
    infer_split,
    load_beir_split,
    resolve_corpus_ids,
    resolve_dataset_dir,
    resolve_embedding_dir,
    resolve_query_ids_and_indices,
)
from utility.sweep_ncells import load_query_tensor  # noqa: E402
from warp.infra.config import ColBERTConfig  # noqa: E402
from warp.modeling.colbert import ColBERT  # noqa: E402
from warp.search.index_storage import IndexScorer  # noqa: E402
from warp.utils.tracker import NOPTracker  # noqa: E402


def configure_torch_threads_from_env() -> None:
    value = os.environ.get("TORCH_NUM_THREADS")
    if not value:
        return
    threads = int(value)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-query WARP retrieval and report peak resident memory."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir")
    parser.add_argument(
        "--dataset-root",
        default="/data1/liuyaoyang/Papers/ACFDE/datasets",
    )
    parser.add_argument("--embedding-dir")
    parser.add_argument(
        "--embedding-root",
        default="/data1/liuyaoyang/Papers/ACFDE/output",
    )
    parser.add_argument("--encoder", default="colbert")
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--split")
    parser.add_argument("--nbits", type=int, default=2)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ncells", type=int, required=True)
    parser.add_argument("--centroid-score-threshold", type=float, required=True)
    parser.add_argument("--ndocs", type=int, required=True)
    parser.add_argument("--max-queries", type=int, default=1)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def peak_rss_gib() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KiB.
    return usage / (1024.0 * 1024.0)


def main() -> None:
    configure_torch_threads_from_env()
    args = parse_args()

    split = args.split or infer_split(args.dataset)
    dataset_dir = resolve_dataset_dir(args)
    embedding_dir = resolve_embedding_dir(args)
    index_path = Path(args.index_root) / args.index

    started = time.perf_counter()

    query_points = np.load(embedding_dir / "query_points.npy", mmap_mode="r")
    query_offsets = np.load(embedding_dir / "query_offsets.npy").astype(np.int64, copy=False)
    corpus, queries, qrels = load_beir_split(dataset_dir, split)
    corpus_ids = resolve_corpus_ids(corpus, embedding_dir)
    all_query_ids = list(queries.keys())
    resolved_query_ids, resolved_query_indices = resolve_query_ids_and_indices(
        all_query_ids, query_offsets, embedding_dir
    )
    query_ids = resolved_query_ids[: args.max_queries]
    query_indices = resolved_query_indices[: args.max_queries]
    qrels = {qid: qrels[qid] for qid in query_ids}

    after_data_rss = peak_rss_gib()

    ColBERT.try_load_torch_extensions(False)
    scorer = IndexScorer(str(index_path), use_gpu=False)
    after_index_rss = peak_rss_gib()

    config = ColBERTConfig.load_from_index(str(index_path))
    config = configure_search(
        args.k,
        config,
        ncells=args.ncells,
        centroid_score_threshold=args.centroid_score_threshold,
        ndocs=args.ndocs,
    )

    for packed_idx in query_indices:
        query = load_query_tensor(query_points, query_offsets, packed_idx)
        pids, scores = scorer.rank(config, query, tracker=NOPTracker())
        _ = {
            corpus_ids[pid]: float(score)
            for pid, score in zip(pids[: args.k], scores[: args.k])
        }

    total_elapsed = time.perf_counter() - started
    peak_rss = peak_rss_gib()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "split",
        "index_name",
        "index_path",
        "ncells",
        "centroid_score_threshold",
        "ndocs",
        "k",
        "max_queries",
        "rss_after_data_gib",
        "rss_after_index_gib",
        "rss_peak_gib",
        "elapsed_sec",
    ]

    row = {
        "dataset": args.dataset,
        "split": split,
        "index_name": args.index,
        "index_path": str(index_path),
        "ncells": args.ncells,
        "centroid_score_threshold": args.centroid_score_threshold,
        "ndocs": args.ndocs,
        "k": args.k,
        "max_queries": args.max_queries,
        "rss_after_data_gib": f"{after_data_rss:.3f}",
        "rss_after_index_gib": f"{after_index_rss:.3f}",
        "rss_peak_gib": f"{peak_rss:.3f}",
        "elapsed_sec": f"{total_elapsed:.3f}",
    }

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(row)


if __name__ == "__main__":
    main()
