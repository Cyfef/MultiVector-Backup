#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
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
    default_index_name,
    evaluate_metrics,
    infer_split,
    load_beir_split,
    resolve_corpus_ids,
    resolve_dataset_dir,
    resolve_embedding_dir,
    resolve_query_ids_and_indices,
)
from warp.infra.config import ColBERTConfig  # noqa: E402
from warp.modeling.colbert import ColBERT  # noqa: E402
from warp.search.index_storage import IndexScorer  # noqa: E402
from warp.utils.tracker import NOPTracker  # noqa: E402


DEFAULT_NCELLS = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def configure_torch_threads_from_env() -> None:
    value = os.environ.get("TORCH_NUM_THREADS")
    if not value:
        return

    threads = int(value)
    if threads < 1:
        raise ValueError("TORCH_NUM_THREADS must be >= 1")

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep WARP search quality over ncells and save metrics to CSV."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--dataset-dir",
        help="Optional direct path to a BEIR-style dataset directory.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/data1/liuyaoyang/Papers/ACFDE/datasets",
    )
    parser.add_argument(
        "--embedding-dir",
        help="Optional direct path to a packed embedding directory.",
    )
    parser.add_argument(
        "--embedding-root",
        default="/data1/liuyaoyang/Papers/ACFDE/output",
    )
    parser.add_argument("--encoder", default="colbert")
    parser.add_argument("--index")
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--nbits", type=int, default=2, choices=[1, 2, 4, 8])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--split")
    parser.add_argument("--baseline")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "resutls"),
    )
    parser.add_argument(
        "--output-csv",
        help="Optional explicit output CSV path.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows to an existing CSV instead of overwriting it.",
    )
    parser.add_argument(
        "--ncells",
        type=int,
        nargs="+",
        default=DEFAULT_NCELLS,
    )
    parser.add_argument("--centroid-score-threshold", type=float)
    parser.add_argument("--ndocs", type=int)
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Evaluate only the first N queries in BEIR/qrels order.",
    )
    parser.add_argument(
        "--metrics-k",
        type=int,
        nargs="+",
        default=[10, 100],
    )
    parser.add_argument(
        "--tsv-output",
        help="Optional TSV path for ranked retrieval results. Requires exactly one ncells value.",
    )
    return parser.parse_args()


def metric_column_name(prefix: str, k: int) -> str:
    return f"{prefix.lower()}_{k}"


def default_baseline_name(encoder: str, nbits: int) -> str:
    return f"warp_precomputed_{encoder}_nbits{nbits}"


def default_output_csv(
    output_dir: Path,
    dataset: str,
    split: str,
    baseline: str,
) -> Path:
    filename = f"{dataset}.split={split}.baseline={baseline}.ncells_sweep.csv"
    return output_dir / filename


def load_query_tensor(
    query_points: np.memmap,
    query_offsets: np.ndarray,
    packed_idx: int,
) -> torch.Tensor:
    start = int(query_offsets[packed_idx])
    end = int(query_offsets[packed_idx + 1])
    query = np.array(query_points[start:end], dtype=np.float32, copy=True)
    return torch.from_numpy(query).unsqueeze(0)


def write_results_tsv(path: Path, results: dict[str, dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "rank", "score"])
        for qid, rels in results.items():
            ranked = sorted(rels.items(), key=lambda item: item[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                writer.writerow([qid, docid, rank, score])


def run_sweep(args: argparse.Namespace) -> Path:
    if args.tsv_output and len(args.ncells) != 1:
        raise ValueError("--tsv-output requires exactly one ncells value")

    split = args.split or infer_split(args.dataset)
    baseline = args.baseline or default_baseline_name(args.encoder, args.nbits)
    index_name = args.index or default_index_name(args.dataset, args.encoder, args.nbits)
    index_path = Path(args.index_root) / index_name

    embedding_dir = resolve_embedding_dir(args)
    query_points = np.load(embedding_dir / "query_points.npy", mmap_mode="r")
    query_offsets = np.load(embedding_dir / "query_offsets.npy").astype(np.int64, copy=False)

    dataset_dir = resolve_dataset_dir(args)
    corpus, queries, qrels = load_beir_split(dataset_dir, split)
    corpus_ids = resolve_corpus_ids(corpus, embedding_dir)
    query_ids = list(queries.keys())

    if args.max_queries is not None:
        if args.max_queries < 1:
            raise ValueError("--max-queries must be >= 1")
        query_ids = query_ids[: args.max_queries]
        qrels = {qid: qrels[qid] for qid in query_ids}

    query_ids, query_indices = resolve_query_ids_and_indices(
        query_ids, query_offsets, embedding_dir
    )

    ColBERT.try_load_torch_extensions(False)
    scorer = IndexScorer(str(index_path), use_gpu=torch.cuda.is_available())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else default_output_csv(output_dir, args.dataset, split, baseline)
    )

    metric_columns: list[str] = []
    for k in args.metrics_k:
        metric_columns.extend(
            [
                metric_column_name("ndcg", k),
                metric_column_name("map", k),
                metric_column_name("recall", k),
                metric_column_name("p", k),
                metric_column_name("mrr", k),
            ]
        )

    fieldnames = [
        "dataset",
        "split",
        "baseline",
        "index_name",
        "index_path",
        "encoder",
        "nbits",
        "k",
        "ncells",
        "centroid_score_threshold",
        "ndocs",
        "num_queries",
        "query_limit",
        "elapsed_sec",
        "qps",
    ] + metric_columns

    write_header = True
    mode = "w"
    if args.append and output_csv.exists() and output_csv.stat().st_size > 0:
        write_header = False
        mode = "a"

    with open(output_csv, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for ncells in args.ncells:
            config = ColBERTConfig.load_from_index(str(index_path))
            config = configure_search(
                args.k,
                config,
                ncells=ncells,
                centroid_score_threshold=args.centroid_score_threshold,
                ndocs=args.ndocs,
            )

            started = time.perf_counter()
            results: dict[str, dict[str, float]] = {}

            for qid, packed_idx in zip(query_ids, query_indices):
                query = load_query_tensor(query_points, query_offsets, packed_idx)
                pids, scores = scorer.rank(config, query, tracker=NOPTracker())
                results[qid] = {
                    corpus_ids[pid]: float(score)
                    for pid, score in zip(pids[: args.k], scores[: args.k])
                }

            elapsed = time.perf_counter() - started
            ndcg, mean_ap, recall, precision, mrr = evaluate_metrics(
                qrels, results, args.metrics_k
            )

            row = {
                "dataset": args.dataset,
                "split": split,
                "baseline": baseline,
                "index_name": index_name,
                "index_path": str(index_path),
                "encoder": args.encoder,
                "nbits": args.nbits,
                "k": args.k,
                "ncells": ncells,
                "centroid_score_threshold": config.centroid_score_threshold,
                "ndocs": config.ndocs,
                "num_queries": len(query_ids),
                "query_limit": args.max_queries or len(query_ids),
                "elapsed_sec": round(elapsed, 3),
                "qps": round(len(query_ids) / elapsed, 3),
            }

            for k in args.metrics_k:
                row[metric_column_name("ndcg", k)] = ndcg[f"NDCG@{k}"]
                row[metric_column_name("map", k)] = mean_ap[f"MAP@{k}"]
                row[metric_column_name("recall", k)] = recall[f"Recall@{k}"]
                row[metric_column_name("p", k)] = precision[f"P@{k}"]
                row[metric_column_name("mrr", k)] = mrr[f"MRR@{k}"]

            if args.tsv_output:
                write_results_tsv(Path(args.tsv_output), results)

            writer.writerow(row)
            f.flush()

            print(
                f"ncells={ncells} "
                f"NDCG@10={row.get('ndcg_10', 'NA')} "
                f"NDCG@100={row.get('ndcg_100', 'NA')} "
                f"Recall@100={row.get('recall_100', 'NA')} "
                f"elapsed={row['elapsed_sec']}s"
            )

    return output_csv


def main() -> None:
    configure_torch_threads_from_env()
    args = parse_args()
    output_csv = run_sweep(args)
    print(f"Saved sweep results to {output_csv}")


if __name__ == "__main__":
    main()
