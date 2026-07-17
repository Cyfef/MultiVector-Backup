#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.evaluate_precomputed_queries import (  # noqa: E402
    configure_search,
    evaluate_metrics,
    infer_split,
    load_beir_split,
    resolve_corpus_ids,
    resolve_query_ids_and_indices,
)
from utility.sweep_ncells import load_query_tensor  # noqa: E402
from warp.infra.config import ColBERTConfig  # noqa: E402
from warp.modeling.colbert import ColBERT  # noqa: E402
from warp.search.index_storage import IndexScorer  # noqa: E402
from warp.utils.tracker import NOPTracker  # noqa: E402


DEFAULT_DATASET_DIRS = {
    "clef-modern-colbert": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef",
    "clerc-modern-colbert": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc",
    "clef": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef",
    "clerc": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc",
    "fiqa-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa",
    "nq-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/nq",
    "scidocs-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs",
    "msmarco-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/msmarco",
}

DEFAULT_EMBEDDING_DIRS = {
    "clef-modern-colbert": "/data/ali/clef-modern-colbert",
    "clerc-modern-colbert": "/data/ali/clerc-modern-colbert",
    "scidocs-modern-colbert": "/data/ali/scidocs-modern-colbert",
    "nq-modern-colbert": "/data/ali/nq-modern-colbert",
    "fiqa-modern-colbert": "/data/ali/fiqa-modern-colbert",
    "clef": "/data/ali/clef-colbert",
    "clerc": "/data/ali/clerc-colbert",
}


@dataclass
class EvalContext:
    query_points: np.memmap
    query_offsets: np.ndarray
    corpus_ids: list[str]
    query_ids: list[str]
    query_indices: list[int]
    qrels: dict[str, dict[str, int]]
    scorer: IndexScorer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill MRR columns into existing sweep CSVs.")
    parser.add_argument("csv_paths", nargs="+", help="CSV files to update in place.")
    parser.add_argument("--log-file", help="Optional progress log file.")
    parser.add_argument(
        "--tsv-root",
        help="Optional root directory for per-setting TSV retrieval outputs.",
    )
    return parser.parse_args()


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


LOG_FILE: Path | None = None
TSV_ROOT: Path | None = None


def log(message: str) -> None:
    print(message, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def infer_dataset_dir(dataset: str) -> Path:
    if dataset in DEFAULT_DATASET_DIRS:
        return Path(DEFAULT_DATASET_DIRS[dataset])
    return Path("/data1/liuyaoyang/Papers/ACFDE/datasets") / dataset


def infer_embedding_dir(dataset: str) -> Path:
    if dataset in DEFAULT_EMBEDDING_DIRS:
        return Path(DEFAULT_EMBEDDING_DIRS[dataset])
    return Path("/data1/liuyaoyang/Papers/ACFDE/output") / dataset / "colbert"


def sanitize_filename_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def tsv_output_path(row: dict[str, str]) -> Path:
    assert TSV_ROOT is not None
    dataset_dir = TSV_ROOT / row["dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    query_limit = row.get("query_limit") or row.get("num_queries") or ""
    filename = (
        f"split={sanitize_filename_part(row['split'])}."
        f"baseline={sanitize_filename_part(row['baseline'])}."
        f"ncells={row['ncells']}."
        f"thr={row['centroid_score_threshold']}."
        f"ndocs={row['ndocs']}."
        f"k={row['k']}."
        f"queries={query_limit}.tsv"
    )
    return dataset_dir / filename


def write_results_tsv(path: Path, results: dict[str, dict[str, float]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["query-id", "corpus-id", "rank", "score"])
        for qid, rels in results.items():
            ranked = sorted(rels.items(), key=lambda item: item[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                writer.writerow([qid, docid, rank, score])


def build_context(row: dict[str, str]) -> EvalContext:
    dataset = row["dataset"]
    split = row.get("split") or infer_split(dataset)
    dataset_dir = infer_dataset_dir(dataset)
    embedding_dir = infer_embedding_dir(dataset)
    index_path = Path(row["index_path"])

    query_points = np.load(embedding_dir / "query_points.npy", mmap_mode="r")
    query_offsets = np.load(embedding_dir / "query_offsets.npy").astype(np.int64, copy=False)

    corpus, queries, qrels = load_beir_split(dataset_dir, split)
    corpus_ids = resolve_corpus_ids(corpus, embedding_dir)
    query_ids = list(queries.keys())

    query_limit_raw = row.get("query_limit") or row.get("num_queries")
    if query_limit_raw:
        query_limit = int(query_limit_raw)
        query_ids = query_ids[:query_limit]
        qrels = {qid: qrels[qid] for qid in query_ids}

    query_ids, query_indices = resolve_query_ids_and_indices(query_ids, query_offsets, embedding_dir)

    ColBERT.try_load_torch_extensions(False)
    scorer = IndexScorer(str(index_path), use_gpu=False)

    return EvalContext(
        query_points=query_points,
        query_offsets=query_offsets,
        corpus_ids=corpus_ids,
        query_ids=query_ids,
        query_indices=query_indices,
        qrels=qrels,
        scorer=scorer,
    )


def run_retrieval_for_row(
    row: dict[str, str],
    context: EvalContext,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    metrics_k = [10, 100]
    config = ColBERTConfig.load_from_index(str(row["index_path"]))
    config = configure_search(
        int(row["k"]),
        config,
        ncells=int(row["ncells"]),
        centroid_score_threshold=float(row["centroid_score_threshold"]),
        ndocs=int(row["ndocs"]),
    )

    results: dict[str, dict[str, float]] = {}
    for qid, packed_idx in zip(context.query_ids, context.query_indices):
        query = load_query_tensor(context.query_points, context.query_offsets, packed_idx)
        pids, scores = context.scorer.rank(config, query, tracker=NOPTracker())
        results[qid] = {
            context.corpus_ids[pid]: float(score)
            for pid, score in zip(pids[: int(row["k"])], scores[: int(row["k"])])
        }

    _, _, _, _, mrr = evaluate_metrics(context.qrels, results, metrics_k)
    return results, {f"mrr_{k}": mrr[f"MRR@{k}"] for k in metrics_k}


def update_csv(path: Path) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    has_mrr = "mrr_10" in fieldnames and "mrr_100" in fieldnames
    if has_mrr and TSV_ROOT is None:
        log(f"Skipping {path}: already has MRR columns")
        return

    new_fieldnames = fieldnames if has_mrr else fieldnames + ["mrr_10", "mrr_100"]

    grouped_rows: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = (row["dataset"], row["index_path"], row.get("query_limit") or row.get("num_queries") or "")
        grouped_rows[key].append((idx, row))

    total = len(rows)
    processed = 0

    for _, group in grouped_rows.items():
        context = build_context(group[0][1])
        for idx, row in group:
            results, mrr_values = run_retrieval_for_row(row, context)
            if not has_mrr:
                row.update(mrr_values)
            if TSV_ROOT is not None:
                output_path = tsv_output_path(row)
                write_results_tsv(output_path, results)
            processed += 1
            log(
                f"{path.name}: row {processed}/{total} "
                f"mrr_10={mrr_values['mrr_10']} mrr_100={mrr_values['mrr_100']}"
            )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global LOG_FILE, TSV_ROOT
    configure_torch_threads_from_env()
    args = parse_args()
    if args.log_file:
        LOG_FILE = Path(args.log_file)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text("", encoding="utf-8")
        log(f"Starting MRR backfill for {len(args.csv_paths)} CSV files")
    if args.tsv_root:
        TSV_ROOT = Path(args.tsv_root)
        TSV_ROOT.mkdir(parents=True, exist_ok=True)
        log(f"Saving TSV retrieval outputs under {TSV_ROOT}")
    for csv_path in args.csv_paths:
        log(f"Processing {csv_path}")
        update_csv(Path(csv_path))


if __name__ == "__main__":
    main()
