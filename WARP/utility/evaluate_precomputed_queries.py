#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOCAL_DEPS = REPO_ROOT / ".deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from warp.infra.config import ColBERTConfig
from warp.modeling.colbert import ColBERT
from warp.search.index_storage import IndexScorer
from warp.utils.tracker import NOPTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a WARP/ColBERT index with precomputed query embeddings."
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
    parser.add_argument("--index-root", default=os.environ.get("INDEX_ROOT"))
    parser.add_argument("--nbits", type=int, default=2, choices=[1, 2, 4, 8])
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--split")
    parser.add_argument("--ncells", type=int)
    parser.add_argument("--centroid-score-threshold", type=float)
    parser.add_argument("--ndocs", type=int)
    parser.add_argument(
        "--metrics-k",
        type=int,
        nargs="+",
        default=[10, 100],
    )
    return parser.parse_args()


def infer_split(dataset: str) -> str:
    return "dev" if dataset in {"msmarco", "msmarco_small"} else "test"


def default_index_name(dataset: str, encoder: str, nbits: int) -> str:
    split = infer_split(dataset)
    return f"beir-{dataset}.split={split}.precomputed={encoder}.nbits={nbits}"


def resolve_dataset_dir(args: argparse.Namespace) -> Path:
    if args.dataset_dir is not None:
        return Path(args.dataset_dir)
    return Path(args.dataset_root) / args.dataset


def resolve_embedding_dir(args: argparse.Namespace) -> Path:
    if args.embedding_dir is not None:
        return Path(args.embedding_dir)
    return Path(args.embedding_root) / args.dataset / args.encoder


def configure_search(
    k: int,
    config: ColBERTConfig,
    ncells: int | None = None,
    centroid_score_threshold: float | None = None,
    ndocs: int | None = None,
) -> ColBERTConfig:
    if k <= 10:
        config.ncells = 1
        config.centroid_score_threshold = 0.5
        config.ndocs = 256
    elif k <= 100:
        config.ncells = 2
        config.centroid_score_threshold = 0.45
        config.ndocs = 1024
    else:
        config.ncells = 4
        config.centroid_score_threshold = 0.4
        config.ndocs = max(k * 4, 4096)

    if ncells is not None:
        config.ncells = ncells
    if centroid_score_threshold is not None:
        config.centroid_score_threshold = centroid_score_threshold
    if ndocs is not None:
        config.ndocs = ndocs

    return config


def load_jsonl_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            docid = str(row["corpus-id"])
            score = int(row["score"])
            qrels.setdefault(qid, {})[docid] = score
    return qrels


def load_beir_split(
    dataset_dir: Path,
    split: str,
) -> tuple[dict[str, dict[str, str | None]], dict[str, str], dict[str, dict[str, int]]]:
    corpus_rows = load_jsonl_rows(dataset_dir / "corpus.jsonl")
    query_rows = load_jsonl_rows(dataset_dir / "queries.jsonl")
    qrels = load_qrels(dataset_dir / "qrels" / f"{split}.tsv")

    corpus = {
        str(row["_id"]): {
            "text": row.get("text"),
            "title": row.get("title"),
        }
        for row in corpus_rows
    }
    all_queries = {str(row["_id"]): str(row["text"]) for row in query_rows}

    # Match BEIR GenericDataLoader.load(split): keep only split queries and
    # preserve the qrels file order rather than queries.jsonl order.
    queries = {qid: all_queries[qid] for qid in qrels}

    return corpus, queries, qrels


def load_packed_ids(path: Path) -> list[str]:
    with open(path) as f:
        return [str(value) for value in json.load(f)]


def resolve_corpus_ids(
    corpus: dict[str, dict[str, str | None]],
    embedding_dir: Path,
) -> list[str]:
    packed_ids_path = embedding_dir / "corpus_ids.json"
    if not packed_ids_path.exists():
        return list(corpus.keys())

    corpus_ids = load_packed_ids(packed_ids_path)
    if len(corpus_ids) != len(corpus):
        raise ValueError(
            f"Corpus count mismatch: {len(corpus_ids)} packed ids vs {len(corpus)} corpus rows"
        )
    return corpus_ids


def resolve_query_ids_and_indices(
    query_ids: list[str],
    query_offsets: np.ndarray,
    embedding_dir: Path,
) -> tuple[list[str], list[int]]:
    packed_count = int(query_offsets.size - 1)
    packed_ids_path = embedding_dir / "query_ids.json"
    if not packed_ids_path.exists():
        if len(query_ids) != packed_count:
            raise ValueError(
                f"Query count mismatch: {len(query_ids)} qids vs {packed_count} packed queries"
            )
        return query_ids, list(range(packed_count))

    packed_query_ids = load_packed_ids(packed_ids_path)
    if len(packed_query_ids) != packed_count:
        raise ValueError(
            f"Packed query id count mismatch: {len(packed_query_ids)} ids vs {packed_count} packed queries"
        )

    packed_index_by_qid = {qid: idx for idx, qid in enumerate(packed_query_ids)}
    missing = [qid for qid in query_ids if qid not in packed_index_by_qid]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} query ids missing from packed embeddings, e.g. {sample}")

    return query_ids, [packed_index_by_qid[qid] for qid in query_ids]


def dcg(scores: list[float]) -> float:
    total = 0.0
    for rank, score in enumerate(scores, start=1):
        total += (2**score - 1.0) / math.log2(rank + 1.0)
    return total


def _evaluate_metrics_fallback(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    ks: list[int],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    ndcg = {f"NDCG@{k}": 0.0 for k in ks}
    mean_ap = {f"MAP@{k}": 0.0 for k in ks}
    recall = {f"Recall@{k}": 0.0 for k in ks}
    precision = {f"P@{k}": 0.0 for k in ks}
    mrr = {f"MRR@{k}": 0.0 for k in ks}

    for qid, rels in qrels.items():
        ranked = sorted(results.get(qid, {}).items(), key=lambda item: item[1], reverse=True)
        relevant = {docid for docid, score in rels.items() if score > 0}
        ideal_gains = sorted((score for score in rels.values()), reverse=True)

        for k in ks:
            top_docs = ranked[:k]
            top_ids = [docid for docid, _ in top_docs]
            top_gains = [rels.get(docid, 0.0) for docid in top_ids]

            hits = 0
            ap = 0.0
            rr = 0.0
            for rank, docid in enumerate(top_ids, start=1):
                if docid in relevant:
                    hits += 1
                    ap += hits / rank
                    if rr == 0.0:
                        rr = 1.0 / rank

            ideal = dcg(ideal_gains[:k])
            actual = dcg(top_gains)

            ndcg[f"NDCG@{k}"] += 0.0 if ideal == 0 else actual / ideal
            mean_ap[f"MAP@{k}"] += 0.0 if not relevant else ap / len(relevant)
            recall[f"Recall@{k}"] += 0.0 if not relevant else hits / len(relevant)
            precision[f"P@{k}"] += hits / k
            mrr[f"MRR@{k}"] += rr

    num_queries = len(qrels)
    for metrics in (ndcg, mean_ap, recall, precision, mrr):
        for key in metrics:
            metrics[key] = round(metrics[key] / num_queries, 5)

    return ndcg, mean_ap, recall, precision, mrr


def evaluate_metrics(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
    ks: list[int],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    cleaned_results = {
        qid: {docid: score for docid, score in rels.items() if docid != qid}
        for qid, rels in results.items()
    }

    try:
        import pytrec_eval
    except ImportError:
        return _evaluate_metrics_fallback(qrels, cleaned_results, ks)

    map_string = "map_cut." + ",".join(str(k) for k in ks)
    ndcg_string = "ndcg_cut." + ",".join(str(k) for k in ks)
    recall_string = "recall." + ",".join(str(k) for k in ks)
    precision_string = "P." + ",".join(str(k) for k in ks)
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {map_string, ndcg_string, recall_string, precision_string},
    )
    scores = evaluator.evaluate(cleaned_results)

    ndcg = {f"NDCG@{k}": 0.0 for k in ks}
    mean_ap = {f"MAP@{k}": 0.0 for k in ks}
    recall = {f"Recall@{k}": 0.0 for k in ks}
    precision = {f"P@{k}": 0.0 for k in ks}
    mrr = {f"MRR@{k}": 0.0 for k in ks}

    for query_id in scores:
        ranked = sorted(
            cleaned_results.get(query_id, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )
        relevant = {
            docid for docid, score in qrels.get(query_id, {}).items() if score > 0
        }
        for k in ks:
            ndcg[f"NDCG@{k}"] += scores[query_id][f"ndcg_cut_{k}"]
            mean_ap[f"MAP@{k}"] += scores[query_id][f"map_cut_{k}"]
            recall[f"Recall@{k}"] += scores[query_id][f"recall_{k}"]
            precision[f"P@{k}"] += scores[query_id][f"P_{k}"]
            rr = 0.0
            for rank, (docid, _) in enumerate(ranked[:k], start=1):
                if docid in relevant:
                    rr = 1.0 / rank
                    break
            mrr[f"MRR@{k}"] += rr

    num_queries = len(scores)
    for metrics in (ndcg, mean_ap, recall, precision, mrr):
        for key in metrics:
            metrics[key] = round(metrics[key] / num_queries, 5)

    return ndcg, mean_ap, recall, precision, mrr


def main() -> None:
    args = parse_args()
    if not args.index_root:
        raise ValueError("--index-root is required when INDEX_ROOT is not set")

    split = args.split or infer_split(args.dataset)
    index_name = args.index or default_index_name(args.dataset, args.encoder, args.nbits)
    index_path = Path(args.index_root) / index_name

    query_points = np.load(
        resolve_embedding_dir(args) / "query_points.npy",
        mmap_mode="r",
    )
    query_offsets = np.load(
        resolve_embedding_dir(args) / "query_offsets.npy"
    ).astype(np.int64, copy=False)

    dataset_dir = resolve_dataset_dir(args)
    embedding_dir = resolve_embedding_dir(args)
    corpus, queries, qrels = load_beir_split(dataset_dir, split)
    corpus_ids = resolve_corpus_ids(corpus, embedding_dir)
    query_ids, query_indices = resolve_query_ids_and_indices(
        list(queries.keys()), query_offsets, embedding_dir
    )

    config = ColBERTConfig.load_from_index(str(index_path))
    config = configure_search(
        args.k,
        config,
        ncells=args.ncells,
        centroid_score_threshold=args.centroid_score_threshold,
        ndocs=args.ndocs,
    )
    ColBERT.try_load_torch_extensions(False)
    scorer = IndexScorer(str(index_path), use_gpu=torch.cuda.is_available())

    results = {}
    for qid, packed_idx in zip(query_ids, query_indices):
        start = int(query_offsets[packed_idx])
        end = int(query_offsets[packed_idx + 1])
        query = np.array(query_points[start:end], dtype=np.float32, copy=True)
        query = torch.from_numpy(query).unsqueeze(0)

        pids, scores = scorer.rank(config, query, tracker=NOPTracker())
        results[qid] = {
            corpus_ids[pid]: float(score) for pid, score in zip(pids[: args.k], scores[: args.k])
        }

    ndcg, _map, recall, precision, mrr = evaluate_metrics(qrels, results, args.metrics_k)

    print("NDCG", ndcg)
    print("MAP", _map)
    print("Recall", recall)
    print("P", precision)
    print("MRR", mrr)


if __name__ == "__main__":
    main()
