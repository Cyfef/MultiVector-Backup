#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_DATASET_DIRS = {
    "clef": Path(
        "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef"
    ),
    "clerc": Path(
        "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc"
    ),
}


def dcg(scores: list[float]) -> float:
    total = 0.0
    for rank, score in enumerate(scores, start=1):
        total += (2**score - 1.0) / math.log2(rank + 1.0)
    return total


def evaluate_metrics(
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved TSV retrieval sweeps against BEIR-style qrels and "
            "write separate CSV files with NDCG/MAP/Recall/P/MRR metrics."
        )
    )
    parser.add_argument("csv_paths", nargs="+", help="Sweep manifest CSV files.")
    parser.add_argument(
        "--output-suffix",
        default=".eval.csv",
        help="Suffix appended to each input CSV stem.",
    )
    parser.add_argument(
        "--metrics-k",
        type=int,
        nargs="+",
        default=[10, 100],
        help="Cutoffs to evaluate.",
    )
    return parser.parse_args()


def infer_dataset_dir(dataset: str) -> Path:
    if dataset.startswith("clef"):
        return DEFAULT_DATASET_DIRS["clef"]
    if dataset.startswith("clerc"):
        return DEFAULT_DATASET_DIRS["clerc"]
    raise ValueError(f"No dataset-dir mapping for dataset={dataset}")


def load_corpus_index(dataset_dir: Path) -> dict[str, int]:
    corpus_index: dict[str, int] = {}
    with open(dataset_dir / "corpus.jsonl", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            row = json.loads(line)
            corpus_index[str(row["_id"])] = idx
    return corpus_index


def load_qrels_with_order(path: Path) -> tuple[list[str], dict[str, dict[str, int]]]:
    query_order: list[str] = []
    qrels: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            docid = str(row["corpus-id"])
            score = int(row["score"])
            if qid not in qrels:
                query_order.append(qid)
                qrels[qid] = {}
            qrels[qid][docid] = score
    return query_order, qrels


def load_results_tsv(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = str(row["query-id"])
            docid = str(row["corpus-id"])
            score = float(row["score"])
            results.setdefault(qid, {})[docid] = score
    return results


def metric_column_name(prefix: str, k: int) -> str:
    return f"{prefix.lower()}_{k}"


def local_tsv_path(csv_path: Path, row: dict[str, str]) -> Path:
    query_limit = row.get("query_limit") or row.get("num_queries") or ""
    filename = (
        f"split={row['split']}."
        f"baseline={row['baseline']}."
        f"ncells={row['ncells']}."
        f"thr={row['centroid_score_threshold']}."
        f"ndocs={row['ndocs']}."
        f"k={row['k']}."
        f"queries={query_limit}.tsv"
    )
    return csv_path.parent / row["dataset"] / filename


def map_qrels_to_local_ids(
    dataset: str,
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
    corpus_index: dict[str, int],
) -> dict[str, dict[str, int]]:
    mapped: dict[str, dict[str, int]] = {}
    for query_idx, qid in enumerate(query_ids):
        local_qid = f"{dataset}-query-{query_idx}"
        mapped[local_qid] = {
            f"{dataset}-doc-{corpus_index[docid]}": score
            for docid, score in qrels[qid].items()
        }
    return mapped


def evaluate_row(
    csv_path: Path,
    row: dict[str, str],
    ks: list[int],
    dataset_cache: dict[str, tuple[list[str], dict[str, dict[str, int]], dict[str, int]]],
) -> dict[str, str]:
    dataset = row["dataset"]
    if dataset not in dataset_cache:
        dataset_dir = infer_dataset_dir(dataset)
        query_order, qrels = load_qrels_with_order(dataset_dir / "qrels" / "test.tsv")
        corpus_index = load_corpus_index(dataset_dir)
        dataset_cache[dataset] = (query_order, qrels, corpus_index)

    query_order, qrels, corpus_index = dataset_cache[dataset]
    query_limit = int(row.get("query_limit") or row.get("num_queries") or len(query_order))
    selected_qids = query_order[:query_limit]
    mapped_qrels = map_qrels_to_local_ids(dataset, selected_qids, qrels, corpus_index)

    tsv_path = local_tsv_path(csv_path, row)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Missing TSV for row: {tsv_path}")

    results = load_results_tsv(tsv_path)
    ndcg, mean_ap, recall, precision, mrr = evaluate_metrics(mapped_qrels, results, ks)

    output_row = dict(row)
    output_row["source_tsv_path"] = row.get("tsv_path", "")
    output_row["tsv_path"] = str(tsv_path)
    output_row["evaluated_queries"] = str(len(mapped_qrels))
    output_row["retrieved_queries"] = str(len(results))
    for k in ks:
        output_row[metric_column_name("ndcg", k)] = str(ndcg[f"NDCG@{k}"])
        output_row[metric_column_name("map", k)] = str(mean_ap[f"MAP@{k}"])
        output_row[metric_column_name("recall", k)] = str(recall[f"Recall@{k}"])
        output_row[metric_column_name("p", k)] = str(precision[f"P@{k}"])
        output_row[metric_column_name("mrr", k)] = str(mrr[f"MRR@{k}"])
    return output_row


def output_path_for(csv_path: Path, suffix: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}{suffix}")


def write_output_csv(path: Path, rows: list[dict[str, str]], ks: list[int]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())
    desired_tail = ["source_tsv_path", "tsv_path", "evaluated_queries", "retrieved_queries"]
    for k in ks:
        desired_tail.extend(
            [
                metric_column_name("ndcg", k),
                metric_column_name("map", k),
                metric_column_name("recall", k),
                metric_column_name("p", k),
                metric_column_name("mrr", k),
            ]
        )

    seen = set(fieldnames)
    for column in desired_tail:
        if column not in seen:
            fieldnames.append(column)
            seen.add(column)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    dataset_cache: dict[str, tuple[list[str], dict[str, dict[str, int]], dict[str, int]]] = {}

    for csv_arg in args.csv_paths:
        csv_path = Path(csv_arg)
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        output_rows = []
        for idx, row in enumerate(rows, start=1):
            output_row = evaluate_row(csv_path, row, args.metrics_k, dataset_cache)
            output_rows.append(output_row)
            print(
                f"{csv_path.name}: row {idx}/{len(rows)} "
                f"dataset={row['dataset']} baseline={row['baseline']} "
                f"ncells={row['ncells']} recall@100={output_row.get('recall_100', 'NA')} "
                f"mrr@10={output_row.get('mrr_10', 'NA')}",
                flush=True,
            )

        output_path = output_path_for(csv_path, args.output_suffix)
        write_output_csv(output_path, output_rows, args.metrics_k)
        print(f"Saved evaluated sweep to {output_path}", flush=True)


if __name__ == "__main__":
    main()
