#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.evaluate_precomputed_queries import evaluate_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one WARP TSV run against BEIR-style qrels."
    )
    parser.add_argument("--run-tsv", required=True, help="TSV with query-id, corpus-id, rank, score.")
    parser.add_argument("--dataset-dir", required=True, help="BEIR-style dataset directory.")
    parser.add_argument("--split", default="test", help="Qrels split name. Default: test.")
    parser.add_argument("--metrics-k", type=int, nargs="+", default=[10, 100])
    parser.add_argument("--output-csv", help="Optional single-row CSV destination.")
    return parser.parse_args()


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")

        for row in reader:
            qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row["score"])
    return qrels


def load_run(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")

        for row in reader:
            results.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = float(row["score"])
    return results


def flatten_metrics(
    ndcg: dict[str, float],
    mean_ap: dict[str, float],
    recall: dict[str, float],
    precision: dict[str, float],
    mrr: dict[str, float],
    ks: list[int],
) -> dict[str, float]:
    row: dict[str, float] = {}
    for k in ks:
        row[f"ndcg_{k}"] = ndcg[f"NDCG@{k}"]
        row[f"map_{k}"] = mean_ap[f"MAP@{k}"]
        row[f"recall_{k}"] = recall[f"Recall@{k}"]
        row[f"p_{k}"] = precision[f"P@{k}"]
        row[f"mrr_{k}"] = mrr[f"MRR@{k}"]
    return row


def main() -> None:
    args = parse_args()
    run_path = Path(args.run_tsv)
    dataset_dir = Path(args.dataset_dir)
    qrels_path = dataset_dir / "qrels" / f"{args.split}.tsv"

    qrels = load_qrels(qrels_path)
    results = load_run(run_path)
    ndcg, mean_ap, recall, precision, mrr = evaluate_metrics(qrels, results, args.metrics_k)
    row = flatten_metrics(ndcg, mean_ap, recall, precision, mrr, args.metrics_k)
    row["num_qrels_queries"] = len(qrels)
    row["num_run_queries"] = len(results)

    print(json.dumps(row, indent=2, sort_keys=True))

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
