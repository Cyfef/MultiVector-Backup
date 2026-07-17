#!/usr/bin/env python3
"""Run a monotonic EMVB search-budget sweep and update metrics_summary.csv."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


BASE_SEARCH_CONFIGS = {
    10: {
        "nprobe": 4,
        "thresh": 0.4,
        "thresh_query": 0.5,
        "out_second_stage": 512,
        "n_doc_to_score": 4000,
    },
    100: {
        "nprobe": 4,
        "thresh": 0.4,
        "thresh_query": 0.5,
        "out_second_stage": 1024,
        "n_doc_to_score": 4000,
    },
}

SUMMARY_FIELDS = [
    "dataset",
    "index_label",
    "profile",
    "search_ratio",
    "k",
    "dataset_dir",
    "split",
    "run",
    "index_doc_count",
    "index_query_count",
    "query_count",
    "corpus_count",
    "results_query_count",
    "search_threads",
    "avg_query_time_ns",
    "avg_query_time_s",
    "qps",
    "nprobe",
    "thresh",
    "thresh_query",
    "out_second_stage",
    "n_doc_to_score",
    "ndcg",
    "map",
    "mrr",
    "recall",
    "precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--ratios", nargs="+", type=float, required=True)
    parser.add_argument("--k-values", nargs="+", type=int, default=[10, 100])
    parser.add_argument("--wait-for-index", action="store_true")
    parser.add_argument("--restrict-to-run-queries", action="store_true")
    parser.add_argument("--search-threads", type=int, default=1)
    parser.add_argument("--search-query-ids-file", type=Path)
    parser.add_argument("--query-ids-file", type=Path)
    parser.add_argument("--corpus-ids-file", type=Path)
    parser.add_argument(
        "--query-id-mode",
        choices=("auto", "positional", "direct"),
        default="auto",
    )
    return parser.parse_args()


def ratio_tag(ratio: float) -> str:
    scaled = int(round(ratio * 100))
    return f"r{scaled:03d}"


def ratio_label(ratio: float) -> str:
    text = f"{ratio:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def wait_for_index(index_dir: Path) -> dict[str, object]:
    metadata_path = index_dir / "metadata.json"
    while not metadata_path.exists():
        time.sleep(30)
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def scaled_config(k: int, ratio: float) -> dict[str, float | int]:
    base = BASE_SEARCH_CONFIGS[k]
    return {
        "nprobe": max(1, math.ceil(base["nprobe"] * ratio)),
        "thresh": base["thresh"],
        "thresh_query": base["thresh_query"],
        "out_second_stage": max(k, int(round(base["out_second_stage"] * ratio))),
        "n_doc_to_score": max(k, int(round(base["n_doc_to_score"] * ratio))),
    }


def run_search(
    *,
    repo_root: Path,
    index_dir: Path,
    results_dir: Path,
    ratio: float,
    k: int,
    config: dict[str, float | int],
    search_query_ids_file: Path | None,
) -> tuple[Path, Path]:
    suffix = ratio_tag(ratio)
    run_path = results_dir / f"results_k{k}_{suffix}.tsv"
    run_log = results_dir / f"run_k{k}_{suffix}.log"
    if run_path.exists():
        return run_path, run_log

    results_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    cmd = [
        str(repo_root / "build" / "perf_emvb"),
        "-k",
        str(k),
        "-nprobe",
        str(config["nprobe"]),
        "-thresh",
        str(config["thresh"]),
        "-out-second-stage",
        str(config["out_second_stage"]),
        "-thresh-query",
        str(config["thresh_query"]),
        "-n-doc-to-score",
        str(config["n_doc_to_score"]),
        "-queries-id-file",
        str(search_query_ids_file if search_query_ids_file is not None else index_dir / "queries_id.txt"),
        "-alldoclens-path",
        str(index_dir / "alldoclens.npy"),
        "-index-dir-path",
        str(index_dir),
        "-out-file",
        str(run_path),
    ]
    with run_log.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
    return run_path, run_log


def run_eval(
    *,
    repo_root: Path,
    dataset_dir: Path,
    split: str,
    run_path: Path,
    run_log: Path,
    ratio: float,
    k: int,
    results_dir: Path,
    restrict_to_run_queries: bool,
    query_ids_file: Path | None,
    corpus_ids_file: Path | None,
    query_id_mode: str,
) -> Path:
    suffix = ratio_tag(ratio)
    output_json = results_dir / f"metrics_k{k}_{suffix}.json"
    output_csv = results_dir / f"metrics_k{k}_{suffix}.csv"
    if output_json.exists():
        return output_json

    cmd = [
        os.environ.get("PYTHON_BIN", sys.executable),
        str(repo_root / "evaluate_beir_emvb.py"),
        "--dataset-dir",
        str(dataset_dir),
        "--split",
        split,
        "--run",
        str(run_path),
        "--k-values",
        str(k),
        "--run-log",
        str(run_log),
        "--search-threads",
        "1",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    ]
    if query_ids_file is not None:
        cmd.extend(["--query-ids-file", str(query_ids_file), "--query-id-mode", query_id_mode])
    if corpus_ids_file is not None:
        cmd.extend(["--corpus-ids-file", str(corpus_ids_file)])
    if restrict_to_run_queries:
        cmd.append("--restrict-to-run-queries")
    subprocess.run(cmd, check=True)
    return output_json


def base_run_name(k: int) -> str:
    return f"results_k{k}.tsv"


def load_existing_rows(summary_path: Path) -> list[dict[str, str]]:
    if not summary_path.exists():
        return []
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_existing_rows(
    rows: list[dict[str, str]],
    *,
    dataset: str,
    index_label: str,
    metadata: dict[str, object],
) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = {field: row.get(field, "") for field in SUMMARY_FIELDS}
        if not item["dataset"]:
            item["dataset"] = dataset
        if not item["index_label"]:
            item["index_label"] = index_label
        if not item["index_doc_count"]:
            item["index_doc_count"] = str(metadata["n_docs"])
        if not item["index_query_count"]:
            item["index_query_count"] = str(metadata["n_queries"])

        run_name = Path(item["run"]).name if item["run"] else ""
        if not item["search_ratio"] and run_name in {base_run_name(10), base_run_name(100)}:
            k = int(item["k"])
            base = BASE_SEARCH_CONFIGS[k]
            item["profile"] = item["profile"] or "base"
            item["search_ratio"] = "1.0"
            item["nprobe"] = item["nprobe"] or str(base["nprobe"])
            item["thresh"] = item["thresh"] or str(base["thresh"])
            item["thresh_query"] = item["thresh_query"] or str(base["thresh_query"])
            item["out_second_stage"] = item["out_second_stage"] or str(base["out_second_stage"])
            item["n_doc_to_score"] = item["n_doc_to_score"] or str(base["n_doc_to_score"])
        normalized.append(item)
    return normalized


def build_summary_row(
    *,
    dataset: str,
    index_label: str,
    metadata: dict[str, object],
    ratio: float,
    k: int,
    config: dict[str, float | int],
    eval_json_path: Path,
) -> dict[str, str]:
    summary = json.loads(eval_json_path.read_text(encoding="utf-8"))
    ndcg_key = f"NDCG@{k}"
    map_key = f"MAP@{k}"
    mrr_key = f"MRR@{k}"
    recall_key = f"Recall@{k}"
    precision_key = f"P@{k}"
    return {
        "dataset": dataset,
        "index_label": index_label,
        "profile": ratio_tag(ratio),
        "search_ratio": ratio_label(ratio),
        "k": str(k),
        "dataset_dir": summary["dataset_dir"],
        "split": summary["split"],
        "run": summary["run"],
        "index_doc_count": str(metadata["n_docs"]),
        "index_query_count": str(metadata["n_queries"]),
        "query_count": str(summary["query_count"]),
        "corpus_count": str(summary["corpus_count"]),
        "results_query_count": str(summary["results_query_count"]),
        "search_threads": str(summary["search_threads"]),
        "avg_query_time_ns": str(summary.get("avg_query_time_ns", "")),
        "avg_query_time_s": str(summary.get("avg_query_time_s", "")),
        "qps": str(summary.get("qps", "")),
        "nprobe": str(config["nprobe"]),
        "thresh": str(config["thresh"]),
        "thresh_query": str(config["thresh_query"]),
        "out_second_stage": str(config["out_second_stage"]),
        "n_doc_to_score": str(config["n_doc_to_score"]),
        "ndcg": str(summary["ndcg"][ndcg_key]),
        "map": str(summary["map"][map_key]),
        "mrr": str(summary["mrr"][mrr_key]),
        "recall": str(summary["recall"][recall_key]),
        "precision": str(summary["precision"][precision_key]),
    }


def write_summary(summary_path: Path, rows: list[dict[str, str]]) -> None:
    ordered = sorted(rows, key=lambda row: (row["k"], float(row["search_ratio"] or 0.0), row["run"]))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    metadata = wait_for_index(args.index_dir) if args.wait_for_index else json.loads(
        (args.index_dir / "metadata.json").read_text(encoding="utf-8")
    )
    index_label = args.index_dir.name
    summary_path = args.results_dir / "metrics_summary.csv"

    rows = normalize_existing_rows(
        load_existing_rows(summary_path),
        dataset=args.dataset,
        index_label=index_label,
        metadata=metadata,
    )
    row_by_run = {row["run"]: row for row in rows if row.get("run")}

    for ratio in args.ratios:
        for k in args.k_values:
            config = scaled_config(k, ratio)
            run_path, run_log = run_search(
                repo_root=repo_root,
                index_dir=args.index_dir,
                results_dir=args.results_dir,
                ratio=ratio,
                k=k,
                config=config,
                search_query_ids_file=args.search_query_ids_file,
            )
            eval_json_path = run_eval(
                repo_root=repo_root,
                dataset_dir=args.dataset_dir,
                split=args.split,
                run_path=run_path,
                run_log=run_log,
                ratio=ratio,
                k=k,
                results_dir=args.results_dir,
                restrict_to_run_queries=args.restrict_to_run_queries,
                query_ids_file=args.query_ids_file,
                corpus_ids_file=args.corpus_ids_file,
                query_id_mode=args.query_id_mode,
            )
            row = build_summary_row(
                dataset=args.dataset,
                index_label=index_label,
                metadata=metadata,
                ratio=ratio,
                k=k,
                config=config,
                eval_json_path=eval_json_path,
            )
            row_by_run[row["run"]] = row

    write_summary(summary_path, list(row_by_run.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
