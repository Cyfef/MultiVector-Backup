#!/usr/bin/env python3
import csv
import json
import os
import struct
import subprocess
from pathlib import Path

from svr_benchmark_batch import CSV_FIELDS, DATA_ROOT, RESULT_ROOT, parse_qps, write_csv


ROOT = Path("/home/ali/SVR-baselines")
DATASET = "clerc-small-single"
METHODS = ("hcnng", "hnswlib", "elpis")
RAW_DIR = Path("/data/ali") / DATASET
INPUT_DIR = DATA_ROOT / DATASET / "inputs"
DATASET_RESULT_ROOT = RESULT_ROOT / DATASET
EXPORT_ROOT = RESULT_ROOT / "by_dataset_method" / DATASET
QUERY_LIMIT = 1000
CORRECTED_QRELS = INPUT_DIR / "qrels_corrected_identity_first1000.tsv"
VALIDATION_JSON = DATASET_RESULT_ROOT / "eval" / "clerc_small_corrected_qrels_validation.json"
SUMMARY_CSV = DATASET_RESULT_ROOT / "summary_corrected_qrels.csv"
COMPARISON_CSV = DATASET_RESULT_ROOT / "summary_qrels_variant_comparison.csv"


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def read_one_fvec(f) -> tuple[float, ...] | None:
    raw = f.read(4)
    if not raw:
        return None
    dim = struct.unpack("<i", raw)[0]
    return struct.unpack("<" + "f" * dim, f.read(4 * dim))


def verify_identity_queries(limit: int) -> dict:
    base_path = RAW_DIR / f"{DATASET}_base.fvecs"
    query_path = RAW_DIR / f"{DATASET}_query.fvecs"
    identical = 0
    first_mismatch = None

    with base_path.open("rb") as base_f, query_path.open("rb") as query_f:
        for idx in range(limit):
            base_vec = read_one_fvec(base_f)
            query_vec = read_one_fvec(query_f)
            if base_vec is None or query_vec is None:
                raise ValueError(f"short read while verifying {DATASET} at row {idx}")
            matches = all(abs(a - b) < 1e-8 for a, b in zip(base_vec, query_vec))
            if matches:
                identical += 1
            elif first_mismatch is None:
                first_mismatch = idx

    summary = {
        "dataset": DATASET,
        "query_limit": limit,
        "identical_queries": identical,
        "first_mismatch_row": first_mismatch,
        "verified_identity_mapping": identical == limit,
        "base_fvecs": str(base_path),
        "query_fvecs": str(query_path),
    }
    VALIDATION_JSON.parent.mkdir(parents=True, exist_ok=True)
    VALIDATION_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if identical != limit:
        raise ValueError(
            f"{DATASET} first {limit} queries are not all identical to base rows 0..{limit - 1}; "
            f"first mismatch at {first_mismatch}"
        )
    return summary


def write_corrected_qrels(limit: int) -> Path:
    query_ids = read_ids(INPUT_DIR / "query_ids.txt")
    doc_ids = read_ids(INPUT_DIR / "doc_ids.txt")
    if len(query_ids) < limit:
        raise ValueError(f"need at least {limit} query ids, found {len(query_ids)}")
    if len(doc_ids) < limit:
        raise ValueError(f"need at least {limit} doc ids, found {len(doc_ids)}")

    CORRECTED_QRELS.parent.mkdir(parents=True, exist_ok=True)
    with CORRECTED_QRELS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        for row in range(limit):
            writer.writerow([query_ids[row], doc_ids[row], 1])
    return CORRECTED_QRELS


def parse_setting(method: str, beir_tsv: Path) -> str:
    name = beir_tsv.name
    prefix = "ans_k100_"
    suffix = "_beir.tsv"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ValueError(f"unexpected results file for {method}: {beir_tsv}")
    return name[len(prefix):-len(suffix)]


def run_eval(results_path: Path, metrics_path: Path) -> None:
    cmd = [
        "python3",
        str(ROOT / "runs" / "eval_beir_metrics.py"),
        "--groundtruth",
        str(CORRECTED_QRELS),
        "--results",
        str(results_path),
        "--k-values",
        "10",
        "100",
        "--output-json",
        str(metrics_path),
    ]
    subprocess.run(cmd, check=True)


def build_summary_row(method: str, setting: str, metrics_path: Path) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    ann_tsv = DATASET_RESULT_ROOT / method / f"ans_k100_{setting}.tsv"
    beir_tsv = DATASET_RESULT_ROOT / method / f"ans_k100_{setting}_beir.tsv"
    log_path = DATASET_RESULT_ROOT / "logs" / f"{method}_{setting}.log"
    search_sec, qps = parse_qps(log_path, method)
    if method == "elpis" and search_sec:
        qps = QUERY_LIMIT / search_sec

    return {
        "dataset": DATASET,
        "method": method,
        "setting": setting,
        "params": json.dumps({"qrels_variant": "corrected_identity_first1000"}, sort_keys=True),
        "queries_qrels": metrics["queries_qrels"],
        "queries_results": metrics["queries_results"],
        "search_time_sec": "" if search_sec is None else f"{search_sec:.6f}",
        "qps": "" if qps is None else f"{qps:.6f}",
        "NDCG@10": metrics["ndcg"].get("NDCG@10", ""),
        "NDCG@100": metrics["ndcg"].get("NDCG@100", ""),
        "MAP@10": metrics["map"].get("MAP@10", ""),
        "MAP@100": metrics["map"].get("MAP@100", ""),
        "Recall@10": metrics["recall"].get("Recall@10", ""),
        "Recall@100": metrics["recall"].get("Recall@100", ""),
        "P@10": metrics["precision"].get("P@10", ""),
        "P@100": metrics["precision"].get("P@100", ""),
        "MRR@10": metrics["mrr"].get("MRR@10", ""),
        "MRR@100": metrics["mrr"].get("MRR@100", ""),
        "R_cap@10": metrics["recall_cap"].get("R_cap@10", ""),
        "R_cap@100": metrics["recall_cap"].get("R_cap@100", ""),
        "Hole@10": metrics["hole"].get("Hole@10", ""),
        "Hole@100": metrics["hole"].get("Hole@100", ""),
        "Accuracy@10": metrics["accuracy"].get("Accuracy@10", ""),
        "Accuracy@100": metrics["accuracy"].get("Accuracy@100", ""),
        "ann_tsv": str(ann_tsv),
        "beir_tsv": str(beir_tsv),
        "metrics_json": str(metrics_path),
        "log": str(log_path),
    }


def build_variant_comparison(corrected_rows: list[dict]) -> None:
    original_rows_path = DATASET_RESULT_ROOT / "summary.csv"
    if not original_rows_path.exists():
        return

    with original_rows_path.open("r", encoding="utf-8", newline="") as f:
        original_rows = list(csv.DictReader(f))

    original_by_key = {(row["method"], row["setting"]): row for row in original_rows}
    fieldnames = [
        "dataset",
        "method",
        "setting",
        "qrels_variant",
        "Recall@10",
        "Recall@100",
        "MRR@10",
        "MRR@100",
        "NDCG@10",
        "NDCG@100",
        "MAP@10",
        "MAP@100",
        "P@10",
        "P@100",
        "R_cap@10",
        "R_cap@100",
        "Accuracy@10",
        "Accuracy@100",
        "Hole@10",
        "Hole@100",
        "qps",
        "metrics_json",
    ]

    rows = []
    for corrected in corrected_rows:
        key = (corrected["method"], corrected["setting"])
        original = original_by_key.get(key)
        if original is not None:
            rows.append({
                "dataset": DATASET,
                "method": corrected["method"],
                "setting": corrected["setting"],
                "qrels_variant": "packaged_filtered_qrels",
                "Recall@10": original["Recall@10"],
                "Recall@100": original["Recall@100"],
                "MRR@10": original["MRR@10"],
                "MRR@100": original["MRR@100"],
                "NDCG@10": original["NDCG@10"],
                "NDCG@100": original["NDCG@100"],
                "MAP@10": original["MAP@10"],
                "MAP@100": original["MAP@100"],
                "P@10": original["P@10"],
                "P@100": original["P@100"],
                "R_cap@10": original["R_cap@10"],
                "R_cap@100": original["R_cap@100"],
                "Accuracy@10": original["Accuracy@10"],
                "Accuracy@100": original["Accuracy@100"],
                "Hole@10": original["Hole@10"],
                "Hole@100": original["Hole@100"],
                "qps": original["qps"],
                "metrics_json": original["metrics_json"],
            })

        rows.append({
            "dataset": DATASET,
            "method": corrected["method"],
            "setting": corrected["setting"],
            "qrels_variant": "corrected_identity_first1000",
            "Recall@10": corrected["Recall@10"],
            "Recall@100": corrected["Recall@100"],
            "MRR@10": corrected["MRR@10"],
            "MRR@100": corrected["MRR@100"],
            "NDCG@10": corrected["NDCG@10"],
            "NDCG@100": corrected["NDCG@100"],
            "MAP@10": corrected["MAP@10"],
            "MAP@100": corrected["MAP@100"],
            "P@10": corrected["P@10"],
            "P@100": corrected["P@100"],
            "R_cap@10": corrected["R_cap@10"],
            "R_cap@100": corrected["R_cap@100"],
            "Accuracy@10": corrected["Accuracy@10"],
            "Accuracy@100": corrected["Accuracy@100"],
            "Hole@10": corrected["Hole@10"],
            "Hole@100": corrected["Hole@100"],
            "qps": corrected["qps"],
            "metrics_json": corrected["metrics_json"],
        })

    COMPARISON_CSV.parent.mkdir(parents=True, exist_ok=True)
    with COMPARISON_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_rows(corrected_rows: list[dict]) -> None:
    by_method: dict[str, list[dict]] = {method: [] for method in METHODS}
    for row in corrected_rows:
        by_method[row["method"]].append(row)

    for method, rows in by_method.items():
        if not rows:
            continue
        rows.sort(key=lambda row: row["setting"])
        method_dir = EXPORT_ROOT / method
        method_dir.mkdir(parents=True, exist_ok=True)
        write_csv(method_dir / "summary_corrected_qrels.csv", rows)
        for row in rows:
            metrics_src = Path(row["metrics_json"])
            metrics_dst = method_dir / "metrics" / metrics_src.name
            metrics_dst.parent.mkdir(parents=True, exist_ok=True)
            if metrics_dst.exists():
                metrics_dst.unlink()
            try:
                os.link(metrics_src, metrics_dst)
            except OSError:
                metrics_dst.write_bytes(metrics_src.read_bytes())


def main() -> None:
    verify_identity_queries(QUERY_LIMIT)
    write_corrected_qrels(QUERY_LIMIT)

    corrected_rows: list[dict] = []
    for method in METHODS:
        method_dir = DATASET_RESULT_ROOT / method
        for beir_tsv in sorted(method_dir.glob("ans_k100_*_beir.tsv")):
            setting = parse_setting(method, beir_tsv)
            metrics_path = DATASET_RESULT_ROOT / "eval" / f"{method}_{setting}_corrected_qrels_beir_metrics.json"
            run_eval(beir_tsv, metrics_path)
            corrected_rows.append(build_summary_row(method, setting, metrics_path))

    corrected_rows.sort(key=lambda row: (row["method"], row["setting"]))
    write_csv(SUMMARY_CSV, corrected_rows)
    build_variant_comparison(corrected_rows)
    export_rows(corrected_rows)


if __name__ == "__main__":
    main()
