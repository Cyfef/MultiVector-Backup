#!/usr/bin/env python3
import csv
import json
import os
import shutil
from pathlib import Path

from svr_benchmark_batch import parse_qps


ROOT = Path("/home/ali/SVR-baselines")
RESULT_ROOT = ROOT / "results"
LEGACY_RUNS_ROOT = ROOT / "runs" / "clerc-large-single"
EXPORT_ROOT = RESULT_ROOT / "by_dataset_method"
METHODS = ("hcnng", "hnswlib", "elpis")
LEGACY_QUERY_COUNT = 1000


def ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def classify_subdir(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".log":
        return "logs"
    if suffix == ".json":
        return "metrics"
    if suffix in {".tsv", ".csv"}:
        return "results"
    return "misc"


def write_manifest(dst_dir: Path, dataset: str, method: str, files: list[Path], summary_rows: list[dict]) -> None:
    manifest = {
        "dataset": dataset,
        "method": method,
        "file_count": len(files),
        "files": [str(path) for path in sorted(files)],
        "summary_rows": len(summary_rows),
    }
    (dst_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_summary(dst_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (dst_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_named_summary(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_global_rows() -> list[dict]:
    path = RESULT_ROOT / "all_datasets_summary.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_dataset_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_legacy_rows(method: str) -> list[dict]:
    method_dir = LEGACY_RUNS_ROOT / method
    logs_dir = LEGACY_RUNS_ROOT / "logs"
    if not method_dir.exists():
        return []

    if method == "hcnng":
        summary_candidate = method_dir / "hcnng_qps_target200_summary.csv"
        if summary_candidate.exists():
            with summary_candidate.open("r", encoding="utf-8", newline="") as f:
                return list(csv.DictReader(f))
        return []

    rows = []
    for metrics_json in sorted(method_dir.glob("beir_metrics*.json")):
        metrics = json.loads(metrics_json.read_text(encoding="utf-8"))
        if method == "hnswlib":
            setting = metrics_json.stem.replace("beir_metrics_", "")
            ann_tsv = method_dir / "ans_k100_scored.tsv"
            log_path = logs_dir / "hnswlib_k100_scored.log"
        elif method == "elpis":
            setting = metrics_json.stem.replace("beir_metrics_", "")
            ann_name = setting.replace("k100_", "ans_k100_") + ".tsv"
            ann_tsv = method_dir / ann_name
            log_path = logs_dir / f"elpis_{setting}.log"
        else:
            continue

        search_sec, qps = parse_qps(log_path, method)
        if method == "elpis" and search_sec:
            qps = LEGACY_QUERY_COUNT / search_sec

        rows.append({
            "setting": setting,
            "max_calc": "",
            "search_time_sec": "" if search_sec is None else f"{search_sec:.6f}",
            "qps": "" if qps is None else f"{qps:.6f}",
            "queries_qrels": metrics.get("queries_qrels", ""),
            "queries_results": metrics.get("queries_results", ""),
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
            "metrics_json": str(metrics_json),
            "log": str(log_path),
        })
    return rows


def collect_standard_files(dataset: str, method: str) -> list[Path]:
    files: list[Path] = []
    method_dir = RESULT_ROOT / dataset / method
    eval_dir = RESULT_ROOT / dataset / "eval"
    logs_dir = RESULT_ROOT / dataset / "logs"

    if method_dir.exists():
        files.extend(sorted(path for path in method_dir.iterdir() if path.is_file()))
    if eval_dir.exists():
        files.extend(sorted(eval_dir.glob(f"{method}_*.json")))
    if logs_dir.exists():
        files.extend(sorted(logs_dir.glob(f"{method}*.log")))
        if method == "elpis":
            files.extend(sorted(logs_dir.glob("elpis_build_*.log")))

    seen = set()
    unique = []
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def collect_legacy_files(method: str) -> list[Path]:
    files: list[Path] = []
    method_dir = LEGACY_RUNS_ROOT / method
    logs_dir = LEGACY_RUNS_ROOT / "logs"

    if method_dir.exists():
        files.extend(sorted(path for path in method_dir.iterdir() if path.is_file()))
    if logs_dir.exists():
        files.extend(sorted(logs_dir.glob(f"{method}*.log")))

    seen = set()
    unique = []
    for path in files:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def export_one(dataset: str, method: str, files: list[Path], summary_rows: list[dict]) -> None:
    dst_dir = EXPORT_ROOT / dataset / method
    dst_dir.mkdir(parents=True, exist_ok=True)

    for src in files:
        subdir = classify_subdir(src)
        link_or_copy(src, dst_dir / subdir / src.name)

    write_summary(dst_dir, summary_rows)
    write_manifest(dst_dir, dataset, method, files, summary_rows)


def main() -> None:
    ensure_empty_dir(EXPORT_ROOT)

    global_rows = load_global_rows()
    standard_datasets = sorted(
        path.name
        for path in RESULT_ROOT.iterdir()
        if path.is_dir() and path.name not in {"by_dataset_method", "elpis_parallel_20260529_1006"}
    )

    for dataset in standard_datasets:
        corrected_rows = load_dataset_rows(RESULT_ROOT / dataset / "summary_corrected_qrels.csv")
        for method in METHODS:
            files = collect_standard_files(dataset, method)
            if not files:
                continue
            summary_rows = [row for row in global_rows if row.get("dataset") == dataset and row.get("method") == method]
            export_one(dataset, method, files, summary_rows)
            method_corrected_rows = [row for row in corrected_rows if row.get("method") == method]
            if method_corrected_rows:
                write_named_summary(
                    EXPORT_ROOT / dataset / method / "summary_corrected_qrels.csv",
                    method_corrected_rows,
                )

    legacy_dataset = "clerc-large-single"
    if LEGACY_RUNS_ROOT.exists():
        for method in METHODS:
            files = collect_legacy_files(method)
            if not files:
                continue
            summary_rows = build_legacy_rows(method)
            export_one(legacy_dataset, method, files, summary_rows)


if __name__ == "__main__":
    main()
