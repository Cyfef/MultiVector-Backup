#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PYTHON_BIN = Path("/data/ali/warp-baseline-cpu/bin/python")
LD_LIBRARY_PATH = "/data/ali/warp-baseline-cpu/lib"

DEFAULT_DATASET_DIRS = {
    "clef": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef",
    "clef-colbert": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef",
    "clef-modern-colbert": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clef",
    "clerc": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc",
    "clerc-colbert": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc",
    "clerc-modern-colbert": "/data1/liuyaoyang/Papers/vectorDB/POQD-ICML25/Decompose_retrieval/output/ali/id_and_gt/clerc",
    "fiqa": "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa",
    "fiqa-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/fiqa",
    "msmarco": "/data1/liuyaoyang/Papers/ACFDE/datasets/msmarco",
    "msmarco-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/msmarco",
    "nq": "/data1/liuyaoyang/Papers/ACFDE/datasets/nq",
    "nq-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/nq",
    "scidocs": "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs",
    "scidocs-modern-colbert": "/data1/liuyaoyang/Papers/ACFDE/datasets/scidocs",
}

DEFAULT_EMBEDDING_DIRS = {
    "clef-colbert": "/data/ali/clef-colbert",
    "clerc-colbert": "/data/ali/clerc-colbert",
    "clef-modern-colbert": "/data/ali/clef-modern-colbert",
    "clerc-modern-colbert": "/data/ali/clerc-modern-colbert",
    "fiqa-modern-colbert": "/data/ali/fiqa-modern-colbert-original",
    "msmarco-modern-colbert": "/data/ali/msmarco-modern-colbert-original",
    "nq-modern-colbert": "/data/ali/nq-modern-colbert-original",
    "scidocs-modern-colbert": "/data/ali/scidocs-modern-colbert-original",
}

FLOAT_KEYS = {
    "centroid_score_threshold",
    "ndcg_10",
    "map_10",
    "recall_10",
    "p_10",
    "ndcg_100",
    "map_100",
    "recall_100",
    "p_100",
}

INT_KEYS = {
    "nbits",
    "k",
    "ncells",
    "ndocs",
    "num_queries",
    "query_limit",
}

CHECK_KEYS = [
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
    "ndcg_10",
    "map_10",
    "recall_10",
    "p_10",
    "ndcg_100",
    "map_100",
    "recall_100",
    "p_100",
]

MRR_COLUMNS = ["mrr_10", "mrr_100"]

LOG_FILE: Path | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved WARP CSV rows via subprocesses, save TSVs, and append MRR columns."
    )
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--tsv-root", required=True)
    parser.add_argument("--log-file")
    parser.add_argument("--torch-ext-dir")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute MRR even when mrr_10 and mrr_100 are already present.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def normalize(value: str | None) -> str:
    if value is None:
        return ""
    return str(value)


def sanitize_filename_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def infer_dataset_dir(dataset: str) -> str:
    if dataset in DEFAULT_DATASET_DIRS:
        return DEFAULT_DATASET_DIRS[dataset]

    if dataset.endswith("-modern-colbert"):
        base = dataset[: -len("-modern-colbert")]
        return f"/data1/liuyaoyang/Papers/ACFDE/datasets/{base}"

    if dataset.endswith("-colbert"):
        base = dataset[: -len("-colbert")]
        return f"/data1/liuyaoyang/Papers/ACFDE/datasets/{base}"

    return f"/data1/liuyaoyang/Papers/ACFDE/datasets/{dataset}"


def infer_embedding_dir(dataset: str) -> str:
    if dataset in DEFAULT_EMBEDDING_DIRS:
        return DEFAULT_EMBEDDING_DIRS[dataset]

    if dataset.endswith("-modern-colbert"):
        return f"/data/ali/{dataset}"

    if dataset.endswith("-colbert"):
        return f"/data/ali/{dataset}"

    return f"/data1/liuyaoyang/Papers/ACFDE/output/{dataset}/colbert"


def row_query_limit(row: dict[str, str]) -> str:
    return normalize(row.get("query_limit")) or normalize(row.get("num_queries")) or "all"


def tsv_output_path(tsv_root: Path, row: dict[str, str]) -> Path:
    dataset_dir = tsv_root / row["dataset"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"split={sanitize_filename_part(row['split'])}."
        f"baseline={sanitize_filename_part(row['baseline'])}."
        f"ncells={row['ncells']}."
        f"thr={row['centroid_score_threshold']}."
        f"ndocs={row['ndocs']}."
        f"k={row['k']}."
        f"queries={row_query_limit(row)}.tsv"
    )
    return dataset_dir / filename


def needs_mrr(row: dict[str, str], force: bool) -> bool:
    if force:
        return True
    return not (normalize(row.get("mrr_10")) and normalize(row.get("mrr_100")))


def float_close(expected: str, actual: str) -> bool:
    return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=1e-5)


def compare_rows(original: dict[str, str], replayed: dict[str, str]) -> None:
    for key in CHECK_KEYS:
        if key not in original:
            continue

        expected = normalize(original.get(key))
        if expected == "":
            continue

        actual = normalize(replayed.get(key))
        if key in FLOAT_KEYS:
            if not float_close(expected, actual):
                raise ValueError(f"Mismatch for {key}: expected {expected}, got {actual}")
            continue

        if key in INT_KEYS:
            if int(expected) != int(actual):
                raise ValueError(f"Mismatch for {key}: expected {expected}, got {actual}")
            continue

        if expected != actual:
            raise ValueError(f"Mismatch for {key}: expected {expected}, got {actual}")


def build_command(row: dict[str, str], tmp_csv: Path, tsv_path: Path) -> list[str]:
    cmd = [
        str(PYTHON_BIN),
        "utility/sweep_ncells.py",
        "--dataset",
        row["dataset"],
        "--dataset-dir",
        infer_dataset_dir(row["dataset"]),
        "--embedding-dir",
        infer_embedding_dir(row["dataset"]),
        "--index-root",
        str(Path(row["index_path"]).parent),
        "--index",
        row["index_name"],
        "--split",
        row["split"],
        "--nbits",
        row["nbits"],
        "--k",
        row["k"],
        "--baseline",
        row["baseline"],
        "--centroid-score-threshold",
        row["centroid_score_threshold"],
        "--ndocs",
        row["ndocs"],
        "--ncells",
        row["ncells"],
        "--metrics-k",
        "10",
        "100",
        "--output-csv",
        str(tmp_csv),
        "--tsv-output",
        str(tsv_path),
    ]

    query_limit = normalize(row.get("query_limit")) or normalize(row.get("num_queries"))
    if query_limit:
        cmd.extend(["--max-queries", query_limit])

    return cmd


def replay_row(
    row: dict[str, str],
    tsv_root: Path,
    torch_ext_dir: Path,
) -> dict[str, str]:
    tsv_path = tsv_output_path(tsv_root, row)

    with tempfile.TemporaryDirectory(prefix="warp-mrr-row-") as tmpdir:
        tmp_csv = Path(tmpdir) / "row.csv"
        cmd = build_command(row, tmp_csv, tsv_path)

        env = os.environ.copy()
        env.update(
            {
                "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
                "TORCH_EXTENSIONS_DIR": str(torch_ext_dir),
                "TORCH_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
        )

        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            message = [
                "Replay command failed.",
                f"dataset={row['dataset']}",
                f"baseline={row['baseline']}",
                f"ncells={row['ncells']}",
                f"stdout:\n{result.stdout}",
                f"stderr:\n{result.stderr}",
            ]
            raise RuntimeError("\n".join(message))

        if not tmp_csv.exists():
            raise FileNotFoundError(f"Expected replay output CSV was not created: {tmp_csv}")

        with open(tmp_csv, newline="", encoding="utf-8") as f:
            replayed = next(csv.DictReader(f))

    compare_rows(row, replayed)
    return replayed


def write_csv_atomic(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def updated_fieldnames(fieldnames: list[str]) -> list[str]:
    output = list(fieldnames)
    for column in MRR_COLUMNS:
        if column not in output:
            output.append(column)
    return output


def update_csv(path: Path, tsv_root: Path, torch_ext_dir: Path, force: bool) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = updated_fieldnames(list(reader.fieldnames or []))

    total = len(rows)
    completed = 0
    for idx, row in enumerate(rows, start=1):
        if not needs_mrr(row, force):
            completed += 1
            log(f"{path.name}: row {idx}/{total} already has MRR, skipping")
            continue

        log(
            f"{path.name}: row {idx}/{total} start "
            f"dataset={row['dataset']} baseline={row['baseline']} "
            f"ncells={row['ncells']} thr={row['centroid_score_threshold']} "
            f"ndocs={row['ndocs']} queries={row_query_limit(row)}"
        )
        replayed = replay_row(row, tsv_root, torch_ext_dir)
        row["mrr_10"] = replayed["mrr_10"]
        row["mrr_100"] = replayed["mrr_100"]
        write_csv_atomic(path, fieldnames, rows)
        completed += 1
        log(
            f"{path.name}: row {idx}/{total} done "
            f"mrr_10={row['mrr_10']} mrr_100={row['mrr_100']} "
            f"completed={completed}/{total}"
        )


def main() -> None:
    global LOG_FILE

    args = parse_args()
    if args.log_file:
        LOG_FILE = Path(args.log_file)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text("", encoding="utf-8")

    if not PYTHON_BIN.exists():
        raise FileNotFoundError(f"Missing Python runtime: {PYTHON_BIN}")

    tsv_root = Path(args.tsv_root)
    tsv_root.mkdir(parents=True, exist_ok=True)

    if args.torch_ext_dir:
        torch_ext_dir = Path(args.torch_ext_dir)
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        torch_ext_dir = Path(tempfile.gettempdir()) / f"warp-mrr-ext-{stamp}"
    torch_ext_dir.mkdir(parents=True, exist_ok=True)

    log(f"Using Python runtime: {PYTHON_BIN}")
    log(f"Using torch extension cache: {torch_ext_dir}")
    log(f"Saving TSV files under: {tsv_root}")

    for csv_path in args.csv_paths:
        path = Path(csv_path)
        log(f"Processing CSV: {path}")
        update_csv(path, tsv_root, torch_ext_dir, args.force)

    log("MRR replay completed")


if __name__ == "__main__":
    main()
