#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utility.sweep_ncells import run_sweep  # noqa: E402


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay saved WARP CSV settings, append MRR columns, and save TSV retrieval results."
    )
    parser.add_argument("csv_paths", nargs="+")
    parser.add_argument("--tsv-root", required=True)
    parser.add_argument("--log-file")
    return parser.parse_args()


def configure_torch_threads_from_env() -> None:
    value = os.environ.get("TORCH_NUM_THREADS")
    if not value:
        return

    import torch

    threads = int(value)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


LOG_FILE: Path | None = None


def log(message: str) -> None:
    print(message, flush=True)
    if LOG_FILE is not None:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def infer_dataset_dir(dataset: str) -> str:
    return DEFAULT_DATASET_DIRS.get(
        dataset, f"/data1/liuyaoyang/Papers/ACFDE/datasets/{dataset}"
    )


def infer_embedding_dir(dataset: str) -> str:
    return DEFAULT_EMBEDDING_DIRS.get(
        dataset, f"/data1/liuyaoyang/Papers/ACFDE/output/{dataset}/colbert"
    )


def sanitize_filename_part(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def tsv_output_path(tsv_root: Path, row: dict[str, str]) -> Path:
    dataset_dir = tsv_root / row["dataset"]
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


def replay_row(row: dict[str, str], tsv_root: Path) -> tuple[dict[str, str], dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="warp-mrr-row-") as tmpdir:
        tmp_csv = Path(tmpdir) / "row.csv"

        argv = argparse.Namespace(
            dataset=row["dataset"],
            dataset_dir=infer_dataset_dir(row["dataset"]),
            dataset_root="/unused",
            embedding_dir=infer_embedding_dir(row["dataset"]),
            embedding_root="/unused",
            encoder=row.get("encoder", "colbert"),
            index=row["index_name"],
            index_root=str(Path(row["index_path"]).parent),
            nbits=int(row["nbits"]),
            k=int(row["k"]),
            split=row["split"],
            baseline=row["baseline"],
            output_dir=str(tmp_csv.parent),
            output_csv=str(tmp_csv),
            append=False,
            ncells=[int(row["ncells"])],
            centroid_score_threshold=float(row["centroid_score_threshold"]),
            ndocs=int(row["ndocs"]),
            max_queries=int(row["query_limit"]) if row.get("query_limit") else None,
            metrics_k=[10, 100],
            tsv_output=str(tsv_output_path(tsv_root, row)),
        )

        run_sweep(argv)

        with open(tmp_csv, newline="", encoding="utf-8") as f:
            replayed_row = next(csv.DictReader(f))

    return row, replayed_row


def update_csv(path: Path, tsv_root: Path) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if "mrr_10" not in fieldnames:
        fieldnames = fieldnames + ["mrr_10", "mrr_100"]

    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        original = dict(row)
        _, replayed = replay_row(row, tsv_root)
        row["mrr_10"] = replayed["mrr_10"]
        row["mrr_100"] = replayed["mrr_100"]
        for key, value in original.items():
            if key not in {"mrr_10", "mrr_100"}:
                row[key] = value
        log(
            f"{path.name}: row {idx}/{total} "
            f"mrr_10={row['mrr_10']} mrr_100={row['mrr_100']}"
        )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global LOG_FILE
    configure_torch_threads_from_env()
    args = parse_args()
    if args.log_file:
        LOG_FILE = Path(args.log_file)
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text("", encoding="utf-8")
    log(f"Starting retrieval replay for {len(args.csv_paths)} CSV files")
    tsv_root = Path(args.tsv_root)
    tsv_root.mkdir(parents=True, exist_ok=True)
    for csv_path in args.csv_paths:
        log(f"Processing {csv_path}")
        update_csv(Path(csv_path), tsv_root)


if __name__ == "__main__":
    main()
