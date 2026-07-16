#!/usr/bin/env python3
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def symlink_or_keep(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink() and Path(os.readlink(target)) == source:
            print(f"kept symlink {target} -> {source}")
            return
        print(f"kept existing {target}")
        return
    target.symlink_to(source)
    print(f"created symlink {target} -> {source}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a ColBERT raw NPY subset directory containing the first N corpus items."
    )
    parser.add_argument("--raw-source", type=Path, required=True, help="Source directory with corpus/query NPY files.")
    parser.add_argument("--raw-target", type=Path, required=True, help="Target directory for the subset raw files.")
    parser.add_argument("--num-docs", type=int, required=True, help="Number of corpus items to keep from the start.")
    parser.add_argument("--corpus-jsonl", type=Path, help="Optional BEIR-style corpus.jsonl used to map qrel doc ids to corpus indices.")
    parser.add_argument("--qrels", type=Path, help="Optional BEIR-style qrels TSV; all positive docs will be forced into the subset.")
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=1_000_000,
        help="Number of embedding rows copied per chunk when writing corpus_points.npy.",
    )
    return parser.parse_args()


def load_jsonl_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            ids.append(str(payload["_id"]))
    return ids


def load_positive_qrel_doc_ids(path: Path) -> list[str]:
    doc_ids: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if int(row["score"]) <= 0:
                continue
            doc_id = str(row["corpus-id"])
            if doc_id in seen:
                continue
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def build_subset_doc_indices(total_docs: int, num_docs: int, corpus_jsonl: Path | None, qrels: Path | None) -> np.ndarray:
    if bool(corpus_jsonl) != bool(qrels):
        raise ValueError("--corpus-jsonl and --qrels must be provided together.")

    if corpus_jsonl is None:
        return np.arange(num_docs, dtype=np.int64)

    corpus_ids = load_jsonl_ids(corpus_jsonl)
    if len(corpus_ids) != total_docs:
        raise RuntimeError(f"Corpus JSONL count mismatch: jsonl={len(corpus_ids)} offsets={total_docs}")
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    keep_doc_ids = load_positive_qrel_doc_ids(qrels)
    keep_indices = []
    seen_indices: set[int] = set()
    for doc_id in keep_doc_ids:
        idx = doc_to_idx.get(doc_id)
        if idx is None:
            continue
        if idx in seen_indices:
            continue
        seen_indices.add(idx)
        keep_indices.append(idx)
    if len(keep_indices) > num_docs:
        raise RuntimeError(
            f"Positive qrel docs exceed subset budget: required={len(keep_indices)} budget={num_docs}"
        )

    subset = list(keep_indices)
    for idx in range(total_docs):
        if idx in seen_indices:
            continue
        subset.append(idx)
        if len(subset) == num_docs:
            break
    subset_arr = np.asarray(subset, dtype=np.int64)
    subset_arr.sort()
    print(f"forced_qrel_docs={len(keep_indices)} total_subset_docs={len(subset_arr)}")
    return subset_arr


def main() -> int:
    args = parse_args()
    source = args.raw_source
    target = args.raw_target

    corpus_points_path = source / "corpus_points.npy"
    corpus_offsets_path = source / "corpus_offsets.npy"
    query_points_path = source / "query_points.npy"
    query_offsets_path = source / "query_offsets.npy"

    for path in [corpus_points_path, corpus_offsets_path, query_points_path, query_offsets_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    offsets = np.load(corpus_offsets_path, mmap_mode="r")
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError(f"Invalid corpus offsets: {corpus_offsets_path}")

    total_docs = len(offsets) - 1
    if args.num_docs <= 0 or args.num_docs > total_docs:
        raise ValueError(f"--num-docs must be in [1, {total_docs}], got {args.num_docs}")

    subset_doc_indices = build_subset_doc_indices(total_docs, args.num_docs, args.corpus_jsonl, args.qrels)
    subset_starts = np.asarray(offsets[subset_doc_indices], dtype=np.int64)
    subset_ends = np.asarray(offsets[subset_doc_indices + 1], dtype=np.int64)
    subset_counts = subset_ends - subset_starts
    subset_offsets = np.empty(len(subset_doc_indices) + 1, dtype=np.int64)
    subset_offsets[0] = 0
    np.cumsum(subset_counts, out=subset_offsets[1:])
    subset_rows = int(subset_offsets[-1])

    corpus_points = np.load(corpus_points_path, mmap_mode="r")
    if corpus_points.ndim != 2:
        raise ValueError(f"Expected 2D corpus points, got shape={corpus_points.shape}")

    target.mkdir(parents=True, exist_ok=True)

    subset_offsets_target = target / "corpus_offsets.npy"
    subset_points_target = target / "corpus_points.npy"
    subset_doc_ids_target = target / "subset_doc_indices.npy"
    if subset_doc_ids_target.exists():
        existing = np.load(subset_doc_ids_target, mmap_mode="r")
        if existing.shape != subset_doc_indices.shape or not np.array_equal(np.asarray(existing), subset_doc_indices):
            raise RuntimeError(f"Refusing to overwrite incompatible subset index file: {subset_doc_ids_target}")
        print(f"kept existing {subset_doc_ids_target} shape={existing.shape}")
    else:
        np.save(subset_doc_ids_target, subset_doc_indices)
        print(f"wrote {subset_doc_ids_target} shape={subset_doc_indices.shape}")
    if subset_offsets_target.exists():
        existing = np.load(subset_offsets_target, mmap_mode="r")
        if existing.shape != subset_offsets.shape or int(existing[-1]) != subset_rows:
            raise RuntimeError(f"Refusing to overwrite incompatible offsets file: {subset_offsets_target}")
        print(f"kept existing {subset_offsets_target} shape={existing.shape} last={int(existing[-1])}")
    else:
        np.save(subset_offsets_target, subset_offsets)
        print(f"wrote {subset_offsets_target} shape={subset_offsets.shape} last={subset_rows}")

    expected_shape = (subset_rows, corpus_points.shape[1])
    if subset_points_target.exists():
        existing = np.load(subset_points_target, mmap_mode="r")
        if existing.shape != expected_shape or existing.dtype != corpus_points.dtype:
            raise RuntimeError(f"Refusing to overwrite incompatible points file: {subset_points_target}")
        print(f"kept existing {subset_points_target} shape={existing.shape} dtype={existing.dtype}")
    else:
        out = np.lib.format.open_memmap(
            subset_points_target,
            mode="w+",
            dtype=corpus_points.dtype,
            shape=expected_shape,
        )
        out_row = 0
        for local_doc_idx, doc_idx in enumerate(subset_doc_indices):
            row_start = int(offsets[doc_idx])
            row_end = int(offsets[doc_idx + 1])
            doc_rows = row_end - row_start
            if doc_rows:
                out[out_row:out_row + doc_rows] = corpus_points[row_start:row_end]
            out_row += doc_rows
            if local_doc_idx == 0 or local_doc_idx + 1 == len(subset_doc_indices) or local_doc_idx % 10000 == 0:
                print(f"copied docs {local_doc_idx + 1}:{len(subset_doc_indices)} rows={out_row}/{subset_rows}")
        del out
        print(f"wrote {subset_points_target} shape={expected_shape} dtype={corpus_points.dtype}")

    symlink_or_keep(query_points_path, target / "query_points.npy")
    symlink_or_keep(query_offsets_path, target / "query_offsets.npy")

    print(f"subset docs={args.num_docs} subset vectors={subset_rows} dim={corpus_points.shape[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
