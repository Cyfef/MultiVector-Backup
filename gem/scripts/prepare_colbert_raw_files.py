#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import numpy as np


def link_or_keep(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            current = Path(os.readlink(target))
            if current == source:
                print(f"kept symlink {target} -> {source}")
                return
            target.unlink()
            target.symlink_to(source)
            print(f"updated symlink {target} -> {source} (was {current})")
            return
        print(f"kept existing {target}")
        return
    target.symlink_to(source)
    print(f"created symlink {target} -> {source}")


def write_counts_from_offsets(offsets_path: Path, target: Path) -> None:
    offsets = np.load(offsets_path, mmap_mode="r")
    if offsets.ndim != 1 or len(offsets) < 2:
        raise ValueError(f"Expected one-dimensional offsets with at least 2 values: {offsets_path}")
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    if np.any(counts < 0):
        raise ValueError(f"Offsets are not monotonic: {offsets_path}")
    if target.exists():
        existing = np.load(target, mmap_mode="r")
        if existing.shape == counts.shape and str(existing.dtype) == str(counts.dtype):
            print(f"kept existing {target} shape={existing.shape} dtype={existing.dtype} sum={int(existing.sum())}")
            return
        raise RuntimeError(f"Refusing to overwrite incompatible chunk-count file: {target}")
    np.save(target, counts)
    print(f"wrote {target} shape={counts.shape} dtype={counts.dtype} sum={int(counts.sum())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expose ColBERT NPY files under the filename convention expected by generate_msmarco_gem_data.py."
    )
    parser.add_argument("--raw-source", type=Path, required=True, help="Directory containing corpus/query points and offsets NPY files.")
    parser.add_argument("--raw-target", type=Path, required=True, help="Directory to hold symlinks and generated chunk-count NPY files.")
    parser.add_argument("--dataset-stem", type=str, required=True, help="Stem used in full_multi_* filenames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.raw_source
    target = args.raw_target
    stem = args.dataset_stem

    required = {
        "corpus_points": source / "corpus_points.npy",
        "corpus_offsets": source / "corpus_offsets.npy",
        "query_points": source / "query_points.npy",
        "query_offsets": source / "query_offsets.npy",
    }
    for name, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    target.mkdir(parents=True, exist_ok=True)
    link_or_keep(required["corpus_points"], target / f"full_multi_embeddings_{stem}.npy")
    link_or_keep(required["query_points"], target / f"full_multi_embeddings_{stem}_query.npy")
    write_counts_from_offsets(required["corpus_offsets"], target / f"full_multi_chunk_num_{stem}.npy")
    write_counts_from_offsets(required["query_offsets"], target / f"full_multi_chunk_num_{stem}_query.npy")

    base = np.load(required["corpus_points"], mmap_mode="r")
    query = np.load(required["query_points"], mmap_mode="r")
    print(f"source corpus_points shape={base.shape} dtype={base.dtype}")
    print(f"source query_points shape={query.shape} dtype={query.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
