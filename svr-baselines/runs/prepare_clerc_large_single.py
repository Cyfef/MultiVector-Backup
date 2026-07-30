#!/usr/bin/env python3
import argparse
import os
import shutil
import struct


def count_vecs(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        raw = f.read(4)
        if len(raw) != 4:
            raise ValueError(f"empty file: {path}")
        dim = struct.unpack("<i", raw)[0]
        f.seek(0, os.SEEK_END)
        size = f.tell()
    rec_size = 4 + 4 * dim
    if size % rec_size != 0:
        raise ValueError(f"invalid vec file size for {path}")
    return size // rec_size, dim


def copy_first_records(src: str, dst: str, n: int) -> tuple[int, int]:
    total, dim = count_vecs(src)
    if n > total:
        raise ValueError(f"requested {n} records from {src}, only {total} available")
    rec_size = 4 + 4 * dim
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for _ in range(n):
            chunk = fin.read(rec_size)
            if len(chunk) != rec_size:
                raise ValueError(f"short read from {src}")
            fout.write(chunk)
    return n, dim


def fvecs_to_raw_bin(src: str, dst: str, n: int | None = None) -> tuple[int, int]:
    total, dim = count_vecs(src)
    limit = total if n is None else min(n, total)
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for _ in range(limit):
            record_dim = struct.unpack("<i", fin.read(4))[0]
            if record_dim != dim:
                raise ValueError(f"inconsistent dim in {src}")
            fout.write(fin.read(4 * dim))
    return limit, dim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-queries", type=int, default=1000)
    parser.add_argument("--dataset-prefix", default="clerc-large-single")
    parser.add_argument("--base-fvecs")
    parser.add_argument("--query-fvecs")
    parser.add_argument("--groundtruth-ivecs")
    args = parser.parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    prefix = args.dataset_prefix
    base_fvecs = os.path.join(dataset_dir, f"{prefix}_base.fvecs")
    query_fvecs = os.path.join(dataset_dir, f"{prefix}_query.fvecs")
    gt_ivecs = os.path.join(dataset_dir, f"{prefix}_groundtruth.ivecs")

    if args.base_fvecs:
        base_fvecs = os.path.abspath(args.base_fvecs)
    if args.query_fvecs:
        query_fvecs = os.path.abspath(args.query_fvecs)
    if args.groundtruth_ivecs:
        gt_ivecs = os.path.abspath(args.groundtruth_ivecs)

    subset_query_fvecs = os.path.join(output_dir, f"query_{args.num_queries}.fvecs")
    subset_gt_ivecs = os.path.join(output_dir, f"groundtruth_{args.num_queries}.ivecs")
    base_bin = os.path.join(output_dir, "base.bin")
    query_bin = os.path.join(output_dir, f"query_{args.num_queries}.bin")
    full_query_bin = os.path.join(output_dir, "query_full.bin")

    q_count, dim = copy_first_records(query_fvecs, subset_query_fvecs, args.num_queries)
    gt_count, gt_k = copy_first_records(gt_ivecs, subset_gt_ivecs, args.num_queries)
    base_count, base_dim = fvecs_to_raw_bin(base_fvecs, base_bin)
    _, query_dim = fvecs_to_raw_bin(query_fvecs, query_bin, args.num_queries)
    full_query_count, full_query_dim = fvecs_to_raw_bin(query_fvecs, full_query_bin)

    if dim != base_dim or dim != query_dim or dim != full_query_dim:
        raise ValueError("dimension mismatch across generated files")

    print(f"base_count={base_count}")
    print(f"query_count={q_count}")
    print(f"query_full_count={full_query_count}")
    print(f"gt_count={gt_count}")
    print(f"dim={dim}")
    print(f"gt_k={gt_k}")
    print(f"base_fvecs={base_fvecs}")
    print(f"query_fvecs={subset_query_fvecs}")
    print(f"groundtruth_ivecs={subset_gt_ivecs}")
    print(f"base_bin={base_bin}")
    print(f"query_bin={query_bin}")
    print(f"query_full_bin={full_query_bin}")


if __name__ == "__main__":
    main()
