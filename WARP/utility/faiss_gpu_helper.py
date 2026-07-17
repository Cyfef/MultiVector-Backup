#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np

EPSILON_RAMP_MAX = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU helper for faiss-based k-means and centroid assignment."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train-kmeans")
    train.add_argument("--sample-path", required=True)
    train.add_argument("--output-centroids", required=True)
    train.add_argument("--num-partitions", type=int, required=True)
    train.add_argument("--dim", type=int, required=True)
    train.add_argument("--iters", type=int, default=4)
    train.add_argument("--seed", type=int, default=123)

    compress = subparsers.add_parser("compress-chunk")
    compress.add_argument("--points-path", required=True)
    compress.add_argument("--centroids-path", required=True)
    compress.add_argument("--bucket-cutoffs-path", required=True)
    compress.add_argument("--emb-start", type=int, required=True)
    compress.add_argument("--emb-end", type=int, required=True)
    compress.add_argument("--nbits", type=int, required=True)
    compress.add_argument("--output-codes", required=True)
    compress.add_argument("--output-residuals", required=True)
    compress.add_argument("--batch-size", type=int, default=250_000)

    return parser.parse_args()


def train_kmeans(args: argparse.Namespace) -> None:
    sample = np.ascontiguousarray(np.load(args.sample_path).astype(np.float32, copy=False))
    faiss.normalize_L2(sample)

    kmeans = faiss.Kmeans(
        args.dim,
        args.num_partitions,
        niter=args.iters,
        spherical=True,
        gpu=True,
        verbose=True,
        seed=args.seed,
    )
    kmeans.train(sample)

    centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    faiss.normalize_L2(centroids)
    np.save(args.output_centroids, centroids)


def pack_residuals(residual_ids: np.ndarray, nbits: int) -> np.ndarray:
    bit_positions = np.arange(nbits, dtype=np.uint8)
    bits = ((residual_ids[..., None] >> bit_positions) & 1).reshape(-1)
    packed = np.packbits(bits)
    packed_dim = residual_ids.shape[1] // 8 * nbits
    return packed.reshape(residual_ids.shape[0], packed_dim)


def perturb_for_gpu_search(x: np.ndarray) -> None:
    ramp = np.linspace(0.0, EPSILON_RAMP_MAX, x.shape[1], dtype=np.float32)
    x += ramp[None, :]


def compress_chunk(args: argparse.Namespace) -> None:
    points = np.load(args.points_path, mmap_mode="r")
    centroids = np.ascontiguousarray(
        np.load(args.centroids_path).astype(np.float32, copy=False)
    )
    bucket_cutoffs = np.ascontiguousarray(
        np.load(args.bucket_cutoffs_path).astype(np.float32, copy=False)
    )

    num_embeddings = args.emb_end - args.emb_start
    packed_dim = centroids.shape[1] // 8 * args.nbits

    codes_out = np.lib.format.open_memmap(
        args.output_codes,
        mode="w+",
        dtype=np.int32,
        shape=(num_embeddings,),
    )
    residuals_out = np.lib.format.open_memmap(
        args.output_residuals,
        mode="w+",
        dtype=np.uint8,
        shape=(num_embeddings, packed_dim),
    )

    res = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    config.useFloat16 = False
    index = faiss.GpuIndexFlatIP(res, centroids.shape[1], config)
    index.add(centroids)

    write_offset = 0
    for emb_start in range(args.emb_start, args.emb_end, args.batch_size):
        emb_end = min(emb_start + args.batch_size, args.emb_end)
        batch = np.asarray(points[emb_start:emb_end], dtype=np.float32)
        batch = np.array(batch, dtype=np.float32, copy=True, order="C")
        perturb_for_gpu_search(batch)
        faiss.normalize_L2(batch)

        _, nearest = index.search(batch, 1)
        codes = nearest[:, 0].astype(np.int32, copy=False)
        residual = batch - centroids[codes]
        residual_ids = np.searchsorted(
            bucket_cutoffs,
            residual,
            side="left",
        ).astype(np.uint8, copy=False)
        residuals = pack_residuals(residual_ids, args.nbits)

        next_offset = write_offset + codes.shape[0]
        codes_out[write_offset:next_offset] = codes
        residuals_out[write_offset:next_offset] = residuals
        write_offset = next_offset

    codes_out.flush()
    residuals_out.flush()


def main() -> None:
    args = parse_args()
    if args.command == "train-kmeans":
        train_kmeans(args)
    elif args.command == "compress-chunk":
        compress_chunk(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
