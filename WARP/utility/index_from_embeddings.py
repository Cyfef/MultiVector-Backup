#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def startup_log(message: str) -> None:
    print(f"[{time.strftime('%b %d, %H:%M:%S')}] #> {message}", flush=True)


startup_log("index_from_embeddings startup: importing numpy")
import numpy as np
startup_log("index_from_embeddings startup: imported numpy")
startup_log("index_from_embeddings startup: importing torch")
import torch
startup_log("index_from_embeddings startup: imported torch")
startup_log("index_from_embeddings startup: importing faiss")
import faiss
startup_log("index_from_embeddings startup: imported faiss")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

startup_log("index_from_embeddings startup: importing WARP modules")
from warp.infra.config import ColBERTConfig
from warp.indexing.codecs.residual import ResidualCodec
from warp.indexing.codecs.residual_embeddings import ResidualEmbeddings
from warp.indexing.index_saver import IndexSaver
from warp.indexing.utils import optimize_ivf
from warp.utils.utils import print_message
startup_log("index_from_embeddings startup: imported WARP modules")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a WARP/ColBERT index directly from packed token embeddings."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--encoder", default="colbert")
    parser.add_argument(
        "--embedding-root",
        default="/data1/liuyaoyang/Papers/ACFDE/output",
    )
    parser.add_argument(
        "--embedding-dir",
        help="Optional direct directory containing corpus/query points and offsets.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data1/liuyaoyang/Papers/ACFDE/emb-Models/colbertv2.0",
        help="Stored in index metadata for compatibility with ColBERT-style loaders.",
    )
    parser.add_argument(
        "--index-root",
        default=os.environ.get("INDEX_ROOT"),
        help="Destination root for indexes. Defaults to $INDEX_ROOT.",
    )
    parser.add_argument("--index-name")
    parser.add_argument("--nbits", type=int, default=2, choices=[1, 2, 4, 8])
    parser.add_argument("--kmeans-iters", type=int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Number of documents per saved index chunk.",
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=65_536,
        help="Upper bound on the number of k-means centroids.",
    )
    parser.add_argument(
        "--sample-per-centroid",
        type=int,
        default=32,
        help="Training sample budget per centroid.",
    )
    parser.add_argument(
        "--max-sample-embeddings",
        type=int,
        default=4_000_000,
        help="Hard cap on sampled embeddings used for k-means training.",
    )
    parser.add_argument("--query-maxlen", type=int)
    parser.add_argument("--doc-maxlen", type=int)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--threads",
        type=int,
        help="CPU thread count for FAISS and Torch during index construction.",
    )
    parser.add_argument(
        "--faiss-gpu-python",
        help="Optional Python executable with faiss-gpu for hybrid GPU acceleration.",
    )
    parser.add_argument(
        "--faiss-gpu-visible-device",
        help="Optional CUDA_VISIBLE_DEVICES value for the faiss-gpu helper.",
    )
    parser.add_argument(
        "--faiss-gpu-batch-size",
        type=int,
        default=250_000,
        help="Embedding batch size per helper invocation when using faiss-gpu.",
    )
    parser.add_argument(
        "--tmp-root",
        default="/tmp",
        help="Temporary directory root for intermediate helper artifacts.",
    )
    return parser.parse_args()


def infer_split(dataset: str) -> str:
    return "dev" if dataset in {"msmarco", "msmarco_small"} else "test"


def default_index_name(dataset: str, encoder: str, split: str, nbits: int) -> str:
    return f"beir-{dataset}.split={split}.precomputed={encoder}.nbits={nbits}"


def load_offsets(path: Path) -> np.ndarray:
    print_message(f"#> Loading offsets from {path}")
    offsets = np.load(path)
    if offsets.ndim != 1 or offsets.size < 2:
        raise ValueError(f"Invalid offsets array at {path}")
    print_message(f"#> Loaded offsets shape={offsets.shape} dtype={offsets.dtype}")
    return offsets.astype(np.int64, copy=False)


def load_points(path: Path) -> np.memmap:
    print_message(f"#> Memory-mapping points from {path}")
    points = np.load(path, mmap_mode="r")
    if points.ndim != 2:
        raise ValueError(f"Invalid points array at {path}")
    print_message(f"#> Memory-mapped points shape={points.shape} dtype={points.dtype}")
    return points


def infer_num_partitions(num_embeddings: int, max_partitions: int) -> int:
    raw = int(2 ** math.floor(math.log2(max(1.0, 16.0 * math.sqrt(num_embeddings)))))
    return min(raw, max_partitions)


def configure_threads(threads: int | None) -> None:
    if threads is None:
        return
    if threads < 1:
        raise ValueError("--threads must be >= 1")

    faiss.omp_set_num_threads(threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def run_faiss_gpu_helper(
    python_bin: str,
    helper_args: list[str],
    visible_device: str | None,
) -> None:
    helper_script = REPO_ROOT / "utility" / "faiss_gpu_helper.py"
    env = os.environ.copy()
    if visible_device:
        env["CUDA_VISIBLE_DEVICES"] = visible_device

    subprocess.run(
        [python_bin, str(helper_script), *helper_args],
        check=True,
        env=env,
    )


def make_config(
    args: argparse.Namespace,
    index_name: str,
    dim: int,
    doc_maxlen: int,
    query_maxlen: int,
) -> ColBERTConfig:
    config = ColBERTConfig(
        checkpoint=args.checkpoint,
        index_root=args.index_root,
        index_name=index_name,
        dim=dim,
        doc_maxlen=doc_maxlen,
        query_maxlen=query_maxlen,
        nbits=args.nbits,
        kmeans_niters=args.kmeans_iters,
        interaction="colbert",
        similarity="cosine",
        rank=0,
        nranks=1,
    )
    return config


def write_plan(
    config: ColBERTConfig,
    index_path: Path,
    num_chunks: int,
    num_partitions: int,
    num_embeddings: int,
    avg_doclen: float,
) -> None:
    plan = {
        "config": config.export(),
        "num_chunks": num_chunks,
        "num_partitions": num_partitions,
        "num_embeddings_est": int(num_embeddings),
        "avg_doclen_est": float(avg_doclen),
    }
    with open(index_path / "plan.json", "w") as f:
        import ujson

        f.write(ujson.dumps(plan, indent=4) + "\n")


def sample_training_embeddings(
    points: np.memmap,
    num_embeddings: int,
    num_partitions: int,
    sample_per_centroid: int,
    max_sample_embeddings: int,
    seed: int,
) -> torch.Tensor:
    started = time.perf_counter()
    target = min(
        num_embeddings,
        sample_per_centroid * num_partitions,
        max_sample_embeddings,
    )
    if target < num_partitions:
        raise ValueError(
            f"Training sample ({target}) is smaller than num_partitions ({num_partitions}). "
            "Lower --max-partitions or raise --max-sample-embeddings."
        )

    print_message(
        f"#> Sampling {target:,} training embeddings from {num_embeddings:,} total "
        f"(seed={seed}, sample_per_centroid={sample_per_centroid}, max_sample_embeddings={max_sample_embeddings})"
    )
    rng = np.random.default_rng(seed)
    use_block_sampling = isinstance(points, np.memmap) and target >= 100_000

    if use_block_sampling:
        block_size = min(65_536, target)
        block_count = math.ceil(target / block_size)
        max_start = max(0, num_embeddings - block_size)
        if block_count == 1 or max_start == 0:
            starts = np.array([0], dtype=np.int64)
        else:
            anchors = np.linspace(0, max_start, num=block_count, dtype=np.int64)
            step = max(1, max_start // max(1, block_count - 1))
            jitter = max(1, step // 4)
            offsets = rng.integers(-jitter, jitter + 1, size=block_count)
            starts = np.clip(anchors + offsets, 0, max_start)
            starts.sort()

        print_message(
            f"#> Using contiguous block sampling with {block_count:,} blocks of up to {block_size:,} embeddings"
        )
        blocks = []
        remaining = target
        for start in starts.tolist():
            if remaining <= 0:
                break
            take = min(block_size, remaining)
            end = start + take
            blocks.append(np.asarray(points[start:end], dtype=np.float32))
            remaining -= take

        if remaining > 0:
            tail_start = max(0, num_embeddings - remaining)
            blocks.append(np.asarray(points[tail_start : tail_start + remaining], dtype=np.float32))

        print_message(f"#> Selected sample blocks in {time.perf_counter() - started:.1f}s")
        sample = np.concatenate(blocks, axis=0)[:target]
        sample = np.ascontiguousarray(sample[rng.permutation(sample.shape[0])])
    else:
        indices = rng.choice(num_embeddings, size=target, replace=False)
        indices.sort()
        print_message(f"#> Sampled training indices in {time.perf_counter() - started:.1f}s")
        sample = np.asarray(points[indices], dtype=np.float32)

    print_message(f"#> Materialized sampled embeddings in {time.perf_counter() - started:.1f}s")
    sample = torch.from_numpy(np.ascontiguousarray(sample))
    sample = torch.nn.functional.normalize(sample, p=2, dim=-1)
    print_message(
        f"#> Training sample ready with shape={tuple(sample.shape)} in {time.perf_counter() - started:.1f}s"
    )
    return sample


def build_codec_from_centroids(
    config: ColBERTConfig,
    sample: torch.Tensor,
    centroids: torch.Tensor,
) -> ResidualCodec:
    centroids = torch.nn.functional.normalize(centroids.float(), dim=-1)

    heldout_fraction = 0.05
    heldout_size = int(min(sample.size(0) * heldout_fraction, 50_000))
    heldout_size = max(1, heldout_size)
    sample = sample[torch.randperm(sample.size(0))]
    train_sample, heldout = sample.split(
        [sample.size(0) - heldout_size, heldout_size],
        dim=0,
    )
    del train_sample

    compressor = ResidualCodec(config=config, centroids=centroids, avg_residual=None)
    heldout_codes = compressor.compress_into_codes(heldout, out_device="cpu")
    heldout_centroids = compressor.lookup_centroids(heldout_codes, out_device="cpu")
    heldout_residual = heldout - heldout_centroids

    avg_residual = torch.abs(heldout_residual).mean(dim=0).cpu()

    num_options = 2 ** config.nbits
    quantiles = torch.arange(0, num_options, dtype=torch.float32) * (1 / num_options)
    bucket_cutoffs = heldout_residual.float().quantile(quantiles[1:])
    bucket_weights = heldout_residual.float().quantile(
        quantiles + (0.5 / num_options)
    )

    return ResidualCodec(
        config=config,
        centroids=centroids,
        avg_residual=avg_residual.mean(),
        bucket_cutoffs=bucket_cutoffs,
        bucket_weights=bucket_weights,
    )


def train_codec(
    args: argparse.Namespace,
    config: ColBERTConfig,
    num_partitions: int,
    sample: torch.Tensor,
    tmp_dir: Path,
) -> ResidualCodec:
    started = time.perf_counter()
    if args.faiss_gpu_python:
        print_message(f"#> Training k-means on {sample.size(0):,} embeddings with faiss-gpu")
        sample_path = tmp_dir / "kmeans_sample.npy"
        centroids_path = tmp_dir / "kmeans_centroids.npy"
        np.save(sample_path, sample.float().numpy())
        run_faiss_gpu_helper(
            args.faiss_gpu_python,
            [
                "train-kmeans",
                "--sample-path",
                str(sample_path),
                "--output-centroids",
                str(centroids_path),
                "--num-partitions",
                str(num_partitions),
                "--dim",
                str(config.dim),
                "--iters",
                str(config.kmeans_niters),
                "--seed",
                str(args.seed),
            ],
            args.faiss_gpu_visible_device,
        )
        centroids = torch.from_numpy(np.load(centroids_path).astype(np.float32, copy=False))
        codec = build_codec_from_centroids(config, sample, centroids)
        print_message(f"#> Built codec from faiss-gpu centroids in {time.perf_counter() - started:.1f}s")
        return codec

    heldout_fraction = 0.05
    heldout_size = int(min(sample.size(0) * heldout_fraction, 50_000))
    heldout_size = max(1, heldout_size)
    sample = sample[torch.randperm(sample.size(0))]
    train_sample, heldout = sample.split(
        [sample.size(0) - heldout_size, heldout_size],
        dim=0,
    )

    print_message(f"#> Training k-means on {train_sample.size(0):,} embeddings")
    kmeans = faiss.Kmeans(
        config.dim,
        num_partitions,
        niter=config.kmeans_niters,
        spherical=True,
        gpu=False,
        verbose=True,
        seed=123,
    )
    kmeans.train(train_sample.float().numpy())
    centroids = torch.from_numpy(kmeans.centroids)
    codec = build_codec_from_centroids(config, torch.cat([train_sample, heldout]), centroids)
    print_message(f"#> Built codec from CPU k-means in {time.perf_counter() - started:.1f}s")
    return codec


def index_embeddings(
    args: argparse.Namespace,
    config: ColBERTConfig,
    index_path: Path,
    points: np.memmap,
    points_path: Path,
    offsets: np.ndarray,
    chunk_size: int,
    tmp_dir: Path,
) -> int:
    started = time.perf_counter()
    saver = IndexSaver(config)
    codec = ResidualCodec.load(str(index_path))
    num_docs = offsets.size - 1
    num_chunks = math.ceil(num_docs / chunk_size)

    saver.codec = codec
    helper_centroids_path = None
    helper_bucket_cutoffs_path = None
    if args.faiss_gpu_python:
        helper_centroids_path = tmp_dir / "codec_centroids.npy"
        helper_bucket_cutoffs_path = tmp_dir / "codec_bucket_cutoffs.npy"
        np.save(helper_centroids_path, codec.centroids.cpu().numpy().astype(np.float32))
        np.save(
            helper_bucket_cutoffs_path,
            codec.bucket_cutoffs.cpu().numpy().astype(np.float32),
        )

    with saver.thread():
        for chunk_idx, doc_start in enumerate(range(0, num_docs, chunk_size)):
            doc_end = min(doc_start + chunk_size, num_docs)
            emb_start = int(offsets[doc_start])
            emb_end = int(offsets[doc_end])
            doclens = np.diff(offsets[doc_start : doc_end + 1]).astype(np.int32).tolist()

            print_message(
                f"#> Saving chunk {chunk_idx} with {doc_end - doc_start:,} docs "
                f"and {emb_end - emb_start:,} embeddings"
            )
            if args.faiss_gpu_python:
                codes_path = tmp_dir / f"chunk_{chunk_idx}.codes.npy"
                residuals_path = tmp_dir / f"chunk_{chunk_idx}.residuals.npy"
                run_faiss_gpu_helper(
                    args.faiss_gpu_python,
                    [
                        "compress-chunk",
                        "--points-path",
                        str(points_path),
                        "--centroids-path",
                        str(helper_centroids_path),
                        "--bucket-cutoffs-path",
                        str(helper_bucket_cutoffs_path),
                        "--emb-start",
                        str(emb_start),
                        "--emb-end",
                        str(emb_end),
                        "--nbits",
                        str(config.nbits),
                        "--output-codes",
                        str(codes_path),
                        "--output-residuals",
                        str(residuals_path),
                        "--batch-size",
                        str(args.faiss_gpu_batch_size),
                    ],
                    args.faiss_gpu_visible_device,
                )
                compressed_embs = ResidualEmbeddings(
                    torch.from_numpy(np.load(codes_path).astype(np.int32, copy=False)),
                    torch.from_numpy(np.load(residuals_path).astype(np.uint8, copy=False)),
                )
                saver.save_compressed_chunk(chunk_idx, doc_start, compressed_embs, doclens)
                codes_path.unlink(missing_ok=True)
                residuals_path.unlink(missing_ok=True)
            else:
                embs = np.asarray(points[emb_start:emb_end], dtype=np.float32)
                embs = torch.from_numpy(np.ascontiguousarray(embs))
                embs = torch.nn.functional.normalize(embs, p=2, dim=-1)
                saver.save_chunk(chunk_idx, doc_start, embs, doclens)

    print_message(f"#> Finished writing {num_chunks:,} chunks in {time.perf_counter() - started:.1f}s")
    return num_chunks


def finalize_index(
    config: ColBERTConfig,
    index_path: Path,
    num_chunks: int,
    num_embeddings: int,
    num_docs: int,
) -> None:
    started = time.perf_counter()
    import ujson

    with open(index_path / "plan.json") as f:
        plan = ujson.load(f)
    num_partitions = int(plan["num_partitions"])

    embedding_offsets = []
    passage_offset = 0
    embedding_offset = 0

    for chunk_idx in range(num_chunks):
        metadata_path = index_path / f"{chunk_idx}.metadata.json"
        with open(metadata_path) as f:
            metadata = ujson.load(f)

        metadata["embedding_offset"] = embedding_offset
        embedding_offsets.append(embedding_offset)

        if metadata["passage_offset"] != passage_offset:
            raise ValueError(
                f"Unexpected passage_offset in chunk {chunk_idx}: "
                f"{metadata['passage_offset']} != {passage_offset}"
            )

        passage_offset += metadata["num_passages"]
        embedding_offset += metadata["num_embeddings"]

        with open(metadata_path, "w") as f:
            f.write(ujson.dumps(metadata, indent=4) + "\n")

    if passage_offset != num_docs or embedding_offset != num_embeddings:
        raise ValueError(
            f"Finalized offsets do not match corpus size: docs={passage_offset}/{num_docs}, "
            f"embeddings={embedding_offset}/{num_embeddings}"
        )

    codes = torch.empty(num_embeddings, dtype=torch.long)
    for chunk_idx in range(num_chunks):
        chunk_codes = ResidualEmbeddings.load_codes(str(index_path), chunk_idx).long()
        start = embedding_offsets[chunk_idx]
        end = start + chunk_codes.size(0)
        codes[start:end] = chunk_codes

    sorter = codes.sort()
    ivf = sorter.indices
    ivf_lengths = torch.bincount(sorter.values, minlength=num_partitions)

    optimize_ivf(ivf, ivf_lengths, str(index_path))

    metadata = {
        "config": config.export(),
        "num_chunks": num_chunks,
        "num_partitions": num_partitions,
        "num_embeddings": int(num_embeddings),
        "avg_doclen": float(num_embeddings / num_docs),
    }
    with open(index_path / "metadata.json", "w") as f:
        f.write(ujson.dumps(metadata, indent=4) + "\n")
    print_message(f"#> Finalized index metadata and IVF in {time.perf_counter() - started:.1f}s")


def main() -> None:
    print_message("#> Entering index_from_embeddings main")
    args = parse_args()
    print_message(f"#> Parsed args for dataset={args.dataset}")
    configure_threads(args.threads)
    print_message(f"#> Configured threads={args.threads}")

    if not args.index_root:
        raise ValueError("--index-root is required when INDEX_ROOT is not set")

    split = infer_split(args.dataset)
    index_name = args.index_name or default_index_name(
        args.dataset, args.encoder, split, args.nbits
    )
    index_root = Path(args.index_root)
    index_path = index_root / index_name

    if index_path.exists() and any(index_path.iterdir()):
        raise FileExistsError(
            f"Index directory already exists and is not empty: {index_path}"
        )
    index_path.mkdir(parents=True, exist_ok=True)
    print_message(f"#> Prepared index directory {index_path}")

    base = Path(args.embedding_dir) if args.embedding_dir else Path(args.embedding_root) / args.dataset / args.encoder
    corpus_points_path = base / "corpus_points.npy"
    corpus_offsets_path = base / "corpus_offsets.npy"
    query_offsets_path = base / "query_offsets.npy"
    print_message(f"#> Using embedding directory {base}")

    points = load_points(corpus_points_path)
    offsets = load_offsets(corpus_offsets_path)

    num_docs = offsets.size - 1
    num_embeddings = int(offsets[-1])
    dim = int(points.shape[1])
    doclens = np.diff(offsets)
    avg_doclen = float(doclens.mean())
    doc_maxlen = args.doc_maxlen or int(doclens.max())

    if args.query_maxlen is not None:
        query_maxlen = args.query_maxlen
    elif query_offsets_path.exists():
        query_offsets = load_offsets(query_offsets_path)
        query_maxlen = int(np.diff(query_offsets).max())
    else:
        query_maxlen = 32

    num_partitions = infer_num_partitions(num_embeddings, args.max_partitions)
    config = make_config(args, index_name, dim, doc_maxlen, query_maxlen)

    num_chunks = math.ceil(num_docs / args.chunk_size)
    write_plan(
        config=config,
        index_path=index_path,
        num_chunks=num_chunks,
        num_partitions=num_partitions,
        num_embeddings=num_embeddings,
        avg_doclen=avg_doclen,
    )

    print_message(
        f"#> dataset={args.dataset} docs={num_docs:,} embeddings={num_embeddings:,} "
        f"dim={dim} query_maxlen={query_maxlen} partitions={num_partitions:,}"
    )

    print_message("#> Starting training sample selection")
    sample = sample_training_embeddings(
        points=points,
        num_embeddings=num_embeddings,
        num_partitions=num_partitions,
        sample_per_centroid=args.sample_per_centroid,
        max_sample_embeddings=args.max_sample_embeddings,
        seed=args.seed,
    )
    with tempfile.TemporaryDirectory(dir=args.tmp_root, prefix=f"warp-index-{args.dataset}-") as tmp:
        tmp_dir = Path(tmp)
        print_message(f"#> Starting codec training with temporary dir {tmp_dir}")
        codec = train_codec(args, config, num_partitions, sample, tmp_dir)
        codec.save(str(index_path))
        print_message("#> Saved codec, starting chunk compression")

        index_embeddings(
            args=args,
            config=config,
            index_path=index_path,
            points=points,
            points_path=corpus_points_path,
            offsets=offsets,
            chunk_size=args.chunk_size,
            tmp_dir=tmp_dir,
        )
    print_message("#> Chunk compression finished, starting index finalization")
    finalize_index(
        config=config,
        index_path=index_path,
        num_chunks=num_chunks,
        num_embeddings=num_embeddings,
        num_docs=num_docs,
    )

    print_message(f"#> Finished building index at {index_path}")


if __name__ == "__main__":
    main()
