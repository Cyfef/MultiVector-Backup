#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import faiss
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-faithful MSMARCO coarse cluster info with adaptive cutoff."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--r-max", type=int, default=10)
    parser.add_argument("--query-top-t", type=int, default=4)
    parser.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive")
    parser.add_argument("--fixed-r", type=int, default=10)
    parser.add_argument("--train-query-embs", type=Path)
    parser.add_argument("--train-qrels", type=Path)
    parser.add_argument("--train-query-batch-size", type=int, default=2048)
    parser.add_argument("--predict-batch-size", type=int, default=200000)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--tree-criterion", type=str, default="gini")
    parser.add_argument("--tree-max-depth", type=int, default=6)
    parser.add_argument("--tree-min-samples-leaf", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def ensure_dirs(root: Path) -> tuple[Path, Path, Path]:
    docdata = root / "docdata"
    cdata = root / "cdata"
    qdata = root / "qdata"
    if not docdata.exists() or not cdata.exists():
        raise FileNotFoundError(f"Expected preprocessed docdata/cdata under {root}")
    qdata.mkdir(parents=True, exist_ok=True)
    return docdata, cdata, qdata


def sorted_shard_indices(docdata: Path) -> list[int]:
    indices = []
    for path in docdata.glob("doclens*.npy"):
        stem = path.stem
        suffix = stem[len("doclens") :]
        if suffix.isdigit():
            indices.append(int(suffix))
    return sorted(indices)


def build_doc_profiles(output_root: Path, r_max: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    docdata, cdata, _ = ensure_dirs(output_root)
    top_ids_path = cdata / f"doc_profile_top_ids_r{r_max}.npy"
    top_scores_path = cdata / f"doc_profile_top_scores_r{r_max}.npy"
    doc_lengths_path = cdata / "doc_profile_doc_lengths.npy"
    idf_path = cdata / "doc_profile_idf.npy"
    fine_to_coarse_path = cdata / "fine_to_coarse.npy"

    if (
        top_ids_path.exists()
        and top_scores_path.exists()
        and doc_lengths_path.exists()
        and idf_path.exists()
        and fine_to_coarse_path.exists()
    ):
        return (
            np.load(top_ids_path, mmap_mode="r"),
            np.load(top_scores_path, mmap_mode="r"),
            np.load(doc_lengths_path, mmap_mode="r"),
            np.load(idf_path, mmap_mode="r"),
        )

    shard_indices = sorted_shard_indices(docdata)
    if not shard_indices:
        raise FileNotFoundError(f"No doclens*.npy shards found in {docdata}")

    fine_centroids = np.asarray(np.load(cdata / "centroids.npy"), dtype=np.float32)
    coarse_centroids = np.asarray(np.load(cdata / "coarse_centroids.npy"), dtype=np.float32)
    coarse_index = faiss.IndexFlatIP(coarse_centroids.shape[1])
    coarse_index.add(np.ascontiguousarray(coarse_centroids))
    _, fine_to_coarse = coarse_index.search(np.ascontiguousarray(fine_centroids), 1)
    fine_to_coarse = fine_to_coarse.reshape(-1).astype(np.int32, copy=False)
    np.save(fine_to_coarse_path, fine_to_coarse)

    num_docs = 0
    for shard_idx in shard_indices:
        shard_chunks = np.asarray(np.load(docdata / f"doclens{shard_idx}.npy"), dtype=np.int64)
        num_docs += len(shard_chunks)

    doc_freq = np.zeros(coarse_centroids.shape[0], dtype=np.int64)
    doc_lengths = np.empty(num_docs, dtype=np.int32)
    doc_cursor = 0
    for shard_idx in tqdm(shard_indices, desc="Pass 1/2 coarse DF"):
        shard_chunks = np.asarray(np.load(docdata / f"doclens{shard_idx}.npy"), dtype=np.int64)
        shard_codes = np.asarray(np.load(docdata / f"doc_codes_{shard_idx}.npy"), dtype=np.int32)
        local_cursor = 0
        for chunk_len in shard_chunks:
            token_codes = shard_codes[local_cursor : local_cursor + chunk_len]
            local_cursor += chunk_len
            coarse_ids = np.unique(fine_to_coarse[token_codes])
            doc_freq[coarse_ids] += 1
            doc_lengths[doc_cursor] = int(chunk_len)
            doc_cursor += 1

    idf = np.log(num_docs / (1.0 + doc_freq.astype(np.float64))).astype(np.float32)
    top_ids = np.full((num_docs, r_max), -1, dtype=np.int32)
    top_scores = np.zeros((num_docs, r_max), dtype=np.float32)

    doc_cursor = 0
    for shard_idx in tqdm(shard_indices, desc="Pass 2/2 TF-IDF profiles"):
        shard_chunks = np.asarray(np.load(docdata / f"doclens{shard_idx}.npy"), dtype=np.int64)
        shard_codes = np.asarray(np.load(docdata / f"doc_codes_{shard_idx}.npy"), dtype=np.int32)
        local_cursor = 0
        for chunk_len in shard_chunks:
            token_codes = shard_codes[local_cursor : local_cursor + chunk_len]
            local_cursor += chunk_len
            coarse_ids, tf_counts = np.unique(fine_to_coarse[token_codes], return_counts=True)
            tfidf = tf_counts.astype(np.float32) * idf[coarse_ids]
            order = np.argsort(tfidf)[::-1][:r_max]
            keep = len(order)
            top_ids[doc_cursor, :keep] = coarse_ids[order]
            top_scores[doc_cursor, :keep] = tfidf[order]
            doc_cursor += 1

    np.save(top_ids_path, top_ids)
    np.save(top_scores_path, top_scores)
    np.save(doc_lengths_path, doc_lengths)
    np.save(idf_path, idf)

    return top_ids, top_scores, doc_lengths, idf


def load_positive_ids(train_qrels: Path) -> np.ndarray:
    positive_ids = []
    with train_qrels.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if line_no == 1 and parts[:2] == ["query-id", "corpus-id"]:
                continue
            if len(parts) < 2:
                continue
            positive_ids.append(int(parts[1]))
    if not positive_ids:
        raise ValueError(f"No training pairs found in {train_qrels}")
    return np.asarray(positive_ids, dtype=np.int32)


def compute_relevant_cluster_sets(
    coarse_index: faiss.IndexFlatIP, query_batch: np.ndarray, top_t: int
) -> list[np.ndarray]:
    batch_size, query_len, dim = query_batch.shape
    flat = np.asarray(query_batch.reshape(batch_size * query_len, dim), dtype=np.float32)
    _, cluster_ids = coarse_index.search(np.ascontiguousarray(flat), top_t)
    cluster_ids = cluster_ids.reshape(batch_size, query_len * top_t)
    return [np.unique(row).astype(np.int32, copy=False) for row in cluster_ids]


def build_training_examples(
    coarse_centroids: np.ndarray,
    top_ids: np.ndarray,
    top_scores: np.ndarray,
    doc_lengths: np.ndarray,
    train_query_embs: Path,
    train_qrels: Path,
    top_t: int,
    r_max: int,
    batch_size: int,
    max_train_pairs: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_tensor = np.load(train_query_embs, mmap_mode="r")
    if query_tensor.ndim != 3:
        raise ValueError(f"Expected train query embeddings with shape [N, L, D], got {query_tensor.shape}")

    positive_ids = load_positive_ids(train_qrels)
    if len(positive_ids) != query_tensor.shape[0]:
        raise ValueError(
            f"Train query / qrels size mismatch: {query_tensor.shape[0]} embeddings vs {len(positive_ids)} pairs. "
            "Paper-faithful adaptive cutoff expects pair-aligned training queries."
        )

    num_pairs = len(positive_ids)
    if max_train_pairs > 0:
        num_pairs = min(num_pairs, max_train_pairs)
        positive_ids = positive_ids[:num_pairs]

    features = np.empty((num_pairs, r_max + 1), dtype=np.float32)
    labels = np.empty(num_pairs, dtype=np.int16)

    coarse_index = faiss.IndexFlatIP(coarse_centroids.shape[1])
    coarse_index.add(np.ascontiguousarray(coarse_centroids))

    out_cursor = 0
    for start in tqdm(range(0, num_pairs, batch_size), desc="Labelling adaptive cutoff pairs"):
        end = min(start + batch_size, num_pairs)
        query_batch = np.asarray(query_tensor[start:end], dtype=np.float32)
        relevant_cluster_sets = compute_relevant_cluster_sets(coarse_index, query_batch, top_t)
        batch_pids = positive_ids[start:end]

        features[out_cursor : out_cursor + len(batch_pids), :r_max] = top_scores[batch_pids]
        features[out_cursor : out_cursor + len(batch_pids), r_max] = doc_lengths[batch_pids]

        for i, positive_pid in enumerate(batch_pids):
            label = r_max
            relevant = relevant_cluster_sets[i]
            for rank, coarse_id in enumerate(top_ids[positive_pid]):
                if coarse_id >= 0 and np.any(relevant == coarse_id):
                    label = rank + 1
                    break
            labels[out_cursor + i] = label

        out_cursor += len(batch_pids)

    return features, labels


def train_adaptive_cutoff_model(
    output_root: Path,
    top_ids: np.ndarray,
    top_scores: np.ndarray,
    doc_lengths: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    _, cdata, _ = ensure_dirs(output_root)
    if args.train_query_embs is None or args.train_qrels is None:
        raise ValueError("Adaptive mode requires --train-query-embs and --train-qrels")

    coarse_centroids = np.asarray(np.load(cdata / "coarse_centroids.npy"), dtype=np.float32)
    features, labels = build_training_examples(
        coarse_centroids=coarse_centroids,
        top_ids=top_ids,
        top_scores=top_scores,
        doc_lengths=doc_lengths,
        train_query_embs=args.train_query_embs,
        train_qrels=args.train_qrels,
        top_t=args.query_top_t,
        r_max=args.r_max,
        batch_size=args.train_query_batch_size,
        max_train_pairs=args.max_train_pairs,
    )

    classifier = DecisionTreeClassifier(
        criterion=args.tree_criterion,
        max_depth=args.tree_max_depth,
        min_samples_leaf=args.tree_min_samples_leaf,
        random_state=args.seed,
    )
    classifier.fit(features, labels)

    model_path = cdata / "adaptive_cutoff_model.joblib"
    joblib.dump(classifier, model_path)

    metadata = {
        "mode": "adaptive",
        "r_max": args.r_max,
        "query_top_t": args.query_top_t,
        "tree_criterion": args.tree_criterion,
        "tree_max_depth": args.tree_max_depth,
        "tree_min_samples_leaf": args.tree_min_samples_leaf,
        "seed": args.seed,
        "train_pairs": int(len(labels)),
        "label_histogram": {str(int(v)): int(c) for v, c in zip(*np.unique(labels, return_counts=True))},
    }
    with (cdata / "adaptive_cutoff_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    predicted_r = np.empty(len(doc_lengths), dtype=np.int16)
    for start in tqdm(range(0, len(doc_lengths), args.predict_batch_size), desc="Predicting per-doc r"):
        end = min(start + args.predict_batch_size, len(doc_lengths))
        batch_features = np.empty((end - start, args.r_max + 1), dtype=np.float32)
        batch_features[:, : args.r_max] = top_scores[start:end]
        batch_features[:, args.r_max] = doc_lengths[start:end]
        predicted_r[start:end] = classifier.predict(batch_features).astype(np.int16, copy=False)

    predicted_r = np.clip(predicted_r, 1, args.r_max)
    np.save(cdata / "predicted_r.npy", predicted_r)
    return predicted_r


def fixed_cutoff(output_root: Path, num_docs: int, fixed_r: int) -> np.ndarray:
    _, cdata, _ = ensure_dirs(output_root)
    predicted_r = np.full(num_docs, fixed_r, dtype=np.int16)
    np.save(cdata / "predicted_r.npy", predicted_r)
    with (cdata / "adaptive_cutoff_metadata.json").open("w", encoding="utf-8") as f:
        json.dump({"mode": "fixed", "fixed_r": fixed_r}, f, indent=2, sort_keys=True)
    return predicted_r


def write_coarse_cluster_info(
    output_root: Path, top_ids: np.ndarray, predicted_r: np.ndarray
) -> None:
    _, cdata, _ = ensure_dirs(output_root)
    coarse_k = int(np.asarray(np.load(cdata / "coarse_centroids.npy"), dtype=np.float32).shape[0])
    cluster_members = [[] for _ in range(coarse_k)]

    for doc_id in tqdm(range(len(predicted_r)), desc="Writing coarse cluster memberships"):
        keep = int(predicted_r[doc_id])
        chosen = top_ids[doc_id, :keep]
        for coarse_id in chosen:
            if coarse_id >= 0:
                cluster_members[int(coarse_id)].append(doc_id)

    coarse_info_path = cdata / "coarse_cluster_info.txt"
    with coarse_info_path.open("w", encoding="utf-8") as f:
        for members in cluster_members:
            f.write(" ".join(str(doc_id) for doc_id in members))
            f.write("\n")


def main() -> None:
    args = parse_args()
    top_ids, top_scores, doc_lengths, _ = build_doc_profiles(args.output_root, args.r_max)

    if args.mode == "adaptive":
        predicted_r = train_adaptive_cutoff_model(
            output_root=args.output_root,
            top_ids=top_ids,
            top_scores=top_scores,
            doc_lengths=doc_lengths,
            args=args,
        )
    else:
        predicted_r = fixed_cutoff(args.output_root, len(doc_lengths), args.fixed_r)

    write_coarse_cluster_info(args.output_root, top_ids, predicted_r)
    print("coarse_info_mode", args.mode)
    print("r_max", args.r_max)
    print("predicted_r_minmax", int(predicted_r.min()), int(predicted_r.max()))


if __name__ == "__main__":
    main()
