import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> None:
    ensure_dir(path.parent)


def replace_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def ensure_symlink(link_path: Path, target_path: Path, force: bool) -> None:
    if link_path.is_symlink() or link_path.exists():
        if not force:
            if link_path.is_symlink() and Path(link_path.resolve()) == target_path.resolve():
                return
            raise FileExistsError(f"{link_path} already exists; pass --force to replace it")
        replace_path(link_path)
    ensure_parent(link_path)
    link_path.symlink_to(target_path)


def load_matrix(path: Path, label: str) -> np.ndarray:
    matrix = np.load(path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2D array, got shape {matrix.shape} from {path}")
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32, copy=False)
    return matrix


def load_lens(path: Path, label: str) -> np.ndarray:
    lens = np.load(path, mmap_mode="r")
    if lens.ndim != 1:
        raise ValueError(f"{label} must be a 1D array, got shape {lens.shape} from {path}")
    if np.any(lens < 0):
        raise ValueError(f"{label} contains negative lengths: {path}")
    return lens.astype(np.int64, copy=False)


def validate_rows(matrix: np.ndarray, lens: np.ndarray, label: str) -> None:
    expected_rows = int(np.sum(lens, dtype=np.int64))
    actual_rows = int(matrix.shape[0])
    if expected_rows != actual_rows:
        raise ValueError(
            f"{label} length sum mismatch: summed lengths={expected_rows}, "
            f"matrix rows={actual_rows}"
        )


def write_collection_tsv(path: Path, n_doc: int) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows([[doc_id] for doc_id in range(n_doc)])


def write_queries_tsv(path: Path, n_query: int) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for query_id in range(n_query):
            f.write(f"{query_id}\n")


def read_ivecs(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)

    width = int(raw[0])
    row_width = width + 1
    if width < 0 or raw.size % row_width != 0:
        raise ValueError(f"{path} is not a valid .ivecs file")

    rows = raw.reshape(-1, row_width)
    header = rows[:, 0]
    if not np.all(header == width):
        raise ValueError(f"{path} has inconsistent .ivecs row widths")
    return rows[:, 1:]


def write_groundtruth_from_ivecs(path: Path, groundtruth: np.ndarray, n_query: int) -> None:
    if groundtruth.shape[0] != n_query:
        raise ValueError(
            f"ground-truth row count {groundtruth.shape[0]} does not match query count {n_query}"
        )

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for query_id, answer_pids in enumerate(groundtruth):
            filtered = [int(pid) for pid in answer_pids.tolist() if int(pid) >= 0]
            record = {"qid": int(query_id), "answer_pids": filtered}
            f.write(json.dumps(record) + "\n")


def load_id_list(path: Path, label: str) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{label} must be a JSON list in {path}")
    return [str(item) for item in data]


def build_local_qrels(query_ids: list[str], corpus_ids: list[str], qrels_tsv: Path) -> dict[int, list[int]]:
    query_id_to_local = {query_id: idx for idx, query_id in enumerate(query_ids)}
    corpus_id_to_local = {corpus_id: idx for idx, corpus_id in enumerate(corpus_ids)}
    grouped = defaultdict(list)

    with qrels_tsv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and row[0] == "query-id":
                continue
            if len(row) < 2:
                raise ValueError(f"Malformed qrels row in {qrels_tsv}: {row}")

            query_id = str(row[0])
            corpus_id = str(row[1])
            score = float(row[2]) if len(row) >= 3 else 1.0
            if score <= 0:
                continue
            if query_id not in query_id_to_local or corpus_id not in corpus_id_to_local:
                continue
            grouped[query_id_to_local[query_id]].append(corpus_id_to_local[corpus_id])

    return grouped


def write_groundtruth_from_qrels(
    *,
    path: Path,
    query_ids_json: Path,
    corpus_ids_json: Path,
    qrels_tsv: Path,
    n_query: int,
) -> None:
    query_ids = load_id_list(query_ids_json, "query ids")
    corpus_ids = load_id_list(corpus_ids_json, "corpus ids")
    if len(query_ids) != n_query:
        raise ValueError(
            f"query id count {len(query_ids)} does not match query count {n_query}"
        )

    grouped = build_local_qrels(
        query_ids=query_ids,
        corpus_ids=corpus_ids,
        qrels_tsv=qrels_tsv,
    )

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for query_id in range(n_query):
            record = {"qid": int(query_id), "answer_pids": grouped.get(query_id, [])}
            f.write(json.dumps(record) + "\n")


def write_groundtruth_from_local_qrels(
    *,
    path: Path,
    qrels_tsv: Path,
    n_query: int,
    n_doc: int,
) -> None:
    grouped = defaultdict(list)

    with qrels_tsv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row_idx, row in enumerate(reader):
            if not row:
                continue
            if row_idx == 0 and row[0] == "query-id":
                continue
            if len(row) < 2:
                raise ValueError(f"Malformed local qrels row in {qrels_tsv}: {row}")

            qid = int(row[0])
            pid = int(row[1])
            score = float(row[2]) if len(row) >= 3 else 1.0
            if score <= 0:
                continue
            if qid < 0 or qid >= n_query:
                raise ValueError(f"query id {qid} out of range 0..{n_query - 1} in {qrels_tsv}")
            if pid < 0 or pid >= n_doc:
                raise ValueError(f"document id {pid} out of range 0..{n_doc - 1} in {qrels_tsv}")
            grouped[qid].append(pid)

    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for query_id in range(n_query):
            record = {"qid": int(query_id), "answer_pids": grouped.get(query_id, [])}
            f.write(json.dumps(record) + "\n")


def write_empty_groundtruth(path: Path, n_query: int) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for query_id in range(n_query):
            record = {"qid": int(query_id), "answer_pids": []}
            f.write(json.dumps(record) + "\n")


def read_groundtruth_jsonl(path: Path, n_query: int) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = {query_id: [] for query_id in range(n_query)}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            record = json.loads(line)
            if "qid" not in record or "answer_pids" not in record:
                raise KeyError(f"{path} line {line_no} must contain qid and answer_pids")
            qid = int(record["qid"])
            if qid < 0 or qid >= n_query:
                raise ValueError(f"qid {qid} in {path} is out of range 0..{n_query - 1}")

            deduped = []
            seen = set()
            for pid in record["answer_pids"]:
                pid = int(pid)
                if pid in seen:
                    continue
                seen.add(pid)
                deduped.append(pid)
            grouped[qid] = deduped
    return grouped


def write_plaid_groundtruth_tsvs(
    *,
    dataset: str,
    embedding_dir: Path,
    groundtruth_jsonl: Path,
    n_query: int,
    topk_values: tuple[int, ...] = (10, 100),
) -> None:
    ensure_dir(embedding_dir)
    grouped = read_groundtruth_jsonl(groundtruth_jsonl, n_query=n_query)

    for topk in topk_values:
        output_path = embedding_dir / f"{dataset}-groundtruth-top{topk}--.tsv"
        with output_path.open("w", encoding="utf-8") as f:
            for qid in range(n_query):
                for rank, pid in enumerate(grouped[qid][:topk], start=1):
                    f.write(f"{qid}\t{pid}\t{rank}\t1\n")


def write_document_shards(
    *,
    doc_embeddings: np.ndarray,
    doc_lens: np.ndarray,
    transformed_dir: Path,
    batch_size_docs: int,
    force: bool,
) -> int:
    if transformed_dir.exists():
        if not force:
            raise FileExistsError(f"{transformed_dir} already exists; pass --force to replace it")
        replace_path(transformed_dir)
    ensure_dir(transformed_dir)

    offsets = np.concatenate(([0], np.cumsum(doc_lens, dtype=np.int64)))
    n_doc = int(doc_lens.shape[0])
    torch.save(n_doc, transformed_dir / "doc_count")

    shard_count = 0
    for start_doc in tqdm(range(0, n_doc, batch_size_docs), desc="writing document shards"):
        end_doc = min(start_doc + batch_size_docs, n_doc)
        vec_start = int(offsets[start_doc])
        vec_end = int(offsets[end_doc])

        batch_vectors = np.asarray(doc_embeddings[vec_start:vec_end], dtype=np.float32)
        batch_doc_lens = doc_lens[start_doc:end_doc].astype(np.int64).tolist()
        batch_tensor = torch.from_numpy(np.array(batch_vectors, copy=True))
        torch.save((batch_tensor, batch_doc_lens), transformed_dir / f"embeddings.{shard_count}.pt")
        shard_count += 1

    return shard_count


def build_manifest(
    *,
    username: str,
    dataset: str,
    doc_embeddings_path: Path,
    doc_lens_path: Path,
    query_embeddings_path: Path,
    query_lens_path: Path,
    prepared_query_lens_path: Path,
    groundtruth_ivecs: Path | None,
    qrels_tsv: Path | None,
    query_ids_json: Path | None,
    corpus_ids_json: Path | None,
    flat_dataset_root: Path,
    transformed_dir: Path,
    groundtruth_jsonl: Path,
    embedding_dir: Path,
    raw_document_dir: Path,
    n_doc: int,
    n_doc_vectors: int,
    n_query: int,
    n_query_vectors: int,
    dim: int,
    shard_count: int,
) -> dict:
    return {
        "username": username,
        "dataset": dataset,
        "source": {
            "doc_embeddings": str(doc_embeddings_path.resolve()),
            "doc_lens": str(doc_lens_path.resolve()),
            "query_embeddings": str(query_embeddings_path.resolve()),
            "query_lens": str(query_lens_path.resolve()),
            "groundtruth_ivecs": str(groundtruth_ivecs.resolve()) if groundtruth_ivecs else None,
            "qrels_tsv": str(qrels_tsv.resolve()) if qrels_tsv else None,
            "query_ids_json": str(query_ids_json.resolve()) if query_ids_json else None,
            "corpus_ids_json": str(corpus_ids_json.resolve()) if corpus_ids_json else None,
        },
        "counts": {
            "n_doc": n_doc,
            "n_doc_vectors": n_doc_vectors,
            "n_query": n_query,
            "n_query_vectors": n_query_vectors,
            "dim": dim,
        },
        "prepared": {
            "flat_dataset_root": str(flat_dataset_root.resolve()),
            "doc_transformed_embeddings": str(transformed_dir.resolve()),
            "doc_count_file": str((transformed_dir / "doc_count").resolve()),
            "prepared_query_lens": str(prepared_query_lens_path.resolve()),
            "groundtruth_jsonl": str(groundtruth_jsonl.resolve()),
            "embedding_dir": str(embedding_dir.resolve()),
            "plaid_groundtruth_tsv_top10": str(
                (embedding_dir / f"{dataset}-groundtruth-top10--.tsv").resolve()
            ),
            "plaid_groundtruth_tsv_top100": str(
                (embedding_dir / f"{dataset}-groundtruth-top100--.tsv").resolve()
            ),
            "raw_document_dir": str(raw_document_dir.resolve()),
            "collection_tsv": str((raw_document_dir / "collection.tsv").resolve()),
            "queries_dev_tsv": str((raw_document_dir / "queries.dev.tsv").resolve()),
            "shard_count": shard_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a flat multi-vector dataset for Plaid/Dessert/MUVERA/IGP without changing model code."
    )
    parser.add_argument("--username", type=str, default="ali")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--doc-embeddings", type=Path, required=True)
    parser.add_argument("--doc-lens", type=Path, required=True)
    parser.add_argument("--query-embeddings", type=Path, required=True)
    parser.add_argument("--query-lens", type=Path, required=True)
    parser.add_argument("--groundtruth-ivecs", type=Path, default=None)
    parser.add_argument("--qrels-tsv", type=Path, default=None)
    parser.add_argument("--local-qrels-tsv", type=Path, default=None)
    parser.add_argument("--query-ids-json", type=Path, default=None)
    parser.add_argument("--corpus-ids-json", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument("--flat-root", type=Path, default=None)
    parser.add_argument("--batch-size-docs", type=int, default=2500)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    runtime_root = args.runtime_root or Path(f"/data/{args.username}/Dataset/multi-vector-retrieval")
    flat_root = args.flat_root or (runtime_root / "FlatData")
    flat_dataset_root = flat_root / args.dataset
    transformed_dir = flat_dataset_root / "doc_embeddings" / "transformed_embeddings"
    query_meta_dir = flat_dataset_root / "query_embeddings" / "transformed_embeddings"
    groundtruth_jsonl = flat_dataset_root / "query_groundtruth" / "queries.gnd.jsonl"
    embedding_dir = runtime_root / "Embedding" / args.dataset
    raw_document_dir = runtime_root / "RawData" / args.dataset / "document"
    manifest_path = flat_dataset_root / "manifest.json"

    if flat_dataset_root.exists() and args.force:
        replace_path(flat_dataset_root)
    ensure_dir(flat_dataset_root)
    ensure_dir(query_meta_dir)

    doc_embeddings = load_matrix(args.doc_embeddings, "document embeddings")
    doc_lens = load_lens(args.doc_lens, "document lengths")
    query_embeddings = load_matrix(args.query_embeddings, "query embeddings")
    query_lens = load_lens(args.query_lens, "query lengths")

    validate_rows(doc_embeddings, doc_lens, "document")
    validate_rows(query_embeddings, query_lens, "query")

    if doc_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            f"document dim {doc_embeddings.shape[1]} does not match query dim {query_embeddings.shape[1]}"
        )

    if args.max_queries is not None:
        if args.max_queries <= 0:
            raise ValueError("--max-queries must be positive")
        query_lens = query_lens[: args.max_queries]

    prepared_query_lens_path = query_meta_dir / "query_n_vec_length.npy"
    np.save(prepared_query_lens_path, query_lens)

    n_doc = int(doc_lens.shape[0])
    n_doc_vectors = int(doc_embeddings.shape[0])
    n_query = int(query_lens.shape[0])
    n_query_vectors = int(np.sum(query_lens, dtype=np.int64))
    dim = int(doc_embeddings.shape[1])

    shard_count = write_document_shards(
        doc_embeddings=doc_embeddings,
        doc_lens=doc_lens,
        transformed_dir=transformed_dir,
        batch_size_docs=args.batch_size_docs,
        force=args.force,
    )

    if args.groundtruth_ivecs is not None:
        groundtruth = read_ivecs(args.groundtruth_ivecs)
        if args.max_queries is not None:
            groundtruth = groundtruth[: args.max_queries]
        write_groundtruth_from_ivecs(
            path=groundtruth_jsonl,
            groundtruth=groundtruth,
            n_query=n_query,
        )
    elif args.local_qrels_tsv is not None:
        write_groundtruth_from_local_qrels(
            path=groundtruth_jsonl,
            qrels_tsv=args.local_qrels_tsv,
            n_query=n_query,
            n_doc=n_doc,
        )
    elif args.qrels_tsv is not None:
        if args.query_ids_json is None or args.corpus_ids_json is None:
            raise ValueError("--qrels-tsv requires both --query-ids-json and --corpus-ids-json")
        write_groundtruth_from_qrels(
            path=groundtruth_jsonl,
            query_ids_json=args.query_ids_json,
            corpus_ids_json=args.corpus_ids_json,
            qrels_tsv=args.qrels_tsv,
            n_query=n_query,
        )
    else:
        write_empty_groundtruth(groundtruth_jsonl, n_query=n_query)

    if raw_document_dir.exists() and args.force:
        replace_path(raw_document_dir)
    ensure_dir(raw_document_dir)
    write_collection_tsv(raw_document_dir / "collection.tsv", n_doc=n_doc)
    write_queries_tsv(raw_document_dir / "queries.dev.tsv", n_query=n_query)
    ensure_symlink(raw_document_dir / "transformed_embeddings", transformed_dir, force=args.force)
    ensure_symlink(raw_document_dir / "queries.gnd.jsonl", groundtruth_jsonl, force=args.force)

    write_plaid_groundtruth_tsvs(
        dataset=args.dataset,
        embedding_dir=embedding_dir,
        groundtruth_jsonl=groundtruth_jsonl,
        n_query=n_query,
    )

    manifest = build_manifest(
        username=args.username,
        dataset=args.dataset,
        doc_embeddings_path=args.doc_embeddings,
        doc_lens_path=args.doc_lens,
        query_embeddings_path=args.query_embeddings,
        query_lens_path=args.query_lens,
        prepared_query_lens_path=prepared_query_lens_path,
        groundtruth_ivecs=args.groundtruth_ivecs,
        qrels_tsv=args.qrels_tsv,
        query_ids_json=args.query_ids_json,
        corpus_ids_json=args.corpus_ids_json,
        flat_dataset_root=flat_dataset_root,
        transformed_dir=transformed_dir,
        groundtruth_jsonl=groundtruth_jsonl,
        embedding_dir=embedding_dir,
        raw_document_dir=raw_document_dir,
        n_doc=n_doc,
        n_doc_vectors=n_doc_vectors,
        n_query=n_query,
        n_query_vectors=n_query_vectors,
        dim=dim,
        shard_count=shard_count,
    )
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"prepared dataset manifest: {manifest_path}")
    print(f"document shards: {transformed_dir}")
    print(f"ground truth jsonl: {groundtruth_jsonl}")
    print(f"plaid compatibility tsvs: {embedding_dir / f'{args.dataset}-groundtruth-top10--.tsv'}")
    print(f"plaid compatibility tsvs: {embedding_dir / f'{args.dataset}-groundtruth-top100--.tsv'}")
    print(f"canonical raw metadata: {raw_document_dir}")
    print(f"documents: {n_doc}, document vectors: {n_doc_vectors}, dim: {dim}")
    print(f"queries: {n_query}, query vectors used: {n_query_vectors}")


if __name__ == "__main__":
    main()
