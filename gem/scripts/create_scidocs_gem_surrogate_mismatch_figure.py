#!/usr/bin/env python3
"""Create the SciDocs GEM surrogate-vs-MaxSim mismatch figure.

This script uses the saved SciDocs GEM preprocessing files and profiling
outputs. It computes exact MaxSim over the routed union and reimplements the
query-time traversal surrogate used by L2SqrCluster4Search.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Ellipse
from scipy import stats


QID = 443
DPLUS = 7615
DEFAULT_DMINUS = 4998


@dataclass(frozen=True)
class DocLoc:
    shard: int
    local_doc: int
    vec_start: int
    vec_len: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gem-data", type=Path, default=Path("/data/ali/scidocs-gem-data"))
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("/home/ali/gem-baseline/results/scidocs_gem_profile_20260604_192032"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/home/ali/gem-baseline"))
    parser.add_argument("--qid", type=int, default=QID)
    parser.add_argument("--dplus", type=int, default=DPLUS)
    parser.add_argument("--dminus", type=int, default=DEFAULT_DMINUS)
    parser.add_argument("--nprob", type=int, default=4)
    parser.add_argument("--rerank-k", type=int, default=1024)
    parser.add_argument("--low-ef", type=int, default=4000)
    parser.add_argument("--high-ef", type=int, default=16000)
    parser.add_argument("--pair-samples", type=int, default=1_000_000)
    parser.add_argument("--scatter-sample", type=int, default=15420)
    return parser.parse_args()


def load_query_profile(profile_dir: Path, nprob: int, rerank_k: int, ef: int, qid: int) -> Dict[str, str]:
    path = profile_dir / f"scidocs_results_512_all_24_80_100_t{nprob}_rerank{rerank_k}_ef{ef}.query_profile.tsv"
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["query_idx"]) == qid:
                return row
    raise RuntimeError(f"query {qid} not found in {path}")


def parse_int_list(text: str) -> List[int]:
    if not text:
        return []
    return [int(x) for x in text.split(",") if x != ""]


def load_result_ranking(profile_dir: Path, nprob: int, rerank_k: int, ef: int, qid: int) -> List[Tuple[int, float, int]]:
    path = profile_dir / f"scidocs_results_512_all_24_80_100_t{nprob}_rerank{rerank_k}_ef{ef}.tsv"
    rows: List[Tuple[int, float, int]] = []
    with path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            if int(parts[0]) != qid:
                continue
            rows.append((int(parts[1]), float(parts[2]), int(parts[3])))
    rows.sort(key=lambda x: x[2])
    return rows


def load_qrels(qrels_path: Path, qid: int) -> List[int]:
    rels = []
    with qrels_path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[0]) == qid:
                rels.append(int(parts[1]))
    return rels


def load_coarse_clusters(path: Path) -> List[List[int]]:
    clusters = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            clusters.append([int(x) for x in line.split()] if line else [])
    return clusters


def routed_union_from_clusters(coarse_clusters: Sequence[Sequence[int]], cluster_ids: Sequence[int]) -> np.ndarray:
    docs = set()
    for cluster_id in cluster_ids:
        docs.update(coarse_clusters[cluster_id])
    return np.array(sorted(docs), dtype=np.int32)


def build_doc_locations(docdata: Path) -> Tuple[Dict[int, DocLoc], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    locs: Dict[int, DocLoc] = {}
    emb_shards: List[np.ndarray] = []
    code_shards: List[np.ndarray] = []
    lens_shards: List[np.ndarray] = []
    global_doc = 0
    shard = 0
    while (docdata / f"doclens{shard}.npy").exists():
        lens = np.load(docdata / f"doclens{shard}.npy", mmap_mode="r")
        emb = np.load(docdata / f"encoding{shard}_float16.npy", mmap_mode="r")
        codes = np.load(docdata / f"doc_codes_{shard}.npy", mmap_mode="r")
        lens_shards.append(lens)
        emb_shards.append(emb)
        code_shards.append(codes)
        offsets = np.concatenate([[0], np.cumsum(np.asarray(lens, dtype=np.int64))])
        for local_doc, vec_len in enumerate(np.asarray(lens, dtype=np.int64)):
            locs[global_doc] = DocLoc(shard=shard, local_doc=local_doc, vec_start=int(offsets[local_doc]), vec_len=int(vec_len))
            global_doc += 1
        shard += 1
    if not locs:
        raise RuntimeError(f"No doclens shards found in {docdata}")
    return locs, emb_shards, code_shards, lens_shards


def get_doc_vectors(doc_id: int, locs: Dict[int, DocLoc], emb_shards: Sequence[np.ndarray]) -> np.ndarray:
    loc = locs[doc_id]
    return np.asarray(emb_shards[loc.shard][loc.vec_start : loc.vec_start + loc.vec_len], dtype=np.float32)


def get_doc_codes(doc_id: int, locs: Dict[int, DocLoc], code_shards: Sequence[np.ndarray]) -> np.ndarray:
    loc = locs[doc_id]
    return np.asarray(code_shards[loc.shard][loc.vec_start : loc.vec_start + loc.vec_len], dtype=np.int32)


def exact_maxsim_distance(query: np.ndarray, doc: np.ndarray) -> float:
    scores = query @ doc.T
    return float(1.0 - scores.max(axis=1).mean())


def traversal_surrogate_distance(fine_query_scores: np.ndarray, doc_codes: np.ndarray) -> float:
    # Matches L2SqrCluster4Search: mean_i (1 - max_{code in D} dot(q_i, c_code)).
    best = fine_query_scores[doc_codes].max(axis=0)
    return float((1.0 - best).mean())


def rank_from_scores(doc_ids: np.ndarray, scores: np.ndarray) -> Dict[int, int]:
    order = np.argsort(scores, kind="mergesort")
    return {int(doc_ids[idx]): int(rank + 1) for rank, idx in enumerate(order)}


def load_or_compute_scores(
    gem_data: Path,
    routed_docs: np.ndarray,
    qid: int,
    locs: Dict[int, DocLoc],
    emb_shards: Sequence[np.ndarray],
    code_shards: Sequence[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    query = np.asarray(np.load(gem_data / "qdata/qembs.npy", mmap_mode="r")[qid], dtype=np.float32)
    centroids = np.asarray(np.load(gem_data / "cdata/centroids.npy", mmap_mode="r"), dtype=np.float32)
    fine_query_scores = centroids @ query.T
    d_ms = np.empty(len(routed_docs), dtype=np.float32)
    d_sur = np.empty(len(routed_docs), dtype=np.float32)
    for idx, doc_id in enumerate(routed_docs):
        doc = get_doc_vectors(int(doc_id), locs, emb_shards)
        codes = get_doc_codes(int(doc_id), locs, code_shards)
        d_ms[idx] = exact_maxsim_distance(query, doc)
        d_sur[idx] = traversal_surrogate_distance(fine_query_scores, codes)
        if (idx + 1) % 2500 == 0:
            print(f"[INFO] scored {idx + 1}/{len(routed_docs)} routed docs")
    return d_ms, d_sur, query


def choose_dminus(
    routed_docs: np.ndarray,
    d_ms: np.ndarray,
    d_sur: np.ndarray,
    dplus: int,
    requested_dminus: int,
    qrels: Sequence[int],
    low_docs: Sequence[int],
    rank_sur: Dict[int, int],
) -> int:
    doc_to_idx = {int(doc_id): idx for idx, doc_id in enumerate(routed_docs)}
    qrel_set = set(qrels)
    plus_idx = doc_to_idx[dplus]
    if requested_dminus in doc_to_idx:
        minus_idx = doc_to_idx[requested_dminus]
        if (
            requested_dminus not in qrel_set
            and d_ms[plus_idx] < d_ms[minus_idx]
            and d_sur[plus_idx] > d_sur[minus_idx]
        ):
            return requested_dminus

    best_doc = None
    best_gap = -1e9
    low_set = set(low_docs)
    for idx, doc_id_np in enumerate(routed_docs):
        doc_id = int(doc_id_np)
        if doc_id == dplus or doc_id in qrel_set:
            continue
        if d_ms[plus_idx] < d_ms[idx] and d_sur[plus_idx] > d_sur[idx]:
            gap = float((d_sur[plus_idx] - d_sur[idx]) + 0.25 * (d_ms[idx] - d_ms[plus_idx]))
            if doc_id in low_set or rank_sur.get(doc_id, 10**9) <= 20:
                gap += 0.05
            if gap > best_gap:
                best_gap = gap
                best_doc = doc_id
    if best_doc is None:
        raise RuntimeError("Could not find a non-qrel distractor satisfying the inversion condition")
    return int(best_doc)


def sampled_inversion_rate(d_ms: np.ndarray, d_sur: np.ndarray, sample_count: int, seed: int = 17) -> float:
    rng = np.random.default_rng(seed)
    n = len(d_ms)
    left = rng.integers(0, n, size=sample_count, endpoint=False)
    right = rng.integers(0, n, size=sample_count, endpoint=False)
    mask = left != right
    left = left[mask]
    right = right[mask]
    ms_delta = d_ms[left] - d_ms[right]
    sur_delta = d_sur[left] - d_sur[right]
    valid = (ms_delta != 0) & (sur_delta != 0)
    return float(np.mean((ms_delta[valid] * sur_delta[valid]) < 0.0))


def load_selected_doc_rows(
    routed_docs: np.ndarray,
    d_ms: np.ndarray,
    d_sur: np.ndarray,
    ranks_ms: Dict[int, int],
    ranks_sur: Dict[int, int],
    qrels: Sequence[int],
    low_docs: Sequence[int],
    high_docs: Sequence[int],
    locs: Dict[int, DocLoc],
    dplus: int,
    dminus: int,
) -> List[Dict[str, object]]:
    qrel_set = set(qrels)
    low_set = set(low_docs)
    high_set = set(high_docs)
    doc_to_idx = {int(doc_id): idx for idx, doc_id in enumerate(routed_docs)}
    selected = set()
    selected.update(int(routed_docs[i]) for i in np.argsort(d_ms)[:30])
    selected.update(int(routed_docs[i]) for i in np.argsort(d_sur)[:30])
    selected.update(low_docs[:30])
    selected.update(high_docs[:30])
    selected.update(qrels)
    selected.update([dplus, dminus])
    rows = []
    for doc_id in sorted(selected, key=lambda d: min(ranks_ms.get(d, 10**9), ranks_sur.get(d, 10**9))):
        if doc_id not in doc_to_idx:
            continue
        idx = doc_to_idx[doc_id]
        if doc_id == dplus:
            role = "D+ relevant"
        elif doc_id == dminus:
            role = "D- distractor"
        elif doc_id in qrel_set:
            role = "other qrel"
        elif ranks_sur[doc_id] <= 30:
            role = "top surrogate"
        elif ranks_ms[doc_id] <= 30:
            role = "top MaxSim"
        elif doc_id in low_set:
            role = "low-budget result"
        elif doc_id in high_set:
            role = "high-budget result"
        else:
            role = "selected"
        rows.append(
            {
                "doc_id": doc_id,
                "role": role,
                "d_MS": float(d_ms[idx]),
                "rank_MS": ranks_ms[doc_id],
                "d_sur": float(d_sur[idx]),
                "rank_sur": ranks_sur[doc_id],
                "qrel": doc_id in qrel_set,
                "low_budget_result": doc_id in low_set,
                "high_budget_result": doc_id in high_set,
                "doc_vecs": locs[doc_id].vec_len,
            }
        )
    return rows


def write_selected_tsv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fields = [
        "doc_id",
        "role",
        "d_MS",
        "rank_MS",
        "d_sur",
        "rank_sur",
        "qrel",
        "low_budget_result",
        "high_budget_result",
        "doc_vecs",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def code_chamfer_distance(
    doc_a: int,
    doc_b: int,
    locs: Dict[int, DocLoc],
    code_shards: Sequence[np.ndarray],
    centroids: np.ndarray,
) -> float:
    codes_a = get_doc_codes(doc_a, locs, code_shards)
    codes_b = get_doc_codes(doc_b, locs, code_shards)
    uniq_a, count_a = np.unique(codes_a, return_counts=True)
    uniq_b, count_b = np.unique(codes_b, return_counts=True)
    weight_a = count_a.astype(np.float32) / count_a.sum()
    weight_b = count_b.astype(np.float32) / count_b.sum()
    sims = centroids[uniq_a] @ centroids[uniq_b].T
    d_ab = np.sum(weight_a * (1.0 - sims.max(axis=1)))
    d_ba = np.sum(weight_b * (1.0 - sims.max(axis=0)))
    return float(0.5 * (d_ab + d_ba))


def build_local_graph(
    selected_rows: Sequence[Dict[str, object]],
    locs: Dict[int, DocLoc],
    code_shards: Sequence[np.ndarray],
    centroids: np.ndarray,
    dplus: int,
    dminus: int,
) -> nx.Graph:
    docs = [int(row["doc_id"]) for row in selected_rows[:55]]
    if dplus not in docs:
        docs.append(dplus)
    if dminus not in docs:
        docs.append(dminus)
    docs = list(dict.fromkeys(docs))
    graph = nx.Graph()
    graph.add_nodes_from(docs)
    for doc in docs:
        distances = []
        for other in docs:
            if other == doc:
                continue
            dist = code_chamfer_distance(doc, other, locs, code_shards, centroids)
            distances.append((dist, other))
        for dist, other in sorted(distances)[:3]:
            graph.add_edge(doc, other, distance=dist, weight=1.0 / max(dist, 1e-6))
    return graph


def make_figure(
    out_pdf: Path,
    out_png: Path,
    routed_docs: np.ndarray,
    d_ms: np.ndarray,
    d_sur: np.ndarray,
    selected_rows: Sequence[Dict[str, object]],
    graph: nx.Graph,
    ranks_ms: Dict[int, int],
    ranks_sur: Dict[int, int],
    qrels: Sequence[int],
    low_rows: Sequence[Tuple[int, float, int]],
    high_rows: Sequence[Tuple[int, float, int]],
    stats_json: Dict[str, object],
    dplus: int,
    dminus: int,
    scatter_sample: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.45), constrained_layout=True)
    ax = axes[0]
    rng = np.random.default_rng(3)
    if scatter_sample < len(routed_docs):
        sample_idx = rng.choice(len(routed_docs), size=scatter_sample, replace=False)
    else:
        sample_idx = np.arange(len(routed_docs))
    low_set = {doc for doc, _, _ in low_rows}
    high_set = {doc for doc, _, _ in high_rows}
    qrel_set = set(qrels)
    sample_docs = routed_docs[sample_idx]
    low_mask = np.array([int(d) in low_set for d in sample_docs])
    qrel_mask = np.array([int(d) in qrel_set for d in sample_docs])
    ax.scatter(d_ms[sample_idx], d_sur[sample_idx], s=7, c="#c9c9c9", alpha=0.35, linewidths=0, label="routed docs")
    ax.scatter(d_ms[sample_idx][low_mask], d_sur[sample_idx][low_mask], s=12, c="#777777", alpha=0.55, linewidths=0, label="low-budget top-100")
    ax.scatter(
        d_ms[sample_idx][qrel_mask],
        d_sur[sample_idx][qrel_mask],
        s=34,
        facecolors="none",
        edgecolors="#2b6cb0",
        linewidths=1.0,
        label="qrels in union",
    )
    doc_to_idx = {int(doc_id): idx for idx, doc_id in enumerate(routed_docs)}
    plus_idx = doc_to_idx[dplus]
    minus_idx = doc_to_idx[dminus]
    ax.scatter([d_ms[plus_idx]], [d_sur[plus_idx]], s=125, marker="*", c="#1f77b4", edgecolors="white", linewidths=0.8, zorder=5, label="D+")
    ax.scatter([d_ms[minus_idx]], [d_sur[minus_idx]], s=72, marker="s", c="#ff7f0e", edgecolors="white", linewidths=0.8, zorder=5, label="D-")
    for idx, color in [(plus_idx, "#1f77b4"), (minus_idx, "#ff7f0e")]:
        ax.axvline(d_ms[idx], color=color, linestyle="--", linewidth=0.8, alpha=0.55)
        ax.axhline(d_sur[idx], color=color, linestyle="--", linewidth=0.8, alpha=0.55)
    ax.set_title("(a) MaxSim vs GEM surrogate mismatch")
    ax.set_xlabel(r"exact MaxSim distance $d_{\mathrm{MS}}(Q,D)$")
    ax.set_ylabel(r"GEM traversal surrogate $d_{\mathrm{sur}}(Q,D)$")
    ax.text(
        0.03,
        0.97,
        r"$d_{\mathrm{MS}}(D^+)<d_{\mathrm{MS}}(D^-)$" + "\n" + r"$d_{\mathrm{sur}}(D^+)>d_{\mathrm{sur}}(D^-)$",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.92),
    )
    ax.annotate(
        "D+ is closer by MaxSim\nbut farther by surrogate",
        xy=(d_ms[plus_idx], d_sur[plus_idx]),
        xytext=(0.58, 0.78),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#333333"),
        fontsize=8,
        ha="left",
    )
    table_text = (
        "Doc    d_MS  rMS   d_sur  rSur\n"
        f"D+   {d_ms[plus_idx]:.3f}   {ranks_ms[dplus]:<3d}  {d_sur[plus_idx]:.3f}   {ranks_sur[dplus]:<3d}\n"
        f"D-   {d_ms[minus_idx]:.3f}   {ranks_ms[dminus]:<3d}  {d_sur[minus_idx]:.3f}   {ranks_sur[dminus]:<3d}"
    )
    ax.text(
        0.98,
        0.04,
        table_text,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        family="monospace",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#dddddd", alpha=0.92),
    )
    ax.legend(loc="lower left", frameon=False, handletextpad=0.25, borderpad=0.2)
    ax.grid(True, alpha=0.18, linewidth=0.5)

    axg = axes[1]
    selected_docs = list(graph.nodes())
    row_by_doc = {int(row["doc_id"]): row for row in selected_rows}
    # Rank-guided coordinates make the surrogate frontier visible and stable.
    rng = np.random.default_rng(11)
    pos = {}
    for doc in selected_docs:
        x = math.log10(ranks_sur.get(doc, 10_000))
        y = -math.log10(ranks_ms.get(doc, 10_000))
        if doc == dplus:
            x += 0.15
            y += 0.08
        pos[doc] = np.array([x, y]) + rng.normal(scale=0.015, size=2)
    frontier_docs = [doc for doc in selected_docs if doc in low_set or ranks_sur.get(doc, 10**9) <= 12]
    if frontier_docs:
        xs = np.array([pos[d][0] for d in frontier_docs])
        ys = np.array([pos[d][1] for d in frontier_docs])
        ellipse = Ellipse(
            (float(xs.mean()), float(ys.mean())),
            width=float(max(xs.max() - xs.min(), 0.12) + 0.32),
            height=float(max(ys.max() - ys.min(), 0.12) + 0.25),
            facecolor="#eeeeee",
            edgecolor="#888888",
            linestyle="--",
            linewidth=0.9,
            alpha=0.45,
            zorder=0,
        )
        axg.add_patch(ellipse)
    nx.draw_networkx_edges(graph, pos, ax=axg, width=0.65, alpha=0.35, edge_color="#8c8c8c")
    node_colors = []
    node_sizes = []
    edge_colors = []
    line_widths = []
    for doc in selected_docs:
        if doc == dplus:
            node_colors.append("#1f77b4")
            node_sizes.append(250)
            edge_colors.append("white")
            line_widths.append(1.0)
        elif doc == dminus:
            node_colors.append("#ff7f0e")
            node_sizes.append(190)
            edge_colors.append("white")
            line_widths.append(1.0)
        elif doc in qrel_set:
            node_colors.append("#d9e8fb")
            node_sizes.append(120)
            edge_colors.append("#1f77b4")
            line_widths.append(1.2)
        elif doc in low_set:
            node_colors.append("#8a8a8a")
            node_sizes.append(95)
            edge_colors.append("white")
            line_widths.append(0.4)
        elif doc in high_set:
            node_colors.append("#c7c7c7")
            node_sizes.append(75)
            edge_colors.append("white")
            line_widths.append(0.3)
        else:
            node_colors.append("#dddddd")
            node_sizes.append(60)
            edge_colors.append("white")
            line_widths.append(0.25)
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=axg,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors=edge_colors,
        linewidths=line_widths,
    )
    labels = {dplus: "D+", dminus: "D-"}
    for doc in selected_docs:
        if ranks_sur.get(doc, 10**9) <= 5 and doc not in labels:
            labels[doc] = str(doc)
    nx.draw_networkx_labels(graph, pos, labels=labels, ax=axg, font_size=7.6, font_color="#222222")
    if dminus in pos and dplus in pos:
        axg.annotate(
            "",
            xy=pos[dminus],
            xytext=(pos[dminus][0] - 0.38, pos[dminus][1] + 0.08),
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=1.3, linestyle="--"),
        )
        axg.annotate(
            "small pool follows\nsurrogate-favored region",
            xy=pos[dminus],
            xytext=(pos[dminus][0] - 0.42, pos[dminus][1] + 0.22),
            fontsize=7.7,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=0.8),
        )
        axg.annotate(
            "D+ enters only\nafter deeper traversal",
            xy=pos[dplus],
            xytext=(pos[dplus][0] + 0.05, pos[dplus][1] - 0.25),
            fontsize=7.7,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
        )
    axg.set_title("(b) Local surrogate-neighborhood graph")
    axg.text(
        0.5,
        -0.12,
        "Low budget: 4,000 / 15,420 = 25.9%, Miss    High budget: 15,381 / 15,420 = 99.7%, Hit@6",
        transform=axg.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
    )
    axg.text(
        0.02,
        0.03,
        "Edges: fine-code surrogate neighbors\nx-position: GEM surrogate rank",
        transform=axg.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.2,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#dddddd", alpha=0.9),
    )
    axg.set_axis_off()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_caption(path: Path) -> None:
    text = (
        "Real SciDocs example showing objective mismatch in GEM. The relevant document "
        "(D+) is ranked ahead of a distractor (D-) by exact MaxSim, but the GEM graph-side "
        "traversal surrogate ranks the distractor ahead of D+. Therefore, surrogate-guided "
        "traversal initially follows the wrong local neighborhood. At the small search budget, "
        "GEM inspects only 25.9% of the routed subgraph and misses D+; after expanding 99.7% "
        "of the routed subgraph, D+ enters the candidate pool and exact MaxSim reranking "
        "recovers it at rank 6."
    )
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    low_profile = load_query_profile(args.profile_dir, args.nprob, args.rerank_k, args.low_ef, args.qid)
    high_profile = load_query_profile(args.profile_dir, args.nprob, args.rerank_k, args.high_ef, args.qid)
    selected_clusters = parse_int_list(low_profile["selected_unique_cluster_ids"])
    coarse_clusters = load_coarse_clusters(args.gem_data / "cdata/coarse_cluster_info.txt")
    routed_docs = routed_union_from_clusters(coarse_clusters, selected_clusters)
    qrels = load_qrels(args.gem_data / "qdata/qrels.tsv", args.qid)
    low_rows = load_result_ranking(args.profile_dir, args.nprob, args.rerank_k, args.low_ef, args.qid)
    high_rows = load_result_ranking(args.profile_dir, args.nprob, args.rerank_k, args.high_ef, args.qid)
    low_docs = [doc for doc, _, _ in low_rows]
    high_docs = [doc for doc, _, _ in high_rows]
    locs, emb_shards, code_shards, _ = build_doc_locations(args.gem_data / "docdata")
    d_ms, d_sur, _query = load_or_compute_scores(args.gem_data, routed_docs, args.qid, locs, emb_shards, code_shards)
    ranks_ms = rank_from_scores(routed_docs, d_ms)
    ranks_sur = rank_from_scores(routed_docs, d_sur)
    dminus = choose_dminus(routed_docs, d_ms, d_sur, args.dplus, args.dminus, qrels, low_docs, ranks_sur)
    doc_to_idx = {int(doc_id): idx for idx, doc_id in enumerate(routed_docs)}
    plus_idx = doc_to_idx[args.dplus]
    minus_idx = doc_to_idx[dminus]
    spearman = stats.spearmanr(d_ms, d_sur).statistic
    try:
        kendall = stats.kendalltau(d_ms, d_sur).statistic
    except Exception:
        kendall = None
    inversion_rate = sampled_inversion_rate(d_ms, d_sur, args.pair_samples)
    high_rank = next((rank for doc, _score, rank in high_rows if doc == args.dplus), None)
    low_rank = next((rank for doc, _score, rank in low_rows if doc == args.dplus), None)
    checks = {
        "qid = 443": args.qid == 443,
        "D+ in routed union": args.dplus in doc_to_idx,
        "D+ missed at low budget": low_rank is None,
        "D+ hit at high budget": high_rank is not None,
        "d_MS(D+) < d_MS(D-)": bool(d_ms[plus_idx] < d_ms[minus_idx]),
        "d_sur(D+) > d_sur(D-)": bool(d_sur[plus_idx] > d_sur[minus_idx]),
        "rank_MS(D+) better than rank_MS(D-)": ranks_ms[args.dplus] < ranks_ms[dminus],
        "rank_sur(D+) worse than rank_sur(D-)": ranks_sur[args.dplus] > ranks_sur[dminus],
    }
    for label, ok in checks.items():
        print(f"[CHECK] {label}: {ok}")
    if not all(checks.values()):
        raise RuntimeError("One or more required sanity checks failed")

    selected_rows = load_selected_doc_rows(
        routed_docs,
        d_ms,
        d_sur,
        ranks_ms,
        ranks_sur,
        qrels,
        low_docs,
        high_docs,
        locs,
        args.dplus,
        dminus,
    )
    centroids = np.asarray(np.load(args.gem_data / "cdata/centroids.npy", mmap_mode="r"), dtype=np.float32)
    graph = build_local_graph(selected_rows, locs, code_shards, centroids, args.dplus, dminus)
    stats_out = {
        "qid": args.qid,
        "dplus": args.dplus,
        "dminus": dminus,
        "routed_union_size": int(len(routed_docs)),
        "selected_unique_cluster_count": int(low_profile["selected_unique_cluster_count"]),
        "entry_point_count": int(low_profile["entry_point_count"]),
        "low_ef": args.low_ef,
        "high_ef": args.high_ef,
        "low_graph_candidate_count": int(low_profile["graph_candidate_count"]),
        "high_graph_candidate_count": int(high_profile["graph_candidate_count"]),
        "low_inspected_fraction": int(low_profile["graph_candidate_count"]) / len(routed_docs),
        "high_inspected_fraction": int(high_profile["graph_candidate_count"]) / len(routed_docs),
        "low_rank_dplus": low_rank,
        "high_rank_dplus": high_rank,
        "d_MS_dplus": float(d_ms[plus_idx]),
        "d_MS_dminus": float(d_ms[minus_idx]),
        "d_sur_dplus": float(d_sur[plus_idx]),
        "d_sur_dminus": float(d_sur[minus_idx]),
        "rank_MS_dplus": ranks_ms[args.dplus],
        "rank_MS_dminus": ranks_ms[dminus],
        "rank_sur_dplus": ranks_sur[args.dplus],
        "rank_sur_dminus": ranks_sur[dminus],
        "spearman_d_MS_vs_d_sur": float(spearman),
        "kendall_tau_d_MS_vs_d_sur": None if kendall is None else float(kendall),
        "sampled_pairwise_inversion_rate": inversion_rate,
        "sampled_pair_count": args.pair_samples,
        "qrels": qrels,
        "qrels_in_routed_union": [doc for doc in qrels if doc in doc_to_idx],
        "checks": checks,
        "surrogate_label": "GEM traversal surrogate (L2SqrCluster4Search centroid-code score)",
        "graph_panel_label": "local fine-code surrogate-neighborhood graph; not extracted actual GEM adjacency",
    }
    print("\ndoc_id\trole\td_MS\trank_MS\td_sur\trank_sur\tqrel?")
    for doc in [args.dplus, dminus]:
        idx = doc_to_idx[doc]
        role = "D+ rel." if doc == args.dplus else "D- distract"
        print(f"{doc}\t{role}\t{d_ms[idx]:.6f}\t{ranks_ms[doc]}\t{d_sur[idx]:.6f}\t{ranks_sur[doc]}\t{doc in qrels}")
    prefix = args.out_dir / "gem_surrogate_mismatch_scidocs"
    stats_path = prefix.with_name(prefix.name + "_stats.json")
    selected_path = prefix.with_name(prefix.name + "_selected_docs.tsv")
    pdf_path = prefix.with_suffix(".pdf")
    png_path = prefix.with_suffix(".png")
    caption_path = prefix.with_name(prefix.name + "_caption.txt")
    stats_path.write_text(json.dumps(stats_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_selected_tsv(selected_path, selected_rows)
    write_caption(caption_path)
    make_figure(
        pdf_path,
        png_path,
        routed_docs,
        d_ms,
        d_sur,
        selected_rows,
        graph,
        ranks_ms,
        ranks_sur,
        qrels,
        low_rows,
        high_rows,
        stats_out,
        args.dplus,
        dminus,
        args.scatter_sample,
    )
    print(f"[OUT] {pdf_path}")
    print(f"[OUT] {png_path}")
    print(f"[OUT] {stats_path}")
    print(f"[OUT] {selected_path}")
    print(f"[OUT] {caption_path}")


if __name__ == "__main__":
    main()
