"""Graph-source construction used by LAGTCN."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch


GRAPH_DESIGN_PROTOCOL_VERSION = "ae_graph_design_v5_accuracy_selected_independent_fusion"
FINAL_GRAPH_SOURCE_POLICY = "source_specific_threshold_topk_v2"
INFORMATIVE_GRAPH_SOURCES = ("hierarchy", "similarity", "adaptive", "dynamic")
VALID_GRAPH_MODES = {
    "I",
    "H",
    "HG",
    "S",
    "A",
    "D",
    "H+S",
    "H+A",
    "H+D",
    "S+A+D",
    "H+S+A+D",
}


def normalize_graph_mode(graph_mode: str) -> str:
    """Normalize graph aliases and impose the canonical H/S/A/D order."""
    value = str(graph_mode).replace(" ", "").upper()
    aliases = {
        "NO": "I",
        "NONE": "I",
        "IDENTITY": "I",
        "HIERARCHY": "H",
        "SIMILARITY": "S",
        "STATIC": "S",
        "ADAPTIVE": "A",
        "DYNAMIC": "D",
    }
    if "+" not in value:
        value = aliases.get(value, value)
    else:
        tokens = {aliases.get(token, token) for token in value.split("+") if token}
        value = "+".join(token for token in ("H", "S", "A", "D") if token in tokens)
    if value not in VALID_GRAPH_MODES:
        raise ValueError(
            f"unsupported graph_mode={graph_mode!r}; choose from {sorted(VALID_GRAPH_MODES)}"
        )
    return value


def graph_sources(graph_mode: str) -> tuple[str, ...]:
    """Return the independent source matrices consumed by LAGTCN."""
    mode = normalize_graph_mode(graph_mode)
    if mode == "I":
        return ("identity",)
    if mode == "HG":
        return ("hierarchy",)
    mapping = {"H": "hierarchy", "S": "similarity", "A": "adaptive", "D": "dynamic"}
    return tuple(mapping[token] for token in ("H", "S", "A", "D") if token in mode.split("+"))


def _validate_similarity_numpy(similarity: np.ndarray) -> np.ndarray:
    values = np.asarray(similarity, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"similarity matrix must be square, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("similarity matrix contains NaN or Inf")
    return values


def build_threshold_similarity_adj(
    similarity: np.ndarray,
    threshold: float,
    include_self_loops: bool = True,
    use_weights: bool = True,
) -> np.ndarray:
    """Create a symmetric threshold graph from a dense similarity matrix."""
    values = _validate_similarity_numpy(similarity)
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    scores = 0.5 * (values + values.T)
    np.fill_diagonal(scores, 0.0)
    adjacency = np.where(
        scores >= threshold,
        scores if use_weights else 1.0,
        0.0,
    ).astype(np.float32)
    if include_self_loops:
        np.fill_diagonal(adjacency, 1.0)
    return adjacency


def build_threshold_similarity_adj_torch(
    similarity: torch.Tensor,
    threshold: float,
    include_self_loops: bool = True,
    use_weights: bool = True,
) -> torch.Tensor:
    """Torch threshold graph builder for one or a batch of matrices."""
    squeeze = similarity.ndim == 2
    matrices = similarity.unsqueeze(0) if squeeze else similarity
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("similarity must be square or batched square")
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError("similarity contains NaN or Inf")
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie in [0, 1]")
    scores = 0.5 * (matrices + matrices.transpose(-1, -2))
    diagonal = torch.arange(scores.shape[-1], device=scores.device)
    scores = scores.clone()
    scores[:, diagonal, diagonal] = 0.0
    adjacency = (
        torch.where(scores >= threshold, scores, torch.zeros_like(scores))
        if use_weights
        else (scores >= threshold).to(scores.dtype)
    )
    if include_self_loops:
        adjacency[:, diagonal, diagonal] = 1.0
    return adjacency.squeeze(0) if squeeze else adjacency


def build_topk_similarity_adj_torch(
    similarity: torch.Tensor,
    top_k: int,
    include_self_loops: bool = True,
    use_weights: bool = True,
) -> torch.Tensor:
    """Keep each node's strongest positive neighbours and symmetrize by max."""
    squeeze = similarity.ndim == 2
    matrices = similarity.unsqueeze(0) if squeeze else similarity
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("similarity must be square or batched square")
    node_count = matrices.shape[-1]
    top_k = min(int(top_k), max(0, node_count - 1))
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    diagonal = torch.arange(node_count, device=matrices.device)
    scores = torch.relu(matrices).clone()
    scores[:, diagonal, diagonal] = -torch.inf
    directed = torch.zeros_like(scores)
    if top_k:
        values, indices = torch.topk(scores, k=top_k, dim=-1)
        selected = values if use_weights else torch.ones_like(values)
        selected = torch.where(values > 0, selected, torch.zeros_like(selected))
        directed.scatter_(-1, indices, selected)
    adjacency = torch.maximum(directed, directed.transpose(-1, -2))
    if include_self_loops:
        adjacency[:, diagonal, diagonal] = 1.0
    return adjacency.squeeze(0) if squeeze else adjacency


def compute_similarity_numpy(
    values: np.ndarray,
    similarity_type: str = "cosine",
    use_abs: bool = True,
) -> np.ndarray:
    """Compute node similarity from an array shaped ``[time, nodes]``."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"values must have shape [time,nodes], got {values.shape}")
    similarity_type = str(similarity_type).lower()
    if similarity_type == "pearson":
        similarity = np.corrcoef(values.T)
    elif similarity_type == "cosine":
        node_values = values.T
        normalized = node_values / np.maximum(
            np.linalg.norm(node_values, axis=1, keepdims=True), 1e-8
        )
        similarity = normalized @ normalized.T
    else:
        raise ValueError("similarity_type must be 'cosine' or 'pearson'")
    if use_abs:
        similarity = np.abs(similarity)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
    np.fill_diagonal(similarity, 1.0)
    return similarity.astype(np.float32)


def load_hierarchy_paths(hierarchy_csv: str | Path) -> tuple[list[list[str]], list[str]]:
    """Read root-to-leaf paths and derive level-major node order."""
    hierarchy_csv = Path(hierarchy_csv)
    with hierarchy_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"hierarchy file has no header: {hierarchy_csv}")
        levels = [name for name in reader.fieldnames if str(name).strip()]
        paths = []
        for row in reader:
            path = [str(row.get(level, "")).strip() for level in levels]
            if not any(path):
                continue
            if not all(path):
                raise ValueError(f"incomplete hierarchy path: {row}")
            clean_path = []
            for node in path:
                if not clean_path or clean_path[-1] != node:
                    clean_path.append(node)
            if clean_path not in paths:
                paths.append(clean_path)
    if not paths:
        raise ValueError(f"hierarchy file has no paths: {hierarchy_csv}")

    maximum_depth = max(len(path) for path in paths)
    level_nodes: list[list[str]] = [[] for _ in range(maximum_depth)]
    for path in paths:
        for level, node in enumerate(path):
            canonical_level = maximum_depth - 1 if level == len(path) - 1 else level
            if node not in level_nodes[canonical_level]:
                level_nodes[canonical_level].append(node)
    node_order = [node for nodes in level_nodes for node in nodes]
    if len(node_order) != len(set(node_order)):
        raise ValueError("a node occurs at more than one hierarchy level")
    return paths, node_order


def build_structural_adjacencies(
    paths: list[list[str]],
    node_order: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build H and HG as symmetric binary matrices with self-loops."""
    index = {node: position for position, node in enumerate(node_order)}
    hierarchy = np.eye(len(node_order), dtype=np.float32)
    children: dict[str, list[str]] = {}

    def add_edge(matrix: np.ndarray, left: str, right: str) -> None:
        left_index, right_index = index[left], index[right]
        if left_index != right_index:
            matrix[left_index, right_index] = 1.0
            matrix[right_index, left_index] = 1.0

    for path in paths:
        for parent, child in zip(path[:-1], path[1:]):
            add_edge(hierarchy, parent, child)
            children.setdefault(parent, [])
            if child not in children[parent]:
                children[parent].append(child)

    enhanced = hierarchy.copy()
    for siblings in children.values():
        for left, right in combinations(siblings, 2):
            add_edge(enhanced, left, right)
    roots = list(dict.fromkeys(path[0] for path in paths))
    leaves = list(dict.fromkeys(path[-1] for path in paths))
    for root in roots:
        for leaf in leaves:
            add_edge(enhanced, root, leaf)
    return hierarchy, enhanced


def write_structural_graphs(dataset_dir: str | Path) -> tuple[Path, Path]:
    """Validate hierarchy metadata and save ``adj_hierarchy.npy``/``adj_HGNN.npy``."""
    dataset_dir = Path(dataset_dir)
    paths, node_order = load_hierarchy_paths(dataset_dir / "hierarchy.csv")
    metadata = json.loads(
        (dataset_dir / "hierarchy_info.json").read_text(encoding="utf-8")
    )
    declared_order = [str(value) for value in metadata.get("node_order", [])]
    if declared_order and len(declared_order) != len(node_order):
        raise ValueError("hierarchy.csv and hierarchy_info.json disagree on node count")
    sum_matrix = np.genfromtxt(dataset_dir / "sum_matrix.csv", delimiter=",")
    if sum_matrix.ndim == 2:
        sum_matrix = sum_matrix[:, ~np.all(np.isnan(sum_matrix), axis=0)]
    if sum_matrix.shape[0] != len(node_order) or not np.isfinite(sum_matrix).all():
        raise ValueError("sum_matrix.csv is incompatible with hierarchy.csv")
    bottom_count = sum_matrix.shape[1]
    if not np.array_equal(
        sum_matrix[-bottom_count:], np.eye(bottom_count, dtype=sum_matrix.dtype)
    ):
        raise ValueError("sum_matrix bottom block must be an identity matrix")

    node_index = {node: index for index, node in enumerate(node_order)}
    supports = [set(np.flatnonzero(np.abs(row) > 1e-8)) for row in sum_matrix]
    for path in paths:
        for parent, child in zip(path[:-1], path[1:]):
            if not supports[node_index[child]].issubset(supports[node_index[parent]]):
                raise ValueError(
                    f"hierarchy edge {parent!r}--{child!r} contradicts sum_matrix.csv"
                )

    hierarchy, enhanced = build_structural_adjacencies(paths, node_order)
    hierarchy_path = dataset_dir / "adj_hierarchy.npy"
    enhanced_path = dataset_dir / "adj_HGNN.npy"
    np.save(hierarchy_path, hierarchy)
    np.save(enhanced_path, enhanced)
    return hierarchy_path, enhanced_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LAGTCN H and HG graph sources")
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    hierarchy_path, enhanced_path = write_structural_graphs(args.dataset_dir)
    print(f"saved {hierarchy_path} and {enhanced_path}")


if __name__ == "__main__":
    main()
