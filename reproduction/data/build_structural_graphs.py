#!/usr/bin/env python3
"""Build the two fixed structural graph sources used by the AE protocol.

This script intentionally builds only:

* H: undirected parent--child edges;
* HG: H plus same-parent sibling edges and root--leaf edges.

Self-loops are stored in both matrices. Data-driven S, A, and D sources are
constructed inside the training pipeline under the frozen validation protocol;
they must not be precomputed here.
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np


def _clean(value: object) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None


def load_hierarchy_paths(hierarchy_csv: Path) -> tuple[list[list[str]], list[str]]:
    """Return unique root-to-leaf paths and level-major node order."""
    with hierarchy_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Hierarchy file has no header: {hierarchy_csv}")
        level_names = [name for name in reader.fieldnames if _clean(name) is not None]
        paths: list[list[str]] = []
        seen_paths: set[tuple[str, ...]] = set()

        for row in reader:
            path = [_clean(row.get(level)) for level in level_names]
            if all(node is None for node in path):
                continue
            if any(node is None for node in path):
                raise ValueError(
                    f"Incomplete hierarchy path in {hierarchy_csv}: {row}"
                )
            clean_path: list[str] = []
            for node in path:
                assert node is not None
                if not clean_path or clean_path[-1] != node:
                    clean_path.append(node)
            key = tuple(clean_path)
            if key not in seen_paths:
                paths.append(clean_path)
                seen_paths.add(key)

    if not paths:
        raise ValueError(f"Hierarchy file has no paths: {hierarchy_csv}")
    max_depth = max(len(path) for path in paths)
    level_nodes: list[list[str]] = [[] for _ in range(max_depth)]
    for path in paths:
        for level_idx, node in enumerate(path):
            # A leaf on a shorter branch still belongs to the bottom block.
            canonical_level = max_depth - 1 if level_idx == len(path) - 1 else level_idx
            if node not in level_nodes[canonical_level]:
                level_nodes[canonical_level].append(node)
    node_order = [node for nodes in level_nodes for node in nodes]
    if len(node_order) != len(set(node_order)):
        raise ValueError("A node label occurs at more than one hierarchy level.")
    return paths, node_order


def build_structural_adjacencies(
    paths: list[list[str]], node_order: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Build H and HG with symmetric binary edges and self-loops."""
    index = {node: idx for idx, node in enumerate(node_order)}
    n_nodes = len(node_order)
    hierarchy = np.eye(n_nodes, dtype=np.float32)
    children: dict[str, list[str]] = {}

    def add_edge(matrix: np.ndarray, left: str, right: str) -> None:
        i, j = index[left], index[right]
        if i != j:
            matrix[i, j] = matrix[j, i] = 1.0

    for path in paths:
        for parent, child in zip(path[:-1], path[1:]):
            add_edge(hierarchy, parent, child)
            siblings = children.setdefault(parent, [])
            if child not in siblings:
                siblings.append(child)

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


def validate_structural_graphs(
    dataset_dir: Path,
    paths: list[list[str]],
    node_order: list[str],
    hierarchy: np.ndarray,
    enhanced: np.ndarray,
) -> dict[str, object]:
    """Cross-check topology against hierarchy metadata and the summing matrix."""
    info_path = dataset_dir / "hierarchy_info.json"
    sum_path = dataset_dir / "sum_matrix.csv"
    if not info_path.is_file() or not sum_path.is_file():
        raise FileNotFoundError(
            "Structural graph generation requires hierarchy_info.json and sum_matrix.csv."
        )

    info = json.loads(info_path.read_text(encoding="utf-8"))
    metadata_order = [str(value) for value in info.get("node_order", [])]
    n_nodes = len(node_order)
    if int(info.get("num_total_nodes", -1)) != n_nodes or len(metadata_order) != n_nodes:
        raise ValueError("hierarchy.csv and hierarchy_info.json disagree on node count.")

    summing = np.loadtxt(sum_path, delimiter=",", dtype=np.float32)
    if summing.ndim == 1:
        summing = summing.reshape(n_nodes, -1)
    if summing.shape[0] != n_nodes or not np.isfinite(summing).all():
        raise ValueError(
            f"Invalid sum_matrix.csv shape/content: {summing.shape}; expected {n_nodes} rows."
        )
    bottom_start = int(info.get("bottom_start_idx", -1))
    num_bottom = int(info.get("num_bottom_nodes", summing.shape[1]))
    if (
        bottom_start < 1
        or num_bottom != summing.shape[1]
        or bottom_start + num_bottom != n_nodes
        or not np.array_equal(
            summing[bottom_start:], np.eye(num_bottom, dtype=np.float32)
        )
    ):
        raise ValueError("sum_matrix.csv does not contain the declared bottom identity block.")

    expected_shape = (n_nodes, n_nodes)
    for name, matrix in (("H", hierarchy), ("HG", enhanced)):
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f"{name} has invalid shape/content: {matrix.shape}.")
        if not np.array_equal(matrix, matrix.T):
            raise ValueError(f"{name} must be symmetric.")
        if not np.array_equal(np.diag(matrix), np.ones(n_nodes, dtype=np.float32)):
            raise ValueError(f"{name} must contain every self-loop.")
        if not np.isin(matrix, (0.0, 1.0)).all():
            raise ValueError(f"{name} must be binary before runtime normalization.")
    if np.any(hierarchy > enhanced):
        raise ValueError("HG must be a strict edge superset (or equality) of H.")

    index = {node: idx for idx, node in enumerate(node_order)}
    supports = [set(np.flatnonzero(np.abs(row) > 1e-8)) for row in summing]
    for path in paths:
        for parent, child in zip(path[:-1], path[1:]):
            parent_idx, child_idx = index[parent], index[child]
            if hierarchy[parent_idx, child_idx] != 1.0:
                raise ValueError(f"Missing H edge {parent!r}--{child!r}.")
            if not supports[child_idx].issubset(supports[parent_idx]):
                raise ValueError(
                    f"H edge {parent!r}--{child!r} contradicts sum_matrix.csv."
                )

    return {
        "metadata_alignment": (
            "exact_labels" if metadata_order == node_order else "position_and_sum_matrix"
        ),
        "bottom_identity_verified": True,
        "sum_matrix_shape": list(summing.shape),
    }


def write_structural_graphs(dataset_dir: Path) -> dict[str, object]:
    dataset_dir = dataset_dir.resolve()
    paths, node_order = load_hierarchy_paths(dataset_dir / "hierarchy.csv")
    hierarchy, enhanced = build_structural_adjacencies(paths, node_order)
    validation = validate_structural_graphs(
        dataset_dir, paths, node_order, hierarchy, enhanced
    )
    np.save(dataset_dir / "adj_hierarchy.npy", hierarchy)
    np.save(dataset_dir / "adj_HGNN.npy", enhanced)
    return {
        "dataset_dir": str(dataset_dir),
        "node_order": node_order,
        "num_nodes": len(node_order),
        "hierarchy_shape": list(hierarchy.shape),
        "enhanced_shape": list(enhanced.shape),
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build only the current fixed H and HG graph sources."
    )
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    result = write_structural_graphs(args.dataset_dir)
    print(
        f"Saved H and HG for {result['dataset_dir']} "
        f"with {result['num_nodes']} nodes."
    )


if __name__ == "__main__":
    main()
