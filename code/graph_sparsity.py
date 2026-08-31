"""Graph construction helpers for the frozen Applied Energy protocol.

Structural graphs are kept intact. Data-driven graph sparsity is selected on
validation data with source-specific controls: thresholds for static and
dynamic similarity, and per-node top-k for the adaptive graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


GRAPH_DESIGN_PROTOCOL_VERSION = "ae_graph_design_v5_accuracy_selected_independent_fusion"
FINAL_GRAPH_SOURCE_POLICY = "source_specific_threshold_topk_v2"
STATIC_THRESHOLD_CANDIDATES = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
DYNAMIC_THRESHOLD_CANDIDATES = STATIC_THRESHOLD_CANDIDATES
ADAPTIVE_TOPK_CANDIDATES = (0, 1, 2, 4, 8, 16, 32, 64, 96, 128)


def adaptive_topk_candidates(node_count: int) -> tuple[int, ...]:
    """Return the formal top-k grid, clipped per dataset and including N-1."""
    if int(node_count) < 1:
        raise ValueError("node_count must be positive")
    max_neighbors = int(node_count) - 1
    return tuple(sorted({
        min(int(value), max_neighbors)
        for value in (*ADAPTIVE_TOPK_CANDIDATES, max_neighbors)
    }))


def _validate_similarity_array(sim: np.ndarray) -> np.ndarray:
    arr = np.asarray(sim, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"sim must be square, got shape={arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("sim contains NaN or Inf values")
    return arr


def build_threshold_similarity_adj(
    sim: np.ndarray,
    threshold: float,
    include_self_loops: bool = True,
    use_weights: bool = True,
) -> np.ndarray:
    """Build a symmetric threshold graph without forced-neighbour coverage."""
    sim_arr = _validate_similarity_array(sim)
    tau = float(threshold)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    scores = 0.5 * (sim_arr + sim_arr.T)
    np.fill_diagonal(scores, 0.0)
    selected = scores >= tau
    adj = np.where(selected, scores if use_weights else 1.0, 0.0).astype(np.float32)
    if include_self_loops:
        np.fill_diagonal(adj, 1.0)
    return adj


def build_threshold_similarity_adj_torch(
    sim: torch.Tensor,
    threshold: float,
    include_self_loops: bool = True,
    use_weights: bool = True,
) -> torch.Tensor:
    """Torch threshold builder for one or a batch of similarity matrices."""
    squeeze_batch = sim.ndim == 2
    matrices = sim.unsqueeze(0) if squeeze_batch else sim
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"sim must be square or batched square, got shape={tuple(sim.shape)}")
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError("sim contains NaN or Inf values")
    tau = float(threshold)
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold}")
    scores = 0.5 * (matrices + matrices.transpose(-1, -2))
    diagonal = torch.arange(scores.shape[-1], device=scores.device)
    scores = scores.clone()
    scores[:, diagonal, diagonal] = 0.0
    if use_weights:
        adj = torch.where(scores >= tau, scores, torch.zeros_like(scores))
    else:
        adj = (scores >= tau).to(dtype=scores.dtype)
    if include_self_loops:
        adj[:, diagonal, diagonal] = 1.0
    return adj.squeeze(0) if squeeze_batch else adj


def build_topk_similarity_adj_torch(
    sim: torch.Tensor,
    top_k: int,
    include_self_loops: bool = True,
    use_weights: bool = True,
) -> torch.Tensor:
    """Keep each node's strongest positive candidates, then symmetrize by max."""
    squeeze_batch = sim.ndim == 2
    matrices = sim.unsqueeze(0) if squeeze_batch else sim
    if matrices.ndim != 3 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError(f"sim must be square or batched square, got shape={tuple(sim.shape)}")
    if not bool(torch.isfinite(matrices).all()):
        raise ValueError("sim contains NaN or Inf values")
    n = int(matrices.shape[-1])
    k = int(top_k)
    if k < 0:
        raise ValueError(f"top_k must be nonnegative, got {top_k}")
    k = min(k, max(0, n - 1))
    scores = torch.relu(matrices).clone()
    diagonal = torch.arange(n, device=scores.device)
    scores[:, diagonal, diagonal] = -torch.inf
    directed = torch.zeros_like(scores)
    if k > 0:
        values, indices = torch.topk(scores, k=k, dim=-1)
        selected = values if use_weights else torch.ones_like(values)
        selected = torch.where(values > 0.0, selected, torch.zeros_like(selected))
        directed = directed.scatter(-1, indices, selected)
    adj = torch.maximum(directed, directed.transpose(-1, -2))
    if include_self_loops:
        adj[:, diagonal, diagonal] = 1.0
    return adj.squeeze(0) if squeeze_batch else adj


def graph_edge_diagnostics(
    adj: np.ndarray,
    hierarchy_adj: np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize undirected edges, degree, isolation, density and H overlap."""
    arr = np.asarray(adj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"adj must be square, got shape={arr.shape}")
    mask = np.abs(arr) > 1e-12
    np.fill_diagonal(mask, False)
    upper = np.triu(mask | mask.T, k=1)
    degree = upper.sum(axis=0) + upper.sum(axis=1)
    overlap = None
    if hierarchy_adj is not None:
        hierarchy = np.asarray(hierarchy_adj)
        if hierarchy.shape != arr.shape:
            raise ValueError(f"hierarchy_adj shape={hierarchy.shape} != adj shape={arr.shape}")
        hmask = np.abs(hierarchy) > 1e-12
        np.fill_diagonal(hmask, False)
        overlap = int(np.count_nonzero(upper & np.triu(hmask | hmask.T, k=1)))
    return {
        "undirected_edge_count": int(np.count_nonzero(upper)),
        "offdiag_density": offdiag_density(arr),
        "degree_min": int(degree.min()) if degree.size else 0,
        "degree_mean": float(degree.mean()) if degree.size else 0.0,
        "degree_max": int(degree.max()) if degree.size else 0,
        "isolated_node_count": int(np.count_nonzero(degree == 0)),
        "hierarchy_overlap_edge_count": overlap,
    }


def offdiag_density(adj: Any, threshold: float = 1e-12) -> float:
    """Return directed off-diagonal density for a dense adjacency matrix."""
    arr = np.asarray(adj, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"adj must be square, got shape={arr.shape}")
    n = int(arr.shape[0])
    if n <= 1:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    edge_count = int(np.count_nonzero(np.abs(arr[mask]) > threshold))
    return edge_count / float(n * (n - 1))


def compute_similarity_numpy(
    data: np.ndarray,
    sim_type: str = "cosine",
    use_abs: bool = True,
) -> np.ndarray:
    """Compute node-node similarity from data shaped ``[T, N]``."""
    values = np.asarray(data, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"data must have shape [T, N], got {values.shape}")
    sim_type = str(sim_type).lower()
    if sim_type == "pearson":
        sim = np.corrcoef(values.T)
    elif sim_type == "cosine":
        x = values.T
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-8, norms)
        sim = (x / norms) @ (x / norms).T
        sim = np.clip(sim, -1.0, 1.0)
    else:
        raise ValueError(f"Unsupported sim_type={sim_type!r}; use pearson or cosine.")
    if use_abs:
        sim = np.abs(sim)
    sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)
    np.fill_diagonal(sim, 1.0)
    return sim.astype(np.float32)
