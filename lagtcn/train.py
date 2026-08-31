import os

import csv
import hashlib
import json
import logging
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse
import argparse

from lagtcn.core.data import LoadDatasetLoader
from lagtcn.core.protocol import is_formal_ae_stage
from lagtcn.core import scaled_error as ae_mase
from lagtcn.core.graphs import (
    GRAPH_DESIGN_PROTOCOL_VERSION,
    FINAL_GRAPH_SOURCE_POLICY,
    build_threshold_similarity_adj,
    compute_similarity_numpy,
    graph_edge_diagnostics,
    offdiag_density,
)
from lagtcn.core.metrics import calculate_level_metrics, compute_coherency_violation
from lagtcn.core.naming import (
    BASE_PRED_FILENAME,
    LAGTCN_GRAPH_SOURCE_VERSION_CURRENT,
    artifact_filename,
    compact_namespace,
    compact_output_namespace,
    compact_timestamp,
    find_artifact,
    graph_components,
    graph_subdir,
    lagtcn_graph_sources,
    normalize_graph_mode,
    short_run_label,
    shorten_existing_run_label,
)
from lagtcn.models.temporal_baselines import (
    DLinearBaseline,
    ITransformerBaseline,
    NHiTSBaseline,
    PatchTSTBaseline,
)
from lagtcn.models.graph_models import (
    DCRNNBaseline,
    LAGTCNBaseline,
    MTGNNBaseline,
)
from lagtcn.core.training import (
    train_model,
    evaluate_model,
    load_best_model_strict,
    benchmark_inference,
    save_predictions,
    plot_loss_curves,
    plot_predictions,
    save_model_info,
)

# =========================
# Logging & Device
# =========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _resolve_device(device_arg: str) -> torch.device:
    device_arg = str(device_arg).strip().lower()
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(device_arg)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
        device_count = torch.cuda.device_count()
        idx = device.index if device.index is not None else 0
        if idx < 0 or idx >= device_count:
            raise RuntimeError(f"Requested CUDA device index {idx}, but only {device_count} device(s) available.")
    return device


def set_seed(seed: int):
    """Reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logging.info(f"Random seed set to {seed}")


def _validate_path_namespace(label_name: str, label_value: str) -> str:
    """Validate a relative output namespace and normalize separators."""
    raw = str(label_value).strip().replace("\\", "/")
    if not raw:
        raise ValueError(f"{label_name} must be a non-empty relative path.")
    parts = raw.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label_name} contains an invalid path segment: {label_value!r}")
    if Path(raw).is_absolute():
        raise ValueError(f"{label_name} must be relative to Data/<dataset>/output: {label_value!r}")
    return "/".join(parts)


def _validate_path_segment(label_name: str, label_value: str) -> str:
    """Validate a single path segment used for run labels and metadata ids."""
    raw = str(label_value).strip()
    if not raw or any(sep and sep in raw for sep in {os.sep, os.altsep, "/", "\\"}):
        raise ValueError(f"{label_name} must be a single non-empty path segment: {label_value!r}")
    if raw in {".", ".."}:
        raise ValueError(f"{label_name} cannot be {raw!r}")
    return raw




def _torch_load_checkpoint_metadata(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_latest_training_checkpoint(base_output_dir: Path, run_label: str) -> Path | None:
    if not base_output_dir.exists():
        return None
    prefix = f"{run_label}_"
    candidates: list[Path] = []
    for run_dir in base_output_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue
        short_checkpoint = run_dir / artifact_filename("last_checkpoint")
        if short_checkpoint.exists():
            candidates.append(short_checkpoint)
        candidates.extend(run_dir.glob("last_checkpoint_*.pth"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _infer_timestamp_from_run_dir(run_dir: Path, run_label: str) -> str | None:
    prefix = f"{run_label}_"
    if run_dir.name.startswith(prefix):
        return run_dir.name[len(prefix):]
    return None


def _resolve_resume_checkpoint(resume_arg: str, base_output_dir: Path, run_label: str) -> Path | None:
    mode = str(resume_arg or "none").strip()
    if mode.lower() in {"", "none", "false", "0", "off"}:
        return None
    if mode.lower() == "auto":
        checkpoint_path = _find_latest_training_checkpoint(base_output_dir, run_label)
        if checkpoint_path is None:
            logging.info("--resume auto requested but no checkpoint found for run_label=%s; starting fresh.", run_label)
            return None
        logging.info("--resume auto found checkpoint: %s", checkpoint_path)
        return checkpoint_path
    checkpoint_path = Path(mode).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _checkpoint_timestamp(checkpoint_path: Path, run_label: str) -> str | None:
    try:
        payload = _torch_load_checkpoint_metadata(checkpoint_path)
    except Exception as exc:
        logging.warning("Could not read checkpoint metadata from %s: %s", checkpoint_path, exc)
        return _infer_timestamp_from_run_dir(checkpoint_path.parent, run_label)
    timestamp_value = payload.get("timestamp")
    if timestamp_value:
        return str(timestamp_value)
    return _infer_timestamp_from_run_dir(checkpoint_path.parent, run_label)

def _infer_value_feature_dim(data_dir: Path, value_file: str) -> int:
    """Infer feature dimension F from a [T, N, F] value tensor."""
    value_path = data_dir / value_file
    if not value_path.exists():
        raise FileNotFoundError(
            f"Feature tensor not found: {value_path}. "
            "Run python -m reproduction.data.build_features first for calendar/weather features."
        )
    arr = np.load(value_path, mmap_mode="r")
    if arr.ndim != 3:
        raise ValueError(f"Expected {value_path} to have shape [T, N, F], got {arr.shape}.")
    return int(arr.shape[2])


def _resolve_feature_io(io_cfg: dict, dataset: str, feature_set: str | None) -> tuple[str, dict]:
    resolved_set = feature_set or "target"

    feature_files = io_cfg.get("feature_files", {})
    selected = feature_files.get(resolved_set)
    if selected is None:
        available = ", ".join(sorted(feature_files.keys()))
        raise ValueError(
            f"Dataset '{dataset}' does not provide feature_set='{resolved_set}'. "
            f"Available feature sets: {available}"
        )
    return resolved_set, selected


def _resolve_graph_adjacency_file(graph_mode: str, sim_type: str, sparsity_policy: str) -> str:
    """Resolve the seed adjacency; S is always rebuilt from training data."""
    if sparsity_policy != FINAL_GRAPH_SOURCE_POLICY:
        raise ValueError(
            f"Unsupported graph_sparsity_policy={sparsity_policy!r}; "
            f"expected {FINAL_GRAPH_SOURCE_POLICY!r}."
        )
    mode = normalize_graph_mode(graph_mode)
    if mode == "HG":
        return "adj_HGNN.npy"
    return "adj_hierarchy.npy"


def _git_source_provenance(project_root: Path) -> dict:
    """Capture the source revision used by a run without mutating the repository."""
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=project_root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current") or "DETACHED"
        tracked_status = run_git("status", "--porcelain", "--untracked-files=no")
        untracked_source = run_git(
            "ls-files", "--others", "--exclude-standard", "--", "lagtcn", "reproduction"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot resolve git source provenance: {exc}") from exc
    return {
        "source_git_commit": commit,
        "source_git_branch": branch,
        "source_git_tracked_dirty": bool(tracked_status),
        "source_git_tracked_status": tracked_status.splitlines(),
        "source_git_untracked_code": bool(untracked_source),
        "source_git_untracked_code_files": untracked_source.splitlines(),
    }


def _to_numpy_adj(adj) -> np.ndarray:
    if torch.is_tensor(adj):
        arr = adj.detach().cpu().numpy()
    else:
        arr = np.asarray(adj)
    arr = arr.astype(np.float32)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"adjacency matrix must be square, got shape={arr.shape}")
    return arr



def _undirected_nonself_mask(adj) -> np.ndarray:
    """Return the upper-triangular mask of undirected non-self edges."""
    arr = _to_numpy_adj(adj)
    mask = np.abs(arr) > 1e-12
    np.fill_diagonal(mask, False)
    return np.triu(mask | mask.T, k=1)


def _numeric_distribution(values: list[float]) -> dict[str, float]:
    """Summarize one per-origin graph diagnostic without losing its range."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty graph-diagnostic distribution.")
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "q05": float(np.quantile(arr, 0.05)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "q95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }

def _edge_index_to_dense(edge_index, edge_weight, num_nodes: int) -> np.ndarray:
    if torch.is_tensor(edge_index):
        edge_index_np = edge_index.detach().cpu().numpy()
    else:
        edge_index_np = np.asarray(edge_index)
    if edge_weight is None:
        edge_weight_np = np.ones(edge_index_np.shape[1], dtype=np.float32)
    elif torch.is_tensor(edge_weight):
        edge_weight_np = edge_weight.detach().cpu().numpy().astype(np.float32)
    else:
        edge_weight_np = np.asarray(edge_weight, dtype=np.float32)

    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    if edge_index_np.size:
        adj[edge_index_np[0].astype(int), edge_index_np[1].astype(int)] = edge_weight_np
    return adj


def _summarize_adjacency(component: str, adj, node_names: list[str] | None = None, top_k: int = 50) -> dict:
    arr = _to_numpy_adj(adj)
    n = arr.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    edge_mask = np.abs(arr) > 1e-12
    off_edge_mask = edge_mask & off_mask
    off_weights = arr[off_edge_mask]
    undirected_mask = np.triu(edge_mask | edge_mask.T, k=1)
    self_loop_count = int(np.count_nonzero(np.diag(edge_mask)))

    top_edges = []
    edge_indices = np.argwhere(off_edge_mask)
    if edge_indices.size:
        weights_abs = np.abs(arr[off_edge_mask])
        order = np.argsort(-weights_abs)[:top_k]
        for idx in order:
            src, dst = edge_indices[idx]
            top_edges.append({
                "source": node_names[src] if node_names and src < len(node_names) else int(src),
                "target": node_names[dst] if node_names and dst < len(node_names) else int(dst),
                "source_idx": int(src),
                "target_idx": int(dst),
                "weight": float(arr[src, dst]),
            })

    summary = {
        "component": component,
        "num_nodes": int(n),
        "directed_offdiag_edges": int(np.count_nonzero(off_edge_mask)),
        "undirected_offdiag_edges": int(np.count_nonzero(undirected_mask)),
        "offdiag_density": float(offdiag_density(arr)),
        "self_loop_count": self_loop_count,
        "top_edges_by_abs_weight": top_edges,
    }
    if off_weights.size:
        summary.update({
            "weight_min": float(np.min(off_weights)),
            "weight_max": float(np.max(off_weights)),
            "weight_mean": float(np.mean(off_weights)),
            "weight_abs_mean": float(np.mean(np.abs(off_weights))),
        })
    else:
        summary.update({
            "weight_min": None,
            "weight_max": None,
            "weight_mean": None,
            "weight_abs_mean": None,
        })
    return summary


def _write_graph_edges_csv(path: str, components: dict[str, np.ndarray], node_names: list[str] | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["component", "source_idx", "target_idx", "source", "target", "weight", "is_self_loop"],
        )
        writer.writeheader()
        for component, adj in components.items():
            arr = _to_numpy_adj(adj)
            edge_indices = np.argwhere(np.abs(arr) > 1e-12)
            for src, dst in edge_indices:
                writer.writerow({
                    "component": component,
                    "source_idx": int(src),
                    "target_idx": int(dst),
                    "source": node_names[src] if node_names and src < len(node_names) else int(src),
                    "target": node_names[dst] if node_names and dst < len(node_names) else int(dst),
                    "weight": float(arr[src, dst]),
                    "is_self_loop": bool(src == dst),
                })


def _save_graph_info(
    model: nn.Module,
    config: dict,
    loader: LoadDatasetLoader,
    hierarchy_adj: np.ndarray,
    static_component_adj: np.ndarray | None,
    test_dataset,
    static_edge_index: torch.Tensor,
    device: torch.device,
) -> str:
    """Save graph metadata and edge lists for this run."""
    node_names = list(getattr(loader, "node_names", []) or [])
    is_lagtcn = isinstance(model, LAGTCNBaseline)
    components: dict[str, np.ndarray] = {"hierarchy": _to_numpy_adj(hierarchy_adj)}
    if not is_lagtcn:
        components["base_graph_used"] = _to_numpy_adj(loader.A)

    if static_component_adj is not None:
        components["static_similarity_component"] = _to_numpy_adj(static_component_adj)

    sample_note = None
    dynamic_distribution = None
    dynamic_pairwise_distribution = None
    if hasattr(model, "_compute_adaptive_adj") and getattr(model, "use_adaptive", False):
        try:
            with torch.no_grad():
                components["adaptive_component_final"] = _to_numpy_adj(model._compute_adaptive_adj())
        except Exception as exc:
            sample_note = f"adaptive_component_final unavailable: {exc}"
    elif hasattr(model, "get_adaptive_adjacency"):
        try:
            with torch.no_grad():
                components["native_adaptive_graph_final"] = _to_numpy_adj(
                    model.get_adaptive_adjacency(device=device, dtype=torch.float32)
                )
        except Exception as exc:
            sample_note = f"native_adaptive_graph_final unavailable: {exc}"

    if len(getattr(test_dataset, "features", [])) > 0 and hasattr(model, "_resolve_graph_edges"):
        sample_count = min(int(config.get("batch_size", 128)), len(test_dataset.features))
        x_sample = torch.tensor(
            np.stack(test_dataset.features[:sample_count], axis=0),
            dtype=torch.float32,
            device=device,
        )
        try:
            with torch.no_grad():
                if getattr(model, "use_dynamic", False):
                    dynamic_batch = model._compute_samplewise_dynamic_adj(x_sample)
                    components["dynamic_component_first_test_origin"] = _to_numpy_adj(
                        dynamic_batch[0]
                    )
                if is_lagtcn:
                    for source_name, source_adj in model._graph_parts(x_sample, static_edge_index, None):
                        if source_adj.dim() == 3:
                            component_name = f"active_{source_name}_source_first_test_origin"
                            source_adj = source_adj[0]
                        else:
                            component_name = f"active_{source_name}_source_first_test_batch"
                        components[component_name] = _to_numpy_adj(source_adj)
                else:
                    resolved_edge_index, resolved_edge_weight = model._resolve_graph_edges(
                        x_sample,
                        static_edge_index,
                        None,
                    )
                    components["resolved_graph_first_test_batch"] = _edge_index_to_dense(
                        resolved_edge_index,
                        resolved_edge_weight,
                        int(config["node_num"]),
                    )
        except Exception as exc:
            sample_note = f"resolved/dynamic graph sample unavailable: {exc}"

    if is_lagtcn and getattr(model, "use_dynamic", False) and len(
        getattr(test_dataset, "features", [])
    ) > 0:
        try:
            edge_counts: list[int] = []
            overlap_counts: list[int] = []
            overlap_rates: list[float] = []
            hierarchy_mask = np.abs(_to_numpy_adj(hierarchy_adj)) > 1e-12
            np.fill_diagonal(hierarchy_mask, False)
            hierarchy_upper = np.triu(hierarchy_mask | hierarchy_mask.T, k=1)
            reference_masks = {
                name: _undirected_nonself_mask(adj)
                for name, adj in components.items()
                if name.startswith("active_") and "dynamic" not in name
            }
            pairwise_samples = {
                name: {
                    "intersection_edge_count": [],
                    "union_edge_count": [],
                    "jaccard": [],
                    "dynamic_unique_edge_count": [],
                    "reference_unique_edge_count": [],
                }
                for name in reference_masks
            }
            diagnostic_batch_size = min(int(config.get("batch_size", 128)), 64)
            with torch.no_grad():
                for start in range(0, len(test_dataset.features), diagnostic_batch_size):
                    x_batch = torch.tensor(
                        np.stack(
                            test_dataset.features[start:start + diagnostic_batch_size],
                            axis=0,
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    dynamic_batch = model._compute_samplewise_dynamic_adj(x_batch)
                    for dynamic_adj in dynamic_batch.detach().cpu().numpy():
                        mask = np.abs(dynamic_adj) > 1e-12
                        np.fill_diagonal(mask, False)
                        upper = np.triu(mask | mask.T, k=1)
                        edge_count = int(np.count_nonzero(upper))
                        overlap_count = int(np.count_nonzero(upper & hierarchy_upper))
                        edge_counts.append(edge_count)
                        overlap_counts.append(overlap_count)
                        overlap_rates.append(
                            overlap_count / edge_count if edge_count else 0.0
                        )
                        for source_name, reference_upper in reference_masks.items():
                            intersection = int(np.count_nonzero(upper & reference_upper))
                            union = int(np.count_nonzero(upper | reference_upper))
                            samples = pairwise_samples[source_name]
                            samples["intersection_edge_count"].append(float(intersection))
                            samples["union_edge_count"].append(float(union))
                            samples["jaccard"].append(
                                intersection / union if union else 0.0
                            )
                            samples["dynamic_unique_edge_count"].append(
                                float(np.count_nonzero(upper & ~reference_upper))
                            )
                            samples["reference_unique_edge_count"].append(
                                float(np.count_nonzero(reference_upper & ~upper))
                            )
            counts = np.asarray(edge_counts, dtype=np.float64)
            rates = np.asarray(overlap_rates, dtype=np.float64)
            dynamic_distribution = {
                "origin_count": len(edge_counts),
                "undirected_edge_count": {
                    "mean": float(counts.mean()),
                    "std": float(counts.std(ddof=1)) if counts.size > 1 else 0.0,
                    "min": int(counts.min()),
                    "q05": float(np.quantile(counts, 0.05)),
                    "q25": float(np.quantile(counts, 0.25)),
                    "median": float(np.median(counts)),
                    "q75": float(np.quantile(counts, 0.75)),
                    "q95": float(np.quantile(counts, 0.95)),
                    "max": int(counts.max()),
                },
                "hierarchy_overlap_edge_count_mean": float(np.mean(overlap_counts)),
                "hierarchy_overlap_rate_mean": float(rates.mean()),
                "hierarchy_overlap_rate_std": (
                    float(rates.std(ddof=1)) if rates.size > 1 else 0.0
                ),
            }
            dynamic_pairwise_distribution = {
                source_name: {
                    metric: _numeric_distribution(values)
                    for metric, values in samples.items()
                }
                for source_name, samples in pairwise_samples.items()
            }
        except Exception as exc:
            note = f"dynamic distribution unavailable: {exc}"
            sample_note = f"{sample_note}; {note}" if sample_note else note

    graph_edges_path = os.path.join(config["output_dir"], artifact_filename("graph_edges"))
    _write_graph_edges_csv(graph_edges_path, components, node_names=node_names)

    component_summaries = []
    for component, adj in components.items():
        summary = _summarize_adjacency(component, adj, node_names=node_names)
        diagnostics = graph_edge_diagnostics(adj, hierarchy_adj=hierarchy_adj)
        overlap_count = diagnostics["hierarchy_overlap_edge_count"]
        edge_count = diagnostics["undirected_edge_count"]
        summary.update({
            "degree_min": diagnostics["degree_min"],
            "degree_mean": diagnostics["degree_mean"],
            "degree_max": diagnostics["degree_max"],
            "isolated_node_count": diagnostics["isolated_node_count"],
            "hierarchy_overlap_edge_count": overlap_count,
            "hierarchy_overlap_rate": (
                overlap_count / edge_count if edge_count else 0.0
            ),
        })
        component_summaries.append(summary)

    active_components = {
        name: adj for name, adj in components.items() if name.startswith("active_")
    }
    pairwise_source_overlap = []
    active_names = sorted(active_components)
    for first_idx, first_name in enumerate(active_names):
        first = np.abs(_to_numpy_adj(active_components[first_name])) > 1e-12
        np.fill_diagonal(first, False)
        first = np.triu(first | first.T, k=1)
        for second_name in active_names[first_idx + 1:]:
            second = np.abs(_to_numpy_adj(active_components[second_name])) > 1e-12
            np.fill_diagonal(second, False)
            second = np.triu(second | second.T, k=1)
            intersection = int(np.count_nonzero(first & second))
            union = int(np.count_nonzero(first | second))
            pairwise_source_overlap.append({
                "source_a": first_name,
                "source_b": second_name,
                "intersection_edge_count": intersection,
                "union_edge_count": union,
                "jaccard": intersection / union if union else 0.0,
            })

    active_source_unique_edges = []
    active_masks = {
        name: _undirected_nonself_mask(adj)
        for name, adj in active_components.items()
    }
    for source_name, source_mask in sorted(active_masks.items()):
        other_masks = [
            mask for name, mask in active_masks.items() if name != source_name
        ]
        other_union = (
            np.logical_or.reduce(other_masks)
            if other_masks else np.zeros_like(source_mask, dtype=bool)
        )
        active_source_unique_edges.append({
            "source": source_name,
            "total_edge_count": int(np.count_nonzero(source_mask)),
            "unique_edge_count": int(np.count_nonzero(source_mask & ~other_union)),
            "overlapping_edge_count": int(np.count_nonzero(source_mask & other_union)),
        })
    graph_info = {
        "dataset": config.get("dataset"),
        "model_name": config.get("model_name"),
        "timestamp": config.get("timestamp"),
        "paper_scope": config.get("paper_scope"),
        "experiment_stage": config.get("experiment_stage"),
        "experiment_id": config.get("experiment_id"),
        "graph_mode": config.get("graph_mode"),
        "sim_type": config.get("sim_type"),
        "gnn_type": config.get("gnn_type"),
        "temporal_type": config.get("temporal_type"),
        "st_mode": config.get("st_mode"),
        "stgnn_graph_source": config.get("stgnn_graph_source"),
        "lagtcn_graph_source_version": config.get("lagtcn_graph_source_version"),
        "active_graph_sources": config.get("lagtcn_active_graph_sources") if is_lagtcn else None,
        "block_source_gates": model.get_graph_source_gates() if is_lagtcn else None,
        "base_graph_used_by_model": not is_lagtcn,
        "num_nodes": int(config.get("node_num")),
        "node_names": node_names,
        "sparsity_policy": {
            "graph_sparsity_policy": config.get("graph_sparsity_policy"),
            "graph_protocol_version": config.get("graph_protocol_version"),
            "graph_design_protocol_version": config.get("graph_design_protocol_version"),
            "static_threshold": config.get("static_threshold"),
            "adaptive_top_k": config.get("adaptive_top_k"),
            "dynamic_threshold": config.get("dynamic_threshold"),
            "static_component_graph_diagnostics": config.get("static_component_graph_diagnostics"),
            "native_top_k": config.get("native_top_k"),
            "hierarchy_density": config.get("hierarchy_density"),
            "fixed_seed_graph_density": config.get("fixed_seed_graph_density"),
            "adj_runtime_rebuilt": config.get("adj_runtime_rebuilt"),
            "adj_runtime_source": config.get("adj_runtime_source"),
            "base_graph_density_actual": config.get("base_graph_density_actual"),
        },
        "component_summaries": component_summaries,
        "pairwise_active_source_overlap": pairwise_source_overlap,
        "active_source_overlap_scope": (
            "fixed_source_topology; first_retained_test_origin_for_dynamic_sources"
        ),
        "active_source_unique_edges": active_source_unique_edges,
        "dynamic_pairwise_source_overlap_distribution": dynamic_pairwise_distribution,
        "dynamic_pairwise_scope": "all_retained_test_origins",
        "dynamic_origin_distribution": dynamic_distribution,
        "edge_list_csv": graph_edges_path,
        "sample_note": sample_note,
    }
    graph_info_path = os.path.join(config["output_dir"], artifact_filename("graph_info"))
    with open(graph_info_path, "w", encoding="utf-8") as f:
        json.dump(graph_info, f, indent=4, ensure_ascii=False, allow_nan=False)
    return graph_info_path


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train LAGTCN or a paper baseline.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Physical mini-batch size used by each forward/backward pass.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Number of physical mini-batches accumulated before each optimizer step; "
            "the nominal effective batch size is batch_size times this value."
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="GEFCom2012_2level",
        help="Formal Applied Energy dataset folder under Data.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root data directory (default: <repo>/Data).",
    )
    parser.add_argument("--num-timesteps-in", type=int, default=168, help="Input window length L_in.")
    parser.add_argument("--num-timesteps-out", type=int, default=24, help="Forecast horizon L_out.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension size.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of GCN-GRU layers.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument(
        "--training-loss-space",
        choices=("original", "normalized_log"),
        default="original",
        help="Training/checkpoint-selection loss space; normalized_log is diagnostic until formally adopted.",
    )
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument(
        "--resume",
        type=str,
        default="none",
        help="Resume training from a checkpoint: none, auto, or a checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=1,
        help="Save a resumable training checkpoint every N epochs; set 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Train/select on train-validation and stop before any test evaluation (tuning only).",
    )
    parser.add_argument(
        "--expected-git-commit",
        default=None,
        help="Optional source commit that must match the runtime checkout.",
    )
    parser.add_argument(
        "--expected-git-branch",
        default=None,
        help="Optional source branch that must match the runtime checkout.",
    )
    parser.add_argument(
        "--require-clean-worktree",
        action="store_true",
        help="Fail before training when tracked files differ from HEAD.",
    )
    parser.add_argument("--selection-source-experiment-id", default=None)
    parser.add_argument("--selection-protocol-version", default=None)
    parser.add_argument("--graph-selection-source-experiment-id", default=None)
    parser.add_argument("--graph-selection-protocol-version", default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto / cpu / cuda / cuda:0",
    )
    parser.add_argument("--graph-mode", type=str, default="H", help="Graph config: H / HG / S / A / D / H+S / H+A / H+D / S+A+D / H+S+A+D")
    parser.add_argument("--sim-type", type=str, default="cosine", help="Similarity type: pearson or cosine.")
    parser.add_argument(
        "--graph-sparsity-policy",
        type=str,
        default=FINAL_GRAPH_SOURCE_POLICY,
        choices=[FINAL_GRAPH_SOURCE_POLICY],
        help="Validation-selected S/D thresholds and adaptive top-k.",
    )
    parser.add_argument("--static-threshold", type=float, default=None)
    parser.add_argument("--adaptive-top-k", type=int, default=None)
    parser.add_argument("--dynamic-threshold", type=float, default=None)
    parser.add_argument(
        "--model-name",
        type=str,
        default="LAGTCN",
        choices=["LAGTCN", "DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER", "DCRNN", "MTGNN"],
        help="Model reported in the paper.",
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gcn",
        help="Spatial backbone used by LAGTCN: none/gcn/gatv2/graphsage/transformer.",
    )
    parser.add_argument(
        "--temporal-type",
        type=str,
        default="gru",
        help=(
            "Temporal backbone: gru/transformer/tcn/timemixer/patchtst/"
            "patch_transformer/dlinear/itransformer; patch_transformer is the "
            "LAGTCN encoder."
        ),
    )
    parser.add_argument(
        "--output-namespace",
        type=str,
        default=None,
        help=(
            "Optional relative output folder under Data/<dataset>/output. "
            "Known long segments are compacted, e.g. journal_applied_energy/stage_1_graph/target "
            "becomes ae/j2/target. Defaults to <paper>/<stage>/<feature> aliases."
        ),
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional run subdirectory prefix. Defaults to model name.",
    )
    parser.add_argument(
        "--paper-scope",
        type=str,
        default="journal_applied_energy",
        choices=["journal_applied_energy"],
        help="Logical result owner; this repository uses journal_applied_energy.",
    )
    parser.add_argument(
        "--experiment-stage",
        type=str,
        default="adhoc",
        help="Single-segment experiment stage label, e.g. stage_1_graph or step0_base_selection.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Batch-level run id shared by a manifest. Defaults to the per-run timestamp.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        choices=[
            "target",
            "target_calendar",
            "target_calendar_weather",
        ],
        help=(
            "Input feature tensor to use. Defaults to target. "
            "Calendar/weather tensors are generated by python -m reproduction.data.build_features."
        ),
    )
    parser.add_argument(
        "--coherency-lambda",
        type=float,
        default=0.0,
        help="Weight for coherency violation loss (0 = disabled). Recommended: 0.01-0.1.",
    )
    parser.add_argument(
        "--st-mode",
        type=str,
        default="sequential",
        choices=["sequential", "alternating", "hier_fusion", "hierarchy_fusion", "hierarchical_fusion"],
        help=(
            "Spatio-temporal interaction mode: sequential (default), alternating "
            "(GNN->T->GNN->T), or hier_fusion (hierarchy-aware temporal/node-token fusion)."
        ),
    )
    parser.add_argument(
        "--stgnn-graph-source",
        type=str,
        default="hybrid",
        choices=["project", "native", "hybrid"],
        help=(
            "Graph source for dedicated STGNN baselines. project uses this project's graph_mode "
            "(I/H/HG/S/A/D/S+A+D/H+S/H+A/H+D/H+S+A+D); native uses the model's own learned graph when available; "
            "hybrid combines project graph and native adaptive graph."
        ),
    )
    parser.add_argument(
        "--lagtcn-ablation",
        choices=["none", "no_level", "no_coevolution", "uniform_fusion"],
        default="none",
        help="Pre-registered LAGTCN ablation; no-cross-node uses graph-mode I with none.",
    )
    parser.add_argument(
        "--lagtcn-decoder-mode",
        choices=["persistence_residual", "seasonal_residual", "direct"],
        default="persistence_residual",
        help=(
            "LAGTCN output parameterization: residual around the latest observation, "
            "residual around the lag-24 seasonal-naive path, or a direct forecast."
        ),
    )
    parser.add_argument(
        "--lagtcn-residual-scale-mode",
        choices=["fixed", "unit", "learnable"],
        default="unit",
        help="Use the configured fixed multiplier, unit residual scale, or a learned sigmoid gate.",
    )
    parser.add_argument(
        "--lagtcn-residual-scale-init",
        type=float,
        default=1.0,
        help=(
            "Initial residual multiplier; fixed value when scale mode is fixed and ignored "
            "when scale mode is unit. Learnable pilot runs must set a value strictly between 0 and 1."
        ),
    )
    parser.add_argument(
        "--lagtcn-seasonal-lag",
        type=int,
        default=24,
        help="Seasonal lag used by seasonal_residual (24 for hourly daily seasonality).",
    )
    parser.add_argument(
        "--native-top-k",
        type=int,
        default=5,
        help="MTGNN native learned directed-graph top-k (submission default: min(N-1, 5)).",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip loss and prediction plot generation.")
    parser.add_argument(
        "--plot-node-limit",
        type=int,
        default=None,
        help="Optional cap on per-node prediction plots. Default plots all nodes.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    device = _resolve_device(args.device)
    logging.info(f"Using device: {device}")

    valid_gnn = {"none", "gcn", "gatv2", "graphsage", "transformer"}
    valid_temporal = {
        "gru", "transformer", "tcn", "timemixer", "patchtst",
        "patch_transformer", "dlinear", "itransformer",
    }
    args.gnn_type = str(args.gnn_type).lower()
    args.temporal_type = str(args.temporal_type).lower()
    args.model_name = str(args.model_name).upper()
    if args.gnn_type not in valid_gnn:
        raise ValueError(f"Unknown --gnn-type={args.gnn_type}. Choose from: {sorted(valid_gnn)}")
    if args.temporal_type not in valid_temporal:
        raise ValueError(f"Unknown --temporal-type={args.temporal_type}. Choose from: {sorted(valid_temporal)}")

    if args.plot_node_limit is not None and args.plot_node_limit < 0:
        raise ValueError("--plot-node-limit must be non-negative.")
    dataset_file_map = {
        "GEFCom2012_2level": {
            "feature_files": {
                "target": {
                    "raw_csv_file": "Load_GEFCom2012_hourly.csv",
                    "value_file": "node_values.npy",
                },
                "target_calendar": {
                    "raw_csv_file": "Load_GEFCom2012_hourly.csv",
                    "value_file": "node_values_calendar.npy",
                },
            },
        },
        "GEFCom2017QualifyingMatch_3level": {
            "feature_files": {
                "target": {
                    "raw_csv_file": "GEFCom2017QualifyingMatchDemand.csv",
                    "value_file": "node_values.npy",
                },
                "target_calendar": {
                    "raw_csv_file": "GEFCom2017QualifyingMatchDemand.csv",
                    "value_file": "node_values_calendar.npy",
                },
            },
        },
        "GEFCom2017FinalMatch_4level": {
            "feature_files": {
                "target": {
                    "raw_csv_file": "load_final_filled.csv",
                    "value_file": "node_values.npy",
                },
                "target_calendar": {
                    "raw_csv_file": "load_final_filled.csv",
                    "value_file": "node_values_calendar.npy",
                },
                "target_calendar_weather": {
                    "raw_csv_file": "load_final_filled.csv",
                    "value_file": "node_values_calendar_weather.npy",
                },
            },
        },
    }
    io_cfg = dataset_file_map.get(args.dataset)
    if io_cfg is None:
        raise ValueError(
            f"Unsupported dataset '{args.dataset}'. "
            f"Supported datasets: {', '.join(dataset_file_map.keys())}"
        )
    feature_set, selected_io = _resolve_feature_io(
        io_cfg,
        dataset=args.dataset,
        feature_set=args.feature_set,
    )
    selected_value_file = selected_io.get("value_file")
    selected_raw_csv_file = selected_io.get("raw_csv_file")
    if not selected_value_file or not selected_raw_csv_file:
        raise ValueError(f"Dataset '{args.dataset}' is missing file mapping for feature_set={feature_set}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    source_provenance = _git_source_provenance(project_root)
    if args.expected_git_commit and source_provenance["source_git_commit"] != args.expected_git_commit:
        raise RuntimeError(
            "Runtime git commit differs from the frozen manifest: "
            f"{source_provenance['source_git_commit']} != {args.expected_git_commit}."
        )
    if args.expected_git_branch and source_provenance["source_git_branch"] != args.expected_git_branch:
        raise RuntimeError(
            "Runtime git branch differs from the frozen manifest: "
            f"{source_provenance['source_git_branch']} != {args.expected_git_branch}."
        )
    if args.require_clean_worktree and (
        source_provenance["source_git_tracked_dirty"]
        or source_provenance["source_git_untracked_code"]
    ):
        raise RuntimeError(
            "Formal run requires clean tracked files and no untracked package/reproduction source; "
            f"tracked={source_provenance['source_git_tracked_status']}, "
            f"untracked_code={source_provenance['source_git_untracked_code_files']}"
        )
    data_root = Path(args.data_root) if args.data_root else project_root / "Data"
    data_dir = data_root / args.dataset
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    selected_input_dim = _infer_value_feature_dim(data_dir, selected_value_file)

    st_mode = str(args.st_mode).lower()
    if st_mode in {"hierarchy_fusion", "hierarchical_fusion"}:
        st_mode = "hier_fusion"

    config = {
        'dataset': args.dataset,
        'raw_data_dir': str(data_dir),
        'output_dir': str(data_dir / 'output'),
        # Infer from loaded data unless explicitly overridden later.
        'node_num': None,
        'input_dim': selected_input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.num_timesteps_out,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        'epochs': args.epochs,
        'lr': args.lr,
        'patience': args.patience,
        'resume': args.resume,
        'training_loss_space': args.training_loss_space,
        'checkpoint_every_epochs': args.checkpoint_every_epochs,
        'seed': args.seed,
        'validation_only': bool(args.validation_only),
        'selection_source_experiment_id': args.selection_source_experiment_id,
        'selection_protocol_version': args.selection_protocol_version,
        'graph_selection_source_experiment_id': args.graph_selection_source_experiment_id,
        'graph_selection_protocol_version': args.graph_selection_protocol_version,
        'num_timesteps_in': args.num_timesteps_in,
        'num_timesteps_out': args.num_timesteps_out,

        # Paper model and its spatial/temporal implementation choices.
        'model_name': args.model_name,
        'gnn_type': args.gnn_type,
        'temporal_type': args.temporal_type,

        'timestamp': timestamp,
        'feature_set': feature_set,
        'value_file': selected_value_file,
        'raw_csv_file': selected_raw_csv_file,
        # Graph config: I / H / HG / S / A / D and LAGTCN multi-source combinations such as H+S / H+A / H+D / H+S+A+D
        'graph_mode': args.graph_mode,
        'sim_type': args.sim_type,  # pearson / cosine
        'graph_sparsity_policy': args.graph_sparsity_policy,
        'static_threshold': args.static_threshold,
        'adaptive_top_k': args.adaptive_top_k,
        'dynamic_threshold': args.dynamic_threshold,
        'include_self_loops': True,
        'stgnn_graph_source': str(args.stgnn_graph_source).lower(),
        'lagtcn_ablation': str(args.lagtcn_ablation),
        'lagtcn_decoder_mode': str(args.lagtcn_decoder_mode),
        'lagtcn_residual_scale_mode': str(args.lagtcn_residual_scale_mode),
        'lagtcn_residual_scale_init': (
            1.0
            if str(args.lagtcn_residual_scale_mode).lower() == "unit"
            else float(args.lagtcn_residual_scale_init)
        ),
        'lagtcn_seasonal_lag': int(args.lagtcn_seasonal_lag),
        'native_top_k': int(args.native_top_k),
        'coherency_lambda': args.coherency_lambda,
        'st_mode': st_mode,
        'no_plots': bool(args.no_plots),
        'plot_node_limit': args.plot_node_limit,
        **source_provenance,
    }

    paper_scope = _validate_path_segment("--paper-scope", args.paper_scope)
    experiment_stage = _validate_path_segment("--experiment-stage", args.experiment_stage)
    experiment_id = _validate_path_segment("--experiment-id", args.experiment_id or timestamp)
    feature_tag_map = {
        "target": "target",
    }
    feature_tag = feature_tag_map.get(feature_set, feature_set)
    config["feature_tag"] = feature_tag
    if args.output_namespace:
        original_output_namespace = _validate_path_namespace("--output-namespace", args.output_namespace)
        output_namespace = compact_namespace(original_output_namespace)
        if output_namespace != original_output_namespace:
            config["output_namespace_original"] = original_output_namespace
    else:
        output_namespace = compact_output_namespace(paper_scope, experiment_stage, feature_tag)
    graph_mode = normalize_graph_mode(config.get("graph_mode", "H"))
    sim_type = config.get("sim_type", "cosine").lower()
    if sim_type not in {"pearson", "cosine"}:
        raise ValueError(f"Unknown sim_type: {sim_type}")
    valid_graph_modes = {"I", "H", "HG", "S", "A", "D", "S+A+D", "H+S", "H+A", "H+D", "H+S+A+D"}
    if graph_mode not in valid_graph_modes:
        raise ValueError(
            f"Unknown graph_mode: {graph_mode}. "
            "Supported modes are I, H, HG, S, A, D, S+A+D, H+S, H+A, H+D, H+S+A+D. "
            "HG is intentionally only compared against H, not combined with S/A/D."
        )

    graph_tokens = graph_components(graph_mode)
    if graph_tokens.intersection({"S", "A", "D"}) and config["model_name"] != "LAGTCN":
        raise ValueError(
            "S/A/D and their fusion modes are implemented only by LAGTCN under "
            "the current independent source-propagation protocol."
        )
    uses_hgnn_base = graph_mode == "HG"
    uses_hierarchy_base = "H" in graph_tokens or uses_hgnn_base
    uses_static_sim = "S" in graph_tokens
    uses_adaptive = "A" in graph_tokens
    uses_dynamic = "D" in graph_tokens
    if args.graph_sparsity_policy == FINAL_GRAPH_SOURCE_POLICY:
        required = {
            "static_threshold": (uses_static_sim, args.static_threshold),
            "adaptive_top_k": (uses_adaptive, args.adaptive_top_k),
            "dynamic_threshold": (uses_dynamic, args.dynamic_threshold),
        }
        missing = [name for name, (active, value) in required.items() if active and value is None]
        if missing:
            raise ValueError(
                f"{FINAL_GRAPH_SOURCE_POLICY} requires explicit selected values for: {', '.join(missing)}."
            )
        for name, value in (("static_threshold", args.static_threshold), ("dynamic_threshold", args.dynamic_threshold)):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")
        if args.adaptive_top_k is not None and int(args.adaptive_top_k) < 0:
            raise ValueError(f"adaptive_top_k must be nonnegative, got {args.adaptive_top_k}.")
    if config["model_name"] == "LAGTCN":
        config["lagtcn_graph_source_version"] = LAGTCN_GRAPH_SOURCE_VERSION_CURRENT
        config["lagtcn_active_graph_sources"] = list(lagtcn_graph_sources(graph_mode))
    adj_file_name = _resolve_graph_adjacency_file(
        graph_mode, sim_type, args.graph_sparsity_policy
    )

    if args.run_label:
        original_run_label = _validate_path_segment("--run-label", args.run_label)
        run_label = shorten_existing_run_label(original_run_label)
        if run_label != original_run_label:
            config["run_label_original"] = original_run_label
    else:
        run_label = short_run_label(
            stage=experiment_stage,
            model_name=config["model_name"],
            gnn_type=config["gnn_type"],
            temporal_type=config["temporal_type"],
            graph_mode=graph_mode,
            horizon=int(config["num_timesteps_out"]),
            seed=int(config["seed"]),
            st_mode=config.get("st_mode"),
            stgnn_graph_source=config.get("stgnn_graph_source"),
        )
    run_label = _validate_path_segment("--run-label", run_label)

    config["graph_mode"] = graph_mode
    config["sim_type"] = sim_type
    config["adj_file"] = adj_file_name
    config["paper_scope"] = paper_scope
    config["experiment_stage"] = experiment_stage
    config["experiment_id"] = experiment_id
    if is_formal_ae_stage(experiment_stage):
        violations = []
        if paper_scope != "journal_applied_energy":
            violations.append("paper_scope must be journal_applied_energy")
        if args.num_timesteps_in != 168 or args.num_timesteps_out != 24:
            violations.append("input/output lengths must be 168/24")
        if args.graph_sparsity_policy != FINAL_GRAPH_SOURCE_POLICY:
            violations.append(f"graph policy must be {FINAL_GRAPH_SOURCE_POLICY}")
        if abs(float(args.coherency_lambda)) > 0.0:
            violations.append("coherency_lambda must be 0 for formal Base forecasting")
        if args.training_loss_space != "original":
            violations.append("formal protocol currently freezes training_loss_space=original")
        if args.lagtcn_decoder_mode != "persistence_residual":
            violations.append("formal protocol currently freezes LAGTCN persistence_residual decoding")
        if args.lagtcn_residual_scale_mode != "unit":
            violations.append("formal protocol freezes the LAGTCN residual scale at eta=1")
        if int(args.lagtcn_seasonal_lag) != 24:
            violations.append("formal protocol currently freezes the LAGTCN seasonal lag at 24")
        if experiment_stage == "ae_final_tuning_v1" and not args.validation_only:
            violations.append("tuning must use --validation-only")
        if experiment_stage == "ae_final_graph_tuning_v3":
            if not args.validation_only:
                violations.append("graph tuning must use --validation-only")
            if not args.selection_source_experiment_id or not args.selection_protocol_version:
                violations.append("graph tuning requires frozen model-hyperparameter provenance")
        if experiment_stage == "ae_final_main_v1":
            if args.validation_only:
                violations.append("main/ablation must evaluate the test partition")
            if not args.selection_source_experiment_id or not args.selection_protocol_version:
                violations.append("audited tuning-selection provenance is required")
            if (uses_static_sim or uses_adaptive or uses_dynamic) and (
                not args.graph_selection_source_experiment_id
                or not args.graph_selection_protocol_version
            ):
                violations.append("data-driven graphs require audited graph-selection provenance")
        if violations:
            raise ValueError(
                "Invalid frozen Applied Energy invocation: " + "; ".join(violations)
            )
    config["output_namespace"] = output_namespace
    config["run_label"] = run_label
    adj_subdir = graph_subdir(graph_mode, sim_type)
    config["output_dir"] = str(data_dir / "output" / output_namespace / adj_subdir)

    base_output_dir = config['output_dir']
    resume_checkpoint = _resolve_resume_checkpoint(args.resume, Path(base_output_dir), run_label)
    if resume_checkpoint is not None:
        checkpoint_ts = _checkpoint_timestamp(resume_checkpoint, run_label)
        if checkpoint_ts:
            timestamp = checkpoint_ts
            config['timestamp'] = timestamp
        config['output_dir'] = str(resume_checkpoint.parent)
        config['resume_checkpoint'] = str(resume_checkpoint)
    else:
        run_subdir = f"{run_label}_{compact_timestamp(timestamp)}"
        config['output_dir'] = os.path.join(base_output_dir, run_subdir)
        config['resume_checkpoint'] = None

    # 设置随机种子
    set_seed(config['seed'])

    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)

    # 加载数据
    logging.info("Loading dataset...")
    loader = LoadDatasetLoader(
        config['raw_data_dir'],
        input_dim=config['input_dim'],
        adj_file=config.get('adj_file', 'adj_hierarchy.npy'),
        value_file=selected_value_file,
        raw_csv_file=selected_raw_csv_file,
    )

    # 将层次信息写入 config，便于后续模型和评估使用
    config['num_total_nodes'] = int(loader.num_total_nodes)
    config['num_bottom_nodes'] = int(loader.num_bottom_nodes)
    config['bottom_start_idx'] = int(loader.bottom_start_idx)
    config['num_mid_nodes'] = int(loader.num_mid_nodes)
    config['middle_levels'] = (
        loader.hierarchy_info.get("middle_levels", [])
        if loader.hierarchy_info else []
    )
    config['middle_levels_provenance'] = (
        loader.hierarchy_info.get("middle_levels_provenance")
        if loader.hierarchy_info else None
    )
    config['node_order'] = [str(name) for name in list(loader.node_names)] if loader.node_names is not None else []
    sum_matrix_bytes = np.ascontiguousarray(loader.sum_matrix, dtype=np.float32).tobytes()
    config['sum_matrix_shape'] = [int(v) for v in loader.sum_matrix.shape]
    config['sum_matrix_sha256'] = hashlib.sha256(sum_matrix_bytes).hexdigest()
    config['output_length'] = int(config['num_timesteps_out'])
    config['forecast_hour_definition'] = "positions_within_single_direct_output"
    config['primary_lead_range'] = (
        "1:24" if int(config['num_timesteps_out']) == 24
        else f"1:{int(config['num_timesteps_out'])}"
    )
    if is_formal_ae_stage(config.get("experiment_stage")):
        if config.get("feature_set") != "target":
            raise ValueError("The frozen Applied Energy protocol requires feature_set=target.")
        ae_mase.assert_unit_stride(loader.time_index, expected_step="1h")
        total_length = int(loader.X.shape[2])
        train_length = int(
            loader.norm_params.get("train_T")
            or int(total_length * float(loader.norm_params.get("train_ratio", 0.8)))
        )
        expected_train_length = int(total_length * 0.8)
        if train_length != expected_train_length:
            raise ValueError(
                "Frozen sMASE train boundary differs from target-timestamp split: "
                f"{train_length} != {expected_train_length}."
            )
        normalized_train = (
            loader.X[:, 0, :train_length].detach().cpu().numpy().T
        )
        original_train = loader.inverse_log_transform(
            loader.denormalize_data(normalized_train)
        )
        mase_scale = ae_mase.compute_naive_scale(
            original_train, seasonal_period=ae_mase.MASE_SEASONAL_PERIOD
        )
        config["_mase_scale"] = mase_scale
        config["smase_scale_metadata"] = ae_mase.naive_scale_metadata(
            mase_scale,
            train_length=train_length,
            node_names=config["node_order"],
            seasonal_period=ae_mase.MASE_SEASONAL_PERIOD,
        )

    # 确保 node_num 与数据一致；默认从数据自动推导
    if config['node_num'] is None:
        config['node_num'] = config['num_total_nodes']
    elif config['node_num'] != config['num_total_nodes']:
        logging.warning(
            f"config['node_num']={config['node_num']} 与数据中的节点数 "
            f"{config['num_total_nodes']} 不一致，已自动对齐。"
        )
        config['node_num'] = config['num_total_nodes']

    hierarchy_adj_path = data_dir / "adj_hierarchy.npy"
    hierarchy_adj = np.load(hierarchy_adj_path).astype(np.float32)
    hierarchy_density = offdiag_density(hierarchy_adj)
    identity_adj = np.eye(config["node_num"], dtype=np.float32)
    hierarchy_source_adj = (
        _to_numpy_adj(loader.A).astype(np.float32)
        if uses_hgnn_base
        else hierarchy_adj.copy()
    )
    similarity_source_adj = None
    graph_base_adj = hierarchy_source_adj if uses_hierarchy_base else identity_adj
    fixed_seed_density = offdiag_density(graph_base_adj)
    static_component_adj = None
    if config["graph_sparsity_policy"] != FINAL_GRAPH_SOURCE_POLICY:
        raise AssertionError("Graph policy validation did not run before graph construction.")

    config.update({
        "graph_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
        "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
        "physical_hierarchy_density": hierarchy_density,
        "hierarchy_density": hierarchy_density,
        "fixed_seed_graph_density": fixed_seed_density,
    })
    if uses_adaptive:
        config["adaptive_top_k"] = min(
            int(config["adaptive_top_k"]), max(0, int(config["node_num"]) - 1)
        )

    if uses_static_sim:
        train_end = max(1, int(loader.X.shape[2] * 0.8))
        static_sim = compute_similarity_numpy(
            loader.X[:, 0, :train_end].detach().cpu().numpy().T,
            sim_type=sim_type,
            use_abs=True,
        )
        static_adj = build_threshold_similarity_adj(
            static_sim,
            threshold=float(config["static_threshold"]),
            include_self_loops=config["include_self_loops"],
            use_weights=True,
        )
        static_component_adj = static_adj
        similarity_source_adj = static_adj.copy()
        config["static_component_density_actual"] = offdiag_density(static_adj)
        config["static_component_graph_diagnostics"] = graph_edge_diagnostics(
            static_adj, hierarchy_adj=hierarchy_adj
        )

    # The seed edge_index remains structural or identity. LAGTCN consumes
    # S/A/D through separate source matrices, never through an adjacency sum.
    loader.A = torch.from_numpy(graph_base_adj.astype(np.float32))
    config["adj_runtime_rebuilt"] = not uses_hierarchy_base
    config["adj_runtime_source"] = (
        "fixed_hierarchy_source" if uses_hierarchy_base else "identity_seed_adjacency"
    )
    config["base_graph_density_actual"] = fixed_seed_density

    if config["model_name"] == "LAGTCN":
        config["lagtcn_hierarchy_source_density"] = offdiag_density(hierarchy_source_adj)
        config["lagtcn_similarity_source_density"] = (
            offdiag_density(similarity_source_adj) if similarity_source_adj is not None else None
        )

    edge_indices, edge_values = dense_to_sparse(loader.A)
    loader.edges = edge_indices.detach().cpu().numpy()
    loader.edge_weights = edge_values.detach().cpu().numpy()

    logging.info(
        "Graph policy=%s | hierarchy_density=%.4f | static_threshold=%s | adaptive_top_k=%s | dynamic_threshold=%s",
        config["graph_sparsity_policy"],
        float(config["hierarchy_density"]),
        config.get("static_threshold"),
        config.get("adaptive_top_k"),
        config.get("dynamic_threshold"),
    )

    # 保存配置（在读入数据并完成节点数对齐后）
    config_path = os.path.join(config['output_dir'], artifact_filename("config"))
    config_serializable = {k: v for k, v in config.items() if not k.startswith('_')}
    with open(config_path, 'w') as f:
        json.dump(config_serializable, f, indent=4, ensure_ascii=False, allow_nan=False)
    logging.info(f"Configuration saved to {config_path}")

    dataset = loader.get_dataset(
        num_timesteps_in=int(config.get("num_timesteps_in", 7)),
        num_timesteps_out=int(config.get("num_timesteps_out", 1)),
    )

    total_snapshots = len(dataset.features)
    logging.info("Total candidate origins: %d", total_snapshots)

    (
        train_dataset,
        val_dataset,
        test_dataset,
        split_provenance,
        split_timestamp_indices,
    ) = loader.split_dataset_by_target_timestamp(
        dataset,
        train_ratio=0.8,
        validation_ratio=0.1,
    )
    config["split_protocol_version"] = split_provenance["split_protocol_version"]
    config["split_provenance"] = split_provenance
    config["validation_time_index"] = [
        str(ts) for ts in split_timestamp_indices["validation"]
    ]
    config["time_index"] = [str(ts) for ts in split_timestamp_indices["test"]]
    config["node_names"] = [str(name) for name in loader.node_names]

    logging.info("Training set size: %d", len(train_dataset.features))
    logging.info("Validation set size: %d", len(val_dataset.features))
    logging.info("Testing set size: %d", len(test_dataset.features))
    logging.info(
        "Dropped %d target-boundary origins under %s.",
        split_provenance["dropped_boundary_origin_count"],
        split_provenance["split_protocol_version"],
    )

    # Overwrite the early config snapshot now that split and hierarchy provenance are frozen.
    config_serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_serializable, f, indent=4, ensure_ascii=False, allow_nan=False)

    # 获取静态边索引
    static_edge_index = torch.tensor(loader.edges, dtype=torch.long).to(device)

    # 根据 training entry 创建相应的模型
    logging.info(
        "Creating training entry: %s | spatial_encoder=%s | temporal_method=%s | st_mode=%s",
        config['model_name'],
        config['gnn_type'],
        config['temporal_type'],
        config['st_mode'],
    )

    if config["training_loss_space"] == "normalized_log":
        supported_normalized_models = {"PATCHTST", "NHITS", "LAGTCN"}
        if config["model_name"] not in supported_normalized_models:
            raise ValueError(
                "normalized_log pilot currently supports only "
                f"{sorted(supported_normalized_models)}, got {config['model_name']}."
            )
        if abs(float(config["coherency_lambda"])) > 0.0:
            raise ValueError(
                "normalized_log loss cannot be combined with an original-scale coherence penalty."
            )

    if config['model_name'] in {"DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER"} and (
        config['gnn_type'] != "none" or config['temporal_type'] != "gru"
    ):
        logging.warning(
            "Model %s is temporal-only and ignores --gnn-type/--temporal-type. "
            "Recommended args: --gnn-type none --temporal-type gru.",
            config['model_name'],
        )
    if config['model_name'] == "LAGTCN" and config['temporal_type'] != "patch_transformer":
        raise ValueError("LAGTCN requires --temporal-type patch_transformer.")
    if config['model_name'] != "LAGTCN" and config['lagtcn_ablation'] != "none":
        raise ValueError("--lagtcn-ablation is only valid for model-name LAGTCN.")
    if config['model_name'] != "LAGTCN" and (
        config['lagtcn_decoder_mode'] != "persistence_residual"
        or config['lagtcn_residual_scale_mode'] != "unit"
        or abs(float(config['lagtcn_residual_scale_init']) - 1.0) > 1e-12
        or int(config['lagtcn_seasonal_lag']) != 24
    ):
        raise ValueError("LAGTCN decoder options are only valid for model-name LAGTCN.")
    if config['model_name'] == "LAGTCN":
        if config['lagtcn_ablation'] == "uniform_fusion" and graph_mode != "H+S+A+D":
            raise ValueError("uniform_fusion ablation is frozen to graph-mode H+S+A+D.")
    if config['model_name'] in {"DCRNN", "MTGNN"} and (
        config['gnn_type'] != "gcn" or config['temporal_type'] != "gru"
    ):
        logging.warning(
            "Model %s is a dedicated graph-temporal baseline and ignores --gnn-type/--temporal-type. "
            "Recommended args: --gnn-type gcn --temporal-type gru.",
            config['model_name'],
        )
    if config["model_name"] == "DCRNN" and config["stgnn_graph_source"] == "native":
        logging.warning(
            "Model %s has no standalone learned graph constructor in this implementation; "
            "--stgnn-graph-source native will behave like project.",
            config["model_name"],
        )

    if config['model_name'] == "DLINEAR":
        model = DLinearBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
            num_timesteps_in=config['num_timesteps_in'],
        ).to(device)
    elif config['model_name'] == "PATCHTST":
        model = PatchTSTBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
            num_timesteps_in=config['num_timesteps_in'],
        ).to(device)
    elif config['model_name'] == "NHITS":
        model = NHiTSBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
            num_timesteps_in=config['num_timesteps_in'],
        ).to(device)
    elif config['model_name'] == "ITRANSFORMER":
        model = ITransformerBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
            num_timesteps_in=config['num_timesteps_in'],
        ).to(device)
    elif config['model_name'] == "LAGTCN":
        model = LAGTCNBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
            num_timesteps_in=config['num_timesteps_in'],
            temporal_backbone=config['temporal_type'],
            use_level_awareness=config['lagtcn_ablation'] != "no_level",
            use_coevolution=config['lagtcn_ablation'] != "no_coevolution",
            learn_source_fusion=config['lagtcn_ablation'] != "uniform_fusion",
            decoder_mode=config['lagtcn_decoder_mode'],
            residual_scale_mode=config['lagtcn_residual_scale_mode'],
            residual_scale_init=config['lagtcn_residual_scale_init'],
            seasonal_lag=config['lagtcn_seasonal_lag'],
        ).to(device)
    elif config['model_name'] == "DCRNN":
        model = DCRNNBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
        ).to(device)
    elif config['model_name'] == "MTGNN":
        model = MTGNNBaseline(
            config['node_num'],
            config['input_dim'],
            config['hidden_dim'],
            config['output_dim'],
            config['num_layers'],
            loader.global_min,
            loader.global_max,
            num_timesteps_in=config['num_timesteps_in'],
            top_k=min(config['node_num'] - 1, int(config['native_top_k'])),
            stgnn_graph_source=config['stgnn_graph_source'],
        ).to(device)
    else:
        raise ValueError(f"Unknown model name: {config['model_name']}")

    config["model_architecture_version"] = getattr(
        model, "architecture_version", "legacy_or_project_specific"
    )
    if isinstance(model, DCRNNBaseline):
        config["dcrnn_filter_type"] = "dual_random_walk"
        config["dcrnn_max_diffusion_step"] = model.diffusion_steps
        config["dcrnn_cl_decay_steps"] = model.cl_decay_steps
        config["dcrnn_scheduled_sampling"] = model.use_curriculum_learning
    if isinstance(model, MTGNNBaseline):
        config["mtgnn_training_horizon_scope"] = "all_outputs_from_first_batch"

    if hasattr(model, "set_norm_params"):
        model.set_norm_params(getattr(loader, "norm_params", None))

    model.set_graph_config(
        config,
        base_adj=loader.A,
        base_edge_index=loader.edges,
        base_edge_weight=loader.edge_weights,
    )
    if isinstance(model, LAGTCNBaseline):
        model.set_static_graph_sources(
            hierarchy_adj=hierarchy_source_adj,
            similarity_adj=similarity_source_adj,
        )
    if hasattr(model, 'set_hierarchy_metadata'):
        model.set_hierarchy_metadata(
            getattr(loader, "sum_matrix", None),
            middle_levels=config.get("middle_levels"),
            bottom_start_idx=config.get("bottom_start_idx"),
        )
        config["hierarchy_level_encoding_version"] = getattr(
            model, "hierarchy_level_encoding_version", None
        )

    # Set spatio-temporal interaction mode.
    if hasattr(model, 'set_st_mode'):
        model.set_st_mode(config.get('st_mode', 'sequential'))

    # Pass sum_matrix to config for coherency loss (not serialized to JSON)
    config['_sum_matrix'] = loader.sum_matrix
    final_config = {k: v for k, v in config.items() if not k.startswith("_")}
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=4, ensure_ascii=False, allow_nan=False)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config['lr'],
    )
    criterion = nn.SmoothL1Loss()

    # 训练模型
    logging.info("Starting training...")
    train_losses, val_losses, train_time, alpha_values, train_efficiency = train_model(
        model, train_dataset, val_dataset, static_edge_index, optimizer, criterion, config, device
    )

    # 保存训练结果
    training_results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_time': train_time
    }
    training_results.update(train_efficiency)
    if alpha_values:
        training_results['alpha_values'] = alpha_values

    training_results_path = os.path.join(
        config['output_dir'], artifact_filename("training_results")
    )
    with open(training_results_path, 'w', encoding="utf-8") as f:
        results_json = {}
        for k, vals in training_results.items():
            if isinstance(vals, list):
                # 列表（loss 曲线等）——逐个转成 float
                results_json[k] = [float(v) for v in vals]
            elif isinstance(vals, (int, float, np.floating)):
                # 单个数值——直接转 float
                results_json[k] = float(vals)
            else:
                # 其它类型（None、字符串等）——原样写入，避免 float(None) 报错
                results_json[k] = vals

        json.dump(results_json, f, indent=4, allow_nan=False)

    if not config.get('no_plots', False):
        plot_loss_curves(train_losses, val_losses, config)

    # 加载最佳模型进行评估
    logging.info("Loading best model for evaluation...")
    best_model_path_obj = find_artifact(config['output_dir'], "best_model", config['model_name'], config['timestamp'])
    best_model_path = str(best_model_path_obj or os.path.join(config['output_dir'], artifact_filename("best_model")))
    best_checkpoint_metadata = load_best_model_strict(
        model,
        best_model_path,
        config,
        device,
    )
    training_results["best_checkpoint_metadata"] = best_checkpoint_metadata
    results_json["best_checkpoint_metadata"] = best_checkpoint_metadata
    if isinstance(model, LAGTCNBaseline):
        decoder_metadata = model.get_decoder_metadata()
        training_results["lagtcn_decoder"] = decoder_metadata
        results_json["lagtcn_decoder"] = decoder_metadata
    with open(training_results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=4, ensure_ascii=False, allow_nan=False)

    logging.info("Saving frozen validation predictions for MinT-SHR weights...")
    validation_predictions, validation_true_values, validation_metrics = evaluate_model(
        model, val_dataset, static_edge_index, criterion, device, config
    )
    validation_config = dict(config)
    validation_config["time_index"] = list(config["validation_time_index"])
    save_predictions(
        validation_predictions,
        validation_true_values,
        validation_config,
        filename="validation_pred.csv",
        true_filename="validation_true.csv",
    )
    with open(os.path.join(config["output_dir"], "validation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(validation_metrics, f, indent=4, ensure_ascii=False, allow_nan=False)

    if config.get("validation_only", False):
        logging.info("Validation-only run complete; test partition was not evaluated.")
        return

    # Evaluate test exactly once after the best checkpoint is validated.
    logging.info("Evaluating model...")
    predictions, true_values, metrics = evaluate_model(
        model, test_dataset, static_edge_index, criterion, device, config
    )

    coherency_metrics = compute_coherency_violation(
        predictions,
        loader.sum_matrix,
        bottom_start_idx=loader.bottom_start_idx,
    )
    metrics.update(coherency_metrics)
    inference_profile = benchmark_inference(
        model,
        test_dataset,
        static_edge_index,
        device=device,
    )
    metrics.update(inference_profile)
    true_coherency = compute_coherency_violation(
        true_values,
        loader.sum_matrix,
        bottom_start_idx=loader.bottom_start_idx,
    )
    metrics["coherency_mae_true"] = float(true_coherency["coherency_mae"])
    metrics["coherency_rmse_true"] = float(true_coherency["coherency_rmse"])

    # 保存整体评估指标（JSON）
    metrics_json_path = os.path.join(config['output_dir'], artifact_filename("metrics"))
    with open(metrics_json_path, 'w') as f:
        # 将 numpy 数值转换为 Python 原生类型，避免 json 序列化报错
        metrics_json = {
            k: float(v) if isinstance(v, (np.floating, float, int)) else v
            for k, v in metrics.items()
        }
        json.dump(metrics_json, f, indent=4, allow_nan=False)
    logging.info(f"Overall metrics saved to {metrics_json_path}")

    graph_info_path = _save_graph_info(
        model=model,
        config=config,
        loader=loader,
        hierarchy_adj=hierarchy_adj,
        static_component_adj=similarity_source_adj,
        test_dataset=test_dataset,
        static_edge_index=static_edge_index,
        device=device,
    )
    logging.info("Graph information saved to %s", graph_info_path)

    # 计算各层级的指标
    level_metrics = calculate_level_metrics(predictions, true_values, config)

    # Save the canonical output and the base-forecast artifact consumed by reconciliation.
    save_predictions(predictions, true_values, config)
    save_predictions(predictions, true_values, config, filename=BASE_PRED_FILENAME)
    if not config.get('no_plots', False):
        num_nodes = predictions.shape[1]
        plot_node_limit = config.get('plot_node_limit')
        num_plot_nodes = num_nodes if plot_node_limit is None else min(num_nodes, int(plot_node_limit))
        if num_plot_nodes < num_nodes:
            logging.info("Plotting first %d/%d node prediction figures.", num_plot_nodes, num_nodes)
        for node_idx in range(num_plot_nodes):
            plot_predictions(predictions, true_values, node_idx, config)

    # 保存模型信息（参数量 + 配置 + 训练/测试指标）
    save_model_info(model, config, metrics, level_metrics, training_results)

    logging.info("Training and evaluation completed!")


# 运行程序
if __name__ == "__main__":
    main()
