#!/usr/bin/env python3
"""Run B x R post-hoc reconciliation matrix on existing base forecast runs.

B: base forecasters (from existing model_info outputs)
R: reconciliation methods (none / bu / td / mint)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT / "code"))
from metrics import _build_level_indices, compute_coherency_violation, compute_mase  # noqa: E402
from output_naming import (  # noqa: E402
    namespace_matches,
    normalize_lagtcn_graph_source_version,
)
from reconcile_posthoc import apply_reconciliation  # noqa: E402
from graph_sparsity import FINAL_GRAPH_SOURCE_POLICY  # noqa: E402


DEFAULT_DATASETS = [
    "GEFCom2012_2level",
    "GEFCom2017QualifyingMatch_3level",
    "GEFCom2017FinalMatch_4level",
]
DEFAULT_BASE_MODELS = [
    "GCN-GRU-LP-NO",
    "DLINEAR",
    "PATCHTST",
    "NHITS",
    "TIMESNET",
    "ITRANSFORMER",
    "GRAPH_DLINEAR",
    "GRAPH_PATCHTST",
    "GRAPH_ITRANSFORMER",
    "GRAPH_ADAPTER",
    "LAGTCN",
    "DCRNN",
    "STGCN",
    "GWNET",
    "MTGNN",
    "AGCRN",
]
DEFAULT_RECON_METHODS = ["none", "bu", "td", "mint"]
METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "MAPE",
    "WAPE",
    "MASE",
    "coherency_mae",
    "coherency_rmse",
    "coherency_nmae_pct",
]
LEVEL_METRIC_COLUMNS = ["MAE", "RMSE", "MAPE", "WAPE", "MASE"]
STRONG_TEMPORAL_MODELS = {"DLINEAR", "PATCHTST", "NHITS", "TIMESNET"}
EXTRA_TEMPORAL_MODELS = {"ITRANSFORMER"}
GRAPH_NATIVE_TEMPORAL_MODELS = {"GRAPH_DLINEAR", "GRAPH_PATCHTST", "GRAPH_ITRANSFORMER", "GRAPH_ADAPTER", "LAGTCN"}
DEDICATED_GRAPH_MODELS = {"DCRNN", "STGCN", "GWNET", "MTGNN", "AGCRN"}
DENSITY_COLUMNS = [
    "graph_sparsity_policy",
    "graph_protocol_version",
    "graph_design_protocol_version",
    "hierarchy_density",
    "physical_hierarchy_density",
    "static_threshold",
    "adaptive_top_k",
    "dynamic_threshold",
    "static_component_density_actual",
    "base_graph_density_actual",
]


def _model_family(model_name: str | None) -> str:
    model = str(model_name or "").upper().split("+", 1)[0]
    if model in STRONG_TEMPORAL_MODELS | EXTRA_TEMPORAL_MODELS:
        return "temporal-only"
    if model in GRAPH_NATIVE_TEMPORAL_MODELS:
        return "graph-enhanced-temporal"
    if model in DEDICATED_GRAPH_MODELS:
        return "dedicated-STGNN"
    if model == "GCN-GRU-LP-NO":
        return "GCN-GRU reference"
    if model.startswith("GCN-GRU-LP-"):
        return "GCN-GRU reconciliation"
    return "other"


def _is_graph_based_family(family: str) -> bool:
    return family in {
        "graph-enhanced-temporal",
        "dedicated-STGNN",
        "GCN-GRU reference",
        "GCN-GRU reconciliation",
    }


def _density_metadata_from_config(cfg: dict) -> dict:
    return {
        "graph_sparsity_policy": cfg.get("graph_sparsity_policy"),
        "graph_protocol_version": cfg.get("graph_protocol_version"),
        "graph_design_protocol_version": cfg.get("graph_design_protocol_version"),
        "hierarchy_density": _to_float(cfg.get("hierarchy_density")),
        "physical_hierarchy_density": _to_float(cfg.get("physical_hierarchy_density")),
        "static_threshold": _to_float(cfg.get("static_threshold")),
        "adaptive_top_k": _to_float(cfg.get("adaptive_top_k")),
        "dynamic_threshold": _to_float(cfg.get("dynamic_threshold")),
        "static_component_density_actual": _to_float(cfg.get("static_component_density_actual")),
        "base_graph_density_actual": _to_float(cfg.get("base_graph_density_actual")),
    }


def _best_val_loss(payload: dict) -> float | None:
    training = payload.get("training_results", {}) or {}
    val_losses = training.get("val_losses")
    if not val_losses:
        return None
    vals = [_to_float(v) for v in val_losses]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _metric_score(payload: dict, metric_name: str) -> float | None:
    metric_key = str(metric_name).strip()
    normalized = metric_key.lower().replace("-", "_")
    if normalized in {"val_loss", "best_val_loss", "validation_loss"}:
        return _best_val_loss(payload)
    metrics = payload.get("metrics", {}) or {}
    return _to_float(metrics.get(metric_key))


def _base_choice_key(cfg: dict, dataset: str, base_model: str, horizon: int) -> tuple:
    return (
        dataset,
        int(horizon),
        str(base_model).upper(),
        str(cfg.get("graph_mode", "")).upper(),
        str(cfg.get("gnn_type", "gcn")).lower(),
        str(cfg.get("temporal_type", "gru")).lower(),
        str(cfg.get("st_mode", "sequential")).lower(),
        cfg.get("stgnn_graph_source"),
        normalize_lagtcn_graph_source_version(
            base_model,
            cfg.get("lagtcn_graph_source_version"),
        ),
        cfg.get("graph_sparsity_policy"),
        _to_float(cfg.get("static_threshold")),
        _to_float(cfg.get("adaptive_top_k")),
        _to_float(cfg.get("dynamic_threshold")),
    )


def _select_top_base_keys(
    model_info_files: list[Path],
    args: argparse.Namespace,
    datasets: set[str],
    base_models: set[str],
    graph_modes: set[str] | None,
    gnn_types: set[str] | None,
    temporal_types: set[str] | None,
    horizons: set[int] | None,
    seeds: set[int] | None,
) -> set[tuple]:
    scores = defaultdict(lambda: defaultdict(list))
    requested_policy = str(args.graph_sparsity_policy).strip()
    for info_path in model_info_files:
        try:
            payload = _read_json(info_path)
        except Exception:
            continue
        cfg = payload.get("config", {})
        base_model = str(cfg.get("model_name", payload.get("model_name", ""))).upper()
        if base_model not in base_models:
            continue
        base_family = _model_family(base_model)
        graph_sparsity_policy = cfg.get("graph_sparsity_policy")
        if (
            _is_graph_based_family(base_family)
            and requested_policy
            and str(graph_sparsity_policy) != requested_policy
        ):
            continue
        if args.paper_scope and str(cfg.get("paper_scope")) != args.paper_scope:
            continue
        if args.experiment_stage and str(cfg.get("experiment_stage")) != args.experiment_stage:
            continue
        if args.experiment_id and str(cfg.get("experiment_id")) != args.experiment_id:
            continue
        if args.output_namespace_prefix and not namespace_matches(cfg.get("output_namespace"), args.output_namespace_prefix):
            continue

        graph_mode = str(cfg.get("graph_mode", "")).upper()
        gnn_type = str(cfg.get("gnn_type", "gcn")).lower()
        temporal_type = str(cfg.get("temporal_type", "gru")).lower()
        horizon = int(cfg.get("num_timesteps_out", cfg.get("output_dim", 1)))
        seed = int(cfg.get("seed", -999))
        if graph_modes and graph_mode not in graph_modes:
            continue
        if gnn_types and gnn_type not in gnn_types:
            continue
        if temporal_types and temporal_type not in temporal_types:
            continue
        if horizons and horizon not in horizons:
            continue
        if seeds and seed not in seeds:
            continue

        raw_data_dir = Path(str(cfg.get("raw_data_dir", "")))
        dataset = str(cfg.get("dataset") or (raw_data_dir.name if raw_data_dir.name else ""))
        if datasets and dataset not in datasets:
            continue
        score = _metric_score(payload, args.selection_metric)
        if score is None:
            continue
        group_key = (dataset, horizon, base_family) if args.select_topk_per_family else (dataset, horizon)
        scores[group_key][_base_choice_key(cfg, dataset, base_model, horizon)].append(score)

    selected: set[tuple] = set()
    for group_key, choice_scores in sorted(scores.items()):
        ranked = [
            (sum(vals) / len(vals), -len(vals), choice_key)
            for choice_key, vals in choice_scores.items()
            if vals
        ]
        ranked.sort()
        for _, _, choice_key in ranked[:max(1, int(args.select_topk))]:
            selected.add(choice_key)
        print(
            f"[INFO] Selected top-{args.select_topk} base choices for {group_key}: "
            f"{len(ranked[:max(1, int(args.select_topk))])}"
        )
    return selected


def _parse_csv_list(value: str | None, cast=str) -> list:
    if value is None:
        return []
    out = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(cast(token))
    return out


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return float(math.sqrt(var))


def _parse_prediction_csv(path: Path) -> tuple[np.ndarray, pd.Index, list[str]]:
    df = pd.read_csv(path, index_col=0)
    cols = list(df.columns)
    if not cols:
        raise ValueError(f"No columns in prediction file: {path}")

    multi_h = any("_t+" in c for c in cols)
    if not multi_h:
        arr = df.to_numpy(dtype=np.float64)
        return arr[:, :, None], df.index, cols

    pat = re.compile(r"^(.*)_t\+([0-9]+)$")
    parsed = []
    node_order = []
    horizon_order = []
    node_seen = set()
    horizon_seen = set()
    for c in cols:
        m = pat.match(str(c))
        if not m:
            raise ValueError(f"Unexpected multi-horizon column name '{c}' in {path}")
        node_name = m.group(1)
        h = int(m.group(2))
        parsed.append((c, node_name, h))
        if node_name not in node_seen:
            node_seen.add(node_name)
            node_order.append(node_name)
        if h not in horizon_seen:
            horizon_seen.add(h)
            horizon_order.append(h)

    horizon_order = sorted(horizon_order)
    node_to_idx = {n: i for i, n in enumerate(node_order)}
    h_to_idx = {h: i for i, h in enumerate(horizon_order)}
    arr = np.zeros((len(df), len(node_order), len(horizon_order)), dtype=np.float64)
    for c, node_name, h in parsed:
        arr[:, node_to_idx[node_name], h_to_idx[h]] = df[c].to_numpy(dtype=np.float64)
    return arr, df.index, node_order


def _write_prediction_csv(path: Path, pred: np.ndarray, index: Iterable, node_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim == 2:
        pred = pred[..., None]
    if pred.ndim != 3:
        raise ValueError(f"pred must be 2D/3D, got shape {pred.shape}")
    num_samples, num_nodes, num_h = pred.shape
    if len(node_names) != num_nodes:
        node_names = [f"node_{i}" for i in range(num_nodes)]

    if num_h == 1:
        out_df = pd.DataFrame(pred[:, :, 0], index=index, columns=node_names)
    else:
        cols = [f"{node}_t+{h+1}" for h in range(num_h) for node in node_names]
        out_df = pd.DataFrame(
            pred.transpose(0, 2, 1).reshape(num_samples, num_h * num_nodes),
            index=index,
            columns=cols,
        )
    out_df.to_csv(path)


def _level_groups(config: dict, num_nodes: int) -> list[tuple[str, list[int]]]:
    top_level, middle_levels, bottom_level = _build_level_indices(config, num_nodes)
    groups = [("All", list(range(num_nodes))), ("top_level", top_level)]
    for i, indices in enumerate(middle_levels, start=1):
        groups.append((f"middle{i}_level", indices))
    groups.append(("bottom_level", bottom_level))
    return [(name, indices) for name, indices in groups if indices]


def _compute_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: dict,
    epsilon: float = 1e-3,
) -> list[dict]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    if y_true.ndim == 2:
        y_true = y_true[..., None]
        y_pred = y_pred[..., None]

    num_nodes = y_true.shape[1]
    num_horizons = y_true.shape[-1]
    num_timesteps_in = int(config.get("num_timesteps_in", 24))
    rows = []

    def _metrics_for_slice(level_name: str, horizon_label: str, yt: np.ndarray, yp: np.ndarray) -> dict:
        yt_flat = yt.reshape(-1)
        yp_flat = yp.reshape(-1)
        return {
            "level": level_name,
            "horizon_label": horizon_label,
            "MAE": float(np.mean(np.abs(yt_flat - yp_flat))),
            "RMSE": float(np.sqrt(np.mean((yt_flat - yp_flat) ** 2))),
            "MAPE": float(np.mean(np.abs((yt_flat - yp_flat) / np.maximum(np.abs(yt_flat), epsilon))) * 100.0),
            "WAPE": float(np.sum(np.abs(yt_flat - yp_flat)) / np.maximum(np.sum(np.abs(yt_flat)), epsilon) * 100.0),
            "MASE": float(compute_mase(yt, yp, num_timesteps_in=num_timesteps_in)),
        }

    for level_name, indices in _level_groups(config, num_nodes):
        level_true = y_true[:, indices, :]
        level_pred = y_pred[:, indices, :]
        rows.append(_metrics_for_slice(level_name, "all", level_true, level_pred))
        if num_horizons > 1:
            for h in range(num_horizons):
                rows.append(
                    _metrics_for_slice(
                        level_name,
                        f"h{h + 1}",
                        level_true[:, :, h:h + 1],
                        level_pred[:, :, h:h + 1],
                    )
                )
    return rows


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_timesteps_in: int = 24, epsilon: float = 1e-3) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    if y_true.ndim == 2:
        y_true = y_true[..., None]
        y_pred = y_pred[..., None]

    y_t = y_true.reshape(-1)
    y_p = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_t - y_p)))
    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mape = float(np.mean(np.abs((y_t - y_p) / np.maximum(np.abs(y_t), epsilon))) * 100.0)
    wape = float(np.sum(np.abs(y_t - y_p)) / np.maximum(np.sum(np.abs(y_t)), epsilon) * 100.0)
    mase = float(compute_mase(y_true, y_pred, num_timesteps_in=num_timesteps_in))

    out = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "WAPE": wape,
        "MASE": mase,
    }
    if y_true.shape[-1] > 1:
        for h in range(y_true.shape[-1]):
            h_true = y_true[:, :, h:h + 1]
            h_pred = y_pred[:, :, h:h + 1]
            h_t = h_true.reshape(-1)
            h_p = h_pred.reshape(-1)
            out[f"h{h+1}_MAE"] = float(np.mean(np.abs(h_t - h_p)))
            out[f"h{h+1}_RMSE"] = float(np.sqrt(np.mean((h_t - h_p) ** 2)))
            out[f"h{h+1}_WAPE"] = float(
                np.sum(np.abs(h_t - h_p)) / np.maximum(np.sum(np.abs(h_t)), epsilon) * 100.0
            )
            out[f"h{h+1}_MASE"] = float(compute_mase(h_true, h_pred, num_timesteps_in=num_timesteps_in))
    return out


def _find_model_info_files(data_root: Path, datasets: list[str]) -> list[Path]:
    files = []
    for dataset in datasets:
        ds_root = data_root / dataset / "output"
        if not ds_root.exists():
            continue
        files.extend(sorted(_iter_model_info_files(ds_root)))
    return files




def _iter_model_info_files(root: Path):
    seen = set()
    for pattern in ("**/model_info.json", "**/model_info_*.json"):
        for path in sorted(root.glob(pattern)):
            if path not in seen:
                seen.add(path)
                yield path


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _timestamp_rank(payload: dict, path: Path) -> tuple[int, float]:
    cfg = payload.get("config", {}) or {}
    timestamp = str(cfg.get("timestamp", payload.get("timestamp", "")))
    digits = re.sub(r"\D", "", timestamp)
    if len(digits) == 12:
        digits = f"20{digits}"
    elif len(digits) == 8:
        digits = f"{digits}000000"
    elif len(digits) > 14:
        digits = digits[:14]
    rank = int(digits) if digits.isdigit() and len(digits) == 14 else 0
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return rank, mtime


def _model_info_instance_key(path: Path, payload: dict) -> tuple:
    cfg = payload.get("config", {}) or {}
    raw_data_dir = Path(str(cfg.get("raw_data_dir", "")))
    dataset = str(cfg.get("dataset") or (raw_data_dir.name if raw_data_dir.name else ""))
    base_model = str(cfg.get("model_name", payload.get("model_name", ""))).upper()
    try:
        horizon = int(cfg.get("num_timesteps_out", cfg.get("output_dim", 1)))
    except (TypeError, ValueError):
        horizon = None
    try:
        num_timesteps_in = int(cfg.get("num_timesteps_in", 24))
    except (TypeError, ValueError):
        num_timesteps_in = None
    return (
        dataset,
        str(cfg.get("paper_scope")),
        str(cfg.get("experiment_stage")),
        str(cfg.get("output_namespace")),
        str(cfg.get("feature_set") or "target"),
        base_model,
        str(cfg.get("seed")),
        num_timesteps_in,
        horizon,
        str(cfg.get("graph_mode", "")).upper(),
        str(cfg.get("gnn_type", "gcn")).lower(),
        str(cfg.get("temporal_type", "gru")).lower(),
        str(cfg.get("st_mode", "sequential")).lower(),
        cfg.get("stgnn_graph_source"),
        normalize_lagtcn_graph_source_version(
            base_model,
            cfg.get("lagtcn_graph_source_version"),
        ),
        cfg.get("graph_sparsity_policy"),
        _to_float(cfg.get("static_threshold")),
        _to_float(cfg.get("adaptive_top_k")),
        _to_float(cfg.get("dynamic_threshold")),
    )


def _dedupe_latest_model_info_files(model_info_files: list[Path]) -> list[Path]:
    latest: dict[tuple, tuple[tuple[int, float], Path]] = {}
    passthrough: list[Path] = []
    for info_path in model_info_files:
        try:
            payload = _read_json(info_path)
            key = _model_info_instance_key(info_path, payload)
            rank = _timestamp_rank(payload, info_path)
        except Exception:
            passthrough.append(info_path)
            continue
        current = latest.get(key)
        if current is None or rank > current[0]:
            latest[key] = (rank, info_path)
    deduped = [path for _, path in latest.values()] + passthrough
    return sorted(deduped)


def _locate_prediction_files(
    output_dir: Path,
    model_name: str,
    timestamp: str,
    fallback_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    candidate_dirs = [output_dir]
    if fallback_dir is not None and fallback_dir != output_dir:
        candidate_dirs.append(fallback_dir)

    checked = []
    for root in candidate_dirs:
        checked.append(str(root))
        short_pred = root / "pred.csv"
        short_true = root / "true.csv"
        if short_pred.exists() and short_true.exists():
            return short_pred, short_true, root

        pred = root / f"predictions_{model_name}_{timestamp}.csv"
        true = root / f"true_values_{model_name}_{timestamp}.csv"
        if pred.exists() and true.exists():
            return pred, true, root

        preds = sorted(root.glob("predictions_*.csv"))
        trues = sorted(root.glob("true_values_*.csv"))
        if len(preds) == 1 and len(trues) == 1:
            return preds[0], trues[0], root

    raise FileNotFoundError(
        "Cannot locate prediction/true files under any candidate output directory: "
        f"{checked}"
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        return
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_group_key(record: dict, group_fields: list[str]) -> tuple:
    values = []
    for field in group_fields:
        if field == "lagtcn_graph_source_version":
            values.append(
                normalize_lagtcn_graph_source_version(
                    record.get("base_model"),
                    record.get(field),
                )
            )
        else:
            values.append(record.get(field))
    return tuple(values)


def _summarize(
    records: list[dict],
    group_fields: list[str],
    metric_columns: list[str] | None = None,
) -> list[dict]:
    if metric_columns is None:
        metric_columns = METRIC_COLUMNS
    group_map = defaultdict(list)
    for rec in records:
        key = _summary_group_key(rec, group_fields)
        group_map[key].append(rec)

    rows = []
    for key, items in sorted(
        group_map.items(),
        key=lambda item: tuple("" if value is None else str(value) for value in item[0]),
    ):
        out = {k: v for k, v in zip(group_fields, key)}
        out["n_runs"] = len(items)
        for metric in metric_columns:
            vals = [r.get(metric) for r in items if r.get(metric) is not None]
            out[f"{metric}_mean"] = _mean(vals)
            out[f"{metric}_std"] = _std(vals)
        rows.append(out)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run post-hoc BxR reconciliation matrix.")
    parser.add_argument("--data-root", type=str, default="Data")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets to scan.",
    )
    parser.add_argument(
        "--base-models",
        type=str,
        default=",".join(DEFAULT_BASE_MODELS),
        help="Comma-separated base model names to include.",
    )
    parser.add_argument(
        "--reconcile-methods",
        type=str,
        default=",".join(DEFAULT_RECON_METHODS),
        help="Comma-separated methods: none,bu,td,mint.",
    )
    parser.add_argument("--graph-modes", type=str, default=None, help="Optional graph_mode filter.")
    parser.add_argument("--gnn-types", type=str, default=None, help="Optional gnn_type filter.")
    parser.add_argument("--temporal-types", type=str, default=None, help="Optional temporal_type filter.")
    parser.add_argument("--horizons", type=str, default=None, help="Optional horizons filter, e.g. 1,6,24.")
    parser.add_argument("--seeds", type=str, default=None, help="Optional seed filter, e.g. 42,43,44.")
    parser.add_argument("--paper-scope", type=str, default=None, help="Optional paper_scope filter.")
    parser.add_argument("--experiment-stage", type=str, default=None, help="Optional experiment_stage filter.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Optional experiment_id filter.")
    parser.add_argument(
        "--graph-sparsity-policy",
        type=str,
        default=FINAL_GRAPH_SOURCE_POLICY,
        choices=[FINAL_GRAPH_SOURCE_POLICY],
        help="Graph protocol filter for base runs.",
    )
    parser.add_argument(
        "--output-namespace-prefix",
        type=str,
        default=None,
        help="Optional output_namespace prefix filter.",
    )
    parser.add_argument(
        "--td-mode",
        type=str,
        default="forecast_proportions",
        choices=["forecast_proportions", "average_proportions"],
    )
    parser.add_argument(
        "--mint-cov-mode",
        type=str,
        default="identity",
        choices=["identity", "diag", "sample", "shrink"],
        help=(
            "Covariance mode for MinT. Saved test predictions do not contain "
            "training residuals, so only identity is leakage-free in this script."
        ),
    )
    parser.add_argument(
        "--select-topk",
        type=int,
        default=None,
        help=(
            "Optional Stage-D selector: keep top-K base forecaster configurations "
            "per dataset+horizon before running post-hoc reconciliation."
        ),
    )
    parser.add_argument(
        "--select-topk-per-family",
        action="store_true",
        help="Apply --select-topk separately within each base model family.",
    )
    parser.add_argument(
        "--selection-metric",
        type=str,
        default="val_loss",
        help="Metric for --select-topk. Default val_loss avoids test-set model selection.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/reconcile_matrix",
        help="Output directory for raw and summary CSV files.",
    )
    parser.add_argument("--save-reconciled", action="store_true", help="Persist reconciled prediction CSV files.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for debug.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = _parse_csv_list(args.datasets, cast=str)
    base_models = {m.upper() for m in _parse_csv_list(args.base_models, cast=str)}
    methods = [m.lower() for m in _parse_csv_list(args.reconcile_methods, cast=str)]
    graph_modes = {m.upper() for m in _parse_csv_list(args.graph_modes, cast=str)} if args.graph_modes else None
    gnn_types = {m.lower() for m in _parse_csv_list(args.gnn_types, cast=str)} if args.gnn_types else None
    temporal_types = {m.lower() for m in _parse_csv_list(args.temporal_types, cast=str)} if args.temporal_types else None
    horizons = {int(h) for h in _parse_csv_list(args.horizons, cast=int)} if args.horizons else None
    seeds = {int(seed) for seed in _parse_csv_list(args.seeds, cast=int)} if args.seeds else None

    model_info_files = _find_model_info_files(data_root, datasets)
    print(f"[INFO] Found model_info files: {len(model_info_files)}")
    deduped_model_info_files = _dedupe_latest_model_info_files(model_info_files)
    if len(deduped_model_info_files) != len(model_info_files):
        print(
            f"[INFO] Deduplicated model_info files: {len(model_info_files)} -> "
            f"{len(deduped_model_info_files)}"
        )
    model_info_files = deduped_model_info_files

    selected_base_keys = None
    if args.select_topk is not None:
        selected_base_keys = _select_top_base_keys(
            model_info_files=model_info_files,
            args=args,
            datasets=set(datasets),
            base_models=base_models,
            graph_modes=graph_modes,
            gnn_types=gnn_types,
            temporal_types=temporal_types,
            horizons=horizons,
            seeds=seeds,
        )
        print(f"[INFO] Selected base choice keys: {len(selected_base_keys)}")

    run_count = 0
    raw_rows = []
    level_raw_rows = []
    skipped = []

    for info_path in model_info_files:
        try:
            payload = _read_json(info_path)
            cfg = payload.get("config", {})
            base_model = str(cfg.get("model_name", payload.get("model_name", ""))).upper()
            if base_model not in base_models:
                continue
            base_model_family = _model_family(base_model)
            graph_sparsity_policy = cfg.get("graph_sparsity_policy")
            requested_policy = str(args.graph_sparsity_policy).strip()
            if (
                _is_graph_based_family(base_model_family)
                and requested_policy
                and str(graph_sparsity_policy) != requested_policy
            ):
                continue

            paper_scope = cfg.get("paper_scope")
            experiment_stage = cfg.get("experiment_stage")
            experiment_id = cfg.get("experiment_id")
            output_namespace = cfg.get("output_namespace")
            if args.paper_scope and str(paper_scope) != args.paper_scope:
                continue
            if args.experiment_stage and str(experiment_stage) != args.experiment_stage:
                continue
            if args.experiment_id and str(experiment_id) != args.experiment_id:
                continue
            if args.output_namespace_prefix and not namespace_matches(output_namespace, args.output_namespace_prefix):
                continue

            graph_mode = str(cfg.get("graph_mode", "")).upper()
            gnn_type = str(cfg.get("gnn_type", "gcn")).lower()
            temporal_type = str(cfg.get("temporal_type", "gru")).lower()
            horizon = int(cfg.get("num_timesteps_out", cfg.get("output_dim", 1)))
            seed = int(cfg.get("seed", -999))
            if graph_modes and graph_mode not in graph_modes:
                continue
            if gnn_types and gnn_type not in gnn_types:
                continue
            if temporal_types and temporal_type not in temporal_types:
                continue
            if horizons and horizon not in horizons:
                continue
            if seeds and seed not in seeds:
                continue

            raw_data_dir = Path(str(cfg.get("raw_data_dir", "")))
            dataset = str(cfg.get("dataset") or (raw_data_dir.name if raw_data_dir.name else ""))
            if datasets and dataset not in datasets:
                continue
            choice_key = _base_choice_key(cfg, dataset, base_model, horizon)
            if selected_base_keys is not None and choice_key not in selected_base_keys:
                continue

            configured_output_dir = Path(str(cfg.get("output_dir", info_path.parent)))
            timestamp = str(cfg.get("timestamp", payload.get("timestamp", "")))
            pred_path, true_path, output_dir = _locate_prediction_files(
                configured_output_dir,
                base_model,
                timestamp,
                fallback_dir=info_path.parent,
            )
            sum_matrix_path = raw_data_dir / "sum_matrix.csv"
            if not sum_matrix_path.exists():
                local_data_dir = PROJECT_ROOT / "Data" / dataset
                local_sum_matrix_path = local_data_dir / "sum_matrix.csv"
                if local_sum_matrix_path.exists():
                    raw_data_dir = local_data_dir
                    sum_matrix_path = local_sum_matrix_path
                else:
                    raise FileNotFoundError(
                        f"Missing sum_matrix.csv: {sum_matrix_path} "
                        f"(fallback checked: {local_sum_matrix_path})"
                    )

            if args.dry_run:
                print(
                    f"[DRY-RUN] dataset={dataset} model={base_model} seed={cfg.get('seed')} "
                    f"family={base_model_family} graph={graph_mode} "
                    f"policy={graph_sparsity_policy} horizon={horizon} file={info_path}"
                )
                run_count += 1
                if args.max_runs and run_count >= args.max_runs:
                    break
                continue

            y_hat, index, node_names = _parse_prediction_csv(pred_path)
            y_true, _, _ = _parse_prediction_csv(true_path)
            if y_hat.shape != y_true.shape:
                raise ValueError(f"Prediction/true shape mismatch: {y_hat.shape} vs {y_true.shape}")
            S = pd.read_csv(sum_matrix_path, header=None).to_numpy(dtype=np.float64)
            bottom_start_idx = cfg.get("bottom_start_idx")
            num_timesteps_in = int(cfg.get("num_timesteps_in", 24))

            for method in methods:
                if method == "mint" and args.mint_cov_mode != "identity":
                    skipped.append({
                        "path": str(info_path),
                        "method": method,
                        "error": (
                            "Skipped non-identity MinT because this post-hoc script only has "
                            "saved test predictions/targets. Estimating covariance from test "
                            "targets would leak test information."
                        ),
                    })
                    continue

                y_rec = apply_reconciliation(
                    method=method,
                    base_predictions=y_hat,
                    sum_matrix=S,
                    bottom_start_idx=bottom_start_idx,
                    td_mode=args.td_mode,
                    mint_cov_mode=args.mint_cov_mode,
                )

                metrics = _compute_metrics(y_true=y_true, y_pred=y_rec, num_timesteps_in=num_timesteps_in)
                metrics.update(
                    compute_coherency_violation(
                        y_rec,
                        S,
                        bottom_start_idx=int(bottom_start_idx) if bottom_start_idx is not None else None,
                    )
                )

                row = {
                    "dataset": dataset,
                    "paper_scope": paper_scope,
                    "experiment_stage": experiment_stage,
                    "experiment_id": experiment_id,
                    "output_namespace": output_namespace,
                    "run_label": cfg.get("run_label"),
                    "model_info_path": str(info_path),
                    "output_dir": str(output_dir),
                    "seed": cfg.get("seed"),
                    "model_family": "posthoc-reconciliation",
                    "base_model": base_model,
                    "base_model_family": base_model_family,
                    "reconcile_method": method,
                    "graph_mode": graph_mode,
                    "sim_type": cfg.get("sim_type"),
                    "gnn_type": gnn_type,
                    "temporal_type": temporal_type,
                    "st_mode": cfg.get("st_mode", "sequential"),
                    "stgnn_graph_source": cfg.get("stgnn_graph_source"),
                    "lagtcn_graph_source_version": normalize_lagtcn_graph_source_version(
                        base_model,
                        cfg.get("lagtcn_graph_source_version"),
                    ),
                    "num_timesteps_in": cfg.get("num_timesteps_in"),
                    "num_timesteps_out": horizon,
                    "td_mode": args.td_mode if method == "td" else None,
                    "mint_cov_mode": args.mint_cov_mode if method == "mint" else None,
                    **_density_metadata_from_config(cfg),
                }
                for metric, value in metrics.items():
                    row[metric] = _to_float(value)
                raw_rows.append(row)

                for level_metrics in _compute_level_metrics(y_true=y_true, y_pred=y_rec, config=cfg):
                    level_row = {
                        k: v
                        for k, v in row.items()
                        if k not in METRIC_COLUMNS and not re.match(r"^h\d+_", str(k))
                    }
                    level_row.update(level_metrics)
                    level_raw_rows.append(level_row)

                if args.save_reconciled and method != "none":
                    pred_out = (
                        output_dir
                        / "posthoc_reconcile"
                        / method
                        / f"rec_{method}.csv"
                    )
                    _write_prediction_csv(pred_out, y_rec, index=index, node_names=node_names)

            run_count += 1
            if args.max_runs and run_count >= args.max_runs:
                break
        except Exception as exc:
            skipped.append({"path": str(info_path), "error": str(exc)})

    if args.dry_run:
        print(f"[INFO] Dry-run selected base runs: {run_count}")
        return

    raw_path = out_dir / "reconcile_matrix_runs_raw.csv"
    _write_csv(raw_path, raw_rows)
    level_raw_path = out_dir / "reconcile_matrix_level_raw.csv"
    _write_csv(level_raw_path, level_raw_rows)

    detailed_group = [
        "paper_scope",
        "experiment_stage",
        "model_family",
        "base_model_family",
        "dataset",
        "num_timesteps_out",
        "graph_sparsity_policy",
        "graph_design_protocol_version",
        "static_threshold",
        "adaptive_top_k",
        "dynamic_threshold",
        "graph_mode",
        "lagtcn_graph_source_version",
        "stgnn_graph_source",
        "base_model",
        "gnn_type",
        "temporal_type",
        "st_mode",
        "reconcile_method",
    ]
    summary_detailed = _summarize(raw_rows, detailed_group)
    summary_detailed_path = out_dir / "reconcile_matrix_summary_detailed.csv"
    _write_csv(summary_detailed_path, summary_detailed)

    compact_group = [
        "paper_scope",
        "experiment_stage",
        "model_family",
        "base_model_family",
        "dataset",
        "num_timesteps_out",
        "base_model",
        "lagtcn_graph_source_version",
        "reconcile_method",
    ]
    summary_compact = _summarize(raw_rows, compact_group)
    summary_compact_path = out_dir / "reconcile_matrix_summary_compact.csv"
    _write_csv(summary_compact_path, summary_compact)

    level_detailed_group = detailed_group + ["level", "horizon_label"]
    level_summary_detailed = _summarize(
        level_raw_rows,
        level_detailed_group,
        metric_columns=LEVEL_METRIC_COLUMNS,
    )
    level_summary_detailed_path = out_dir / "reconcile_matrix_level_summary_detailed.csv"
    _write_csv(level_summary_detailed_path, level_summary_detailed)

    level_compact_group = compact_group + ["level", "horizon_label"]
    level_summary_compact = _summarize(
        level_raw_rows,
        level_compact_group,
        metric_columns=LEVEL_METRIC_COLUMNS,
    )
    level_summary_compact_path = out_dir / "reconcile_matrix_level_summary_compact.csv"
    _write_csv(level_summary_compact_path, level_summary_compact)

    if skipped:
        skipped_path = out_dir / "reconcile_matrix_skipped.json"
        with skipped_path.open("w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)
        print(f"[WARN] Skipped runs: {len(skipped)} (details: {skipped_path})")

    print(f"[INFO] Processed base runs: {run_count}")
    print(f"[INFO] Reconciliation rows: {len(raw_rows)}")
    print(f"[INFO] Level reconciliation rows: {len(level_raw_rows)}")
    print(f"[INFO] Raw CSV: {raw_path}")
    print(f"[INFO] Level raw CSV: {level_raw_path}")
    print(f"[INFO] Detailed summary CSV: {summary_detailed_path}")
    print(f"[INFO] Compact summary CSV: {summary_compact_path}")
    print(f"[INFO] Level detailed summary CSV: {level_summary_detailed_path}")
    print(f"[INFO] Level compact summary CSV: {level_summary_compact_path}")


if __name__ == "__main__":
    main()
