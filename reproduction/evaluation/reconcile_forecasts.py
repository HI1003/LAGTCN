"""Applied Energy Phase 1 unified post-processing (provisional diagnostics).

Recomputes, from saved base-forecast artifacts and WITHOUT retraining:
  A2  hierarchy metadata taken from dataset metadata (middle_levels), never
      from the old run configs; every output records postprocess_version.
  B   final-definition sMASE-24 (training-period per-node 24-hour seasonal-
      naive scale, macro average per level) via lagtcn.core.scaled_error, with the frozen
      denominators persisted per dataset.
  C   nonnegative BU / TD-FP / MinT-NNLS reconciliation with full solver
      diagnostics via lagtcn.reconciliation.methods. Metric-only revisions may
      explicitly reuse an existing archive after strict structural, finite,
      nonnegative, and coherence validation; invalid/missing archives fall
      back to fresh reconciliation.
  D   non-finite isolation: runs whose predictions/true values contain
      NaN/Inf are quarantined and excluded from summaries.
  E   boundary-trim robustness check: no mask vs exact triangular mask vs
      whole-row trim at the validation->test boundary (h=1 masks are empty).
  F1  negative base-prediction rate per dataset x model x level.
  F2  clamp-trigger rates against LAGTCN's original-scale bounds
      [0, expm1(mu + 6*sigma)].

All outputs are tagged ``provisional_diagnostic`` and never overwrite run
artifacts. The script only reads run directories; it is purely additive and
does not import any in-flight training module. With ``--save-reconciled``, the
three primary reconciled forecasts are persisted as float64 compressed NPZ
archives under the post-processing output directory.

Usage:
    python -m reproduction.evaluation.reconcile_forecasts --runs-root <dir> [--data-root Data]
        [--output <dir>] [--mint-weights ols] [--save-reconciled]
        [--reuse-reconciled] [--limit N]
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import os
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from lagtcn.core import scaled_error as mase
from lagtcn.reconciliation import methods as reconcile_ae

POSTPROCESS_VERSION = "ae_phase1_provisional_diagnostic_v3_smase24"
RESULT_TAG = "provisional_diagnostic"
RECONCILED_ARCHIVE_VERSION = "ae_reconciled_npz_v1"

# Keep these fixed artifact names local. Importing output_naming.py would tie
# this read-only Phase 1 tool to an in-flight training module, contrary to the
# isolation requirement in CODE_FIX_PLAN.md.
BASE_PRED_FILENAME = "base_pred.csv"
PRED_FILENAME = "pred.csv"
TRUE_FILENAME = "true.csv"

PROVENANCE_KEYS = (
    "graph_mode",
    "gnn_type",
    "temporal_type",
    "st_mode",
    "stgnn_graph_source",
    "feature_set",
    "experiment_id",
    "experiment_stage",
    "output_namespace",
    "run_label",
    "timestamp",
    "num_timesteps_in",
)

CLAMP_INTRODUCTIONS = {
    "LAGTCN": {
        "introduced_at": "2026-06-23T16:32:17",
        "commit": "4ebc6f54da13cffd7feb0e63ad2bbc085c071813",
    },
}


# Base-forecast discovery. For neural-reconciliation runs main.py saves the
# reconciled output as pred.csv / predictions_*.csv and the pre-reconciliation
# forecast as base_pred.csv / base_predictions_*.csv, so the base_* name must
# win; rec_pred.csv / reconciled_predictions_*.csv are never read as base.
PREDICTION_CANDIDATES = (BASE_PRED_FILENAME, PRED_FILENAME)
LEGACY_PREDICTION_PATTERNS = ("base_predictions_*.csv", "predictions_*.csv")
TRUE_CANDIDATES = (TRUE_FILENAME,)
LEGACY_TRUE_PATTERNS = ("true_values_*.csv",)
CONFIG_CANDIDATES = ("config.json", "model_info.json")
LEGACY_CONFIG_PATTERNS = ("config_*.json", "model_info_*.json")

BOUNDARY_VARIANTS = ("none", "triangular", "row_trim")

CLAMP_LOWER_ATOL = 1e-5     # |y_hat| below this counts as a lower-bound hit
CLAMP_UPPER_RTOL = 1e-6     # y_hat >= upper*(1-rtol) counts as an upper hit


# --------------------------------------------------------------------------- #
# Dataset metadata (A2: single source of truth for hierarchy + sMASE-24 scale)
# --------------------------------------------------------------------------- #

@dataclass
class DatasetMeta:
    name: str
    node_order: list[str]
    bottom_start_idx: int
    num_bottom_nodes: int
    middle_levels: list[list[int]]
    num_top_nodes: int
    sum_matrix: np.ndarray
    naive_scale: np.ndarray
    train_length: int
    clamp_upper_bound: float | None
    scale_metadata: dict = field(repr=False, default_factory=dict)

    def level_groups(self) -> list[tuple[str, list[int]]]:
        return mase.build_level_groups(
            len(self.node_order),
            bottom_start_idx=self.bottom_start_idx,
            num_bottom_nodes=self.num_bottom_nodes,
            middle_levels=self.middle_levels,
            num_top_nodes=self.num_top_nodes,
        )


def load_dataset_meta(dataset_dir: Path) -> DatasetMeta:
    info = json.loads((dataset_dir / "hierarchy_info.json").read_text())
    if "middle_levels" not in info:
        raise RuntimeError(
            f"{dataset_dir}: hierarchy_info.json lacks 'middle_levels'; "
            "rebuild hierarchy_info.json with the dataset-preparation notebook."
        )
    node_order = [str(x) for x in info["node_order"]]
    sum_matrix = (
        pd.read_csv(dataset_dir / "sum_matrix.csv", header=None)
        .dropna(axis=1, how="all")
        .to_numpy(dtype=np.float64)
    )

    norm = np.load(dataset_dir / "normalization_params.npy", allow_pickle=True).item()
    values = np.load(dataset_dir / "node_values.npy")  # [T, N, F], normalized log space
    target = values[:, :, 0].astype(np.float64)
    original = inverse_transform(target, norm)

    total_t = original.shape[0]
    train_length = int(norm.get("train_T") or int(total_t * float(norm.get("train_ratio", 0.8))))
    if not 2 <= train_length <= total_t:
        raise RuntimeError(f"{dataset_dir}: invalid frozen train length {train_length} (T={total_t}).")
    naive_scale = mase.compute_naive_scale(
        original[:train_length], seasonal_period=mase.MASE_SEASONAL_PERIOD
    )

    clamp_upper = None
    if norm.get("use_log") and str(norm.get("norm_method", norm.get("method", ""))).lower() == "zscore":
        clamp_upper = float(np.expm1(float(norm["mean"]) + 6.0 * float(norm["std"])))

    scale_metadata = mase.naive_scale_metadata(
        naive_scale,
        train_length=train_length,
        node_names=node_order,
        seasonal_period=mase.MASE_SEASONAL_PERIOD,
    )
    scale_metadata.update({
        "dataset": dataset_dir.name,
        "postprocess_version": POSTPROCESS_VERSION,
        "result_tag": RESULT_TAG,
        "total_length": int(total_t),
        "clamp_upper_bound_original_scale": clamp_upper,
    })

    return DatasetMeta(
        name=dataset_dir.name,
        node_order=node_order,
        bottom_start_idx=int(info["bottom_start_idx"]),
        num_bottom_nodes=int(info["num_bottom_nodes"]),
        middle_levels=[[int(i) for i in lvl] for lvl in info["middle_levels"]],
        num_top_nodes=len(info.get("top_nodes", [])) or 1,
        sum_matrix=sum_matrix,
        naive_scale=naive_scale,
        train_length=train_length,
        clamp_upper_bound=clamp_upper,
        scale_metadata=scale_metadata,
    )


def inverse_transform(normalized: np.ndarray, norm: dict) -> np.ndarray:
    """Invert the stored normalization (zscore or minmax, optional log1p)."""
    method = str(norm.get("norm_method", norm.get("method", "zscore"))).lower()
    if method == "zscore":
        data = normalized * float(norm["std"]) + float(norm["mean"])
    else:
        low = float(norm.get("min", norm.get("global_min")))
        high = float(norm.get("max", norm.get("global_max")))
        data = normalized * (high - low) + low
    if norm.get("use_log"):
        data = np.exp(data) - float(norm.get("log_offset", 1.0))
    return data


# --------------------------------------------------------------------------- #
# Run discovery and artifact loading
# --------------------------------------------------------------------------- #

def _find_artifact(run_dir: Path, names: tuple[str, ...], patterns: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists():
            return path
    for pattern in patterns:
        matches = sorted(p for p in run_dir.iterdir() if fnmatch.fnmatch(p.name, pattern))
        if matches:
            return matches[-1]
    return None


def discover_runs(runs_root: Path) -> list[Path]:
    run_dirs = []
    for candidate in sorted(runs_root.rglob("*")):
        if not candidate.is_dir():
            continue
        pred = _find_artifact(candidate, PREDICTION_CANDIDATES, LEGACY_PREDICTION_PATTERNS)
        true = _find_artifact(candidate, TRUE_CANDIDATES, LEGACY_TRUE_PATTERNS)
        if pred is not None and true is not None:
            run_dirs.append(candidate)
    return run_dirs


def load_run_config(run_dir: Path) -> dict:
    path = _find_artifact(run_dir, CONFIG_CANDIDATES, LEGACY_CONFIG_PATTERNS)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return {"_config_parse_error": str(path)}
    # model_info.json nests the run config under a "config" key.
    if isinstance(data.get("config"), dict):
        merged = dict(data["config"])
        for key in ("model_name", "timestamp"):
            if key in data:
                merged.setdefault(key, data[key])
        return merged
    return data


def parse_prediction_csv(path: Path) -> tuple[np.ndarray, list[str], pd.Index]:
    """Parse a wide prediction CSV into ([S, N, H], node_names, time_index)."""
    df = pd.read_csv(path, index_col=0)
    columns = [str(c) for c in df.columns]
    if any("_t+" in c for c in columns):
        nodes: list[str] = []
        horizons: set[int] = set()
        for col in columns:
            node, _, suffix = col.rpartition("_t+")
            if not node or not suffix.isdigit():
                raise ValueError(f"{path}: unparseable column '{col}'.")
            horizons.add(int(suffix))
            if int(suffix) == 1:
                nodes.append(node)
        num_h = max(horizons)
        if horizons != set(range(1, num_h + 1)):
            raise ValueError(f"{path}: horizon suffixes {sorted(horizons)} are not 1..H.")
        expected = [f"{node}_t+{h}" for h in range(1, num_h + 1) for node in nodes]
        if columns != expected:
            raise ValueError(f"{path}: column order is not horizon-major node blocks.")
        arr = df.to_numpy(dtype=np.float64).reshape(len(df), num_h, len(nodes)).transpose(0, 2, 1)
        return arr, nodes, df.index
    return df.to_numpy(dtype=np.float64)[:, :, None], columns, df.index


def resolve_dataset_name(config: dict, run_dir: Path, known: set[str]) -> str | None:
    raw = config.get("raw_data_dir") or config.get("dataset")
    if raw:
        name = Path(str(raw)).name
        if name in known:
            return name
    for part in run_dir.parts[::-1]:
        if part in known:
            return part
    return None


# --------------------------------------------------------------------------- #
# Metric computation
# --------------------------------------------------------------------------- #

def boundary_cell_mask(num_samples: int, num_horizons: int, variant: str) -> np.ndarray | None:
    """True marks origin/horizon cells kept for metric computation.

    triangular: horizon step h (1-based) drops the first H-h origins whose
    targets overlap the previous split segment. row_trim drops the first H-1
    origins entirely. For H == 1 both masks keep everything.
    """
    if variant == "none":
        return None
    mask = np.ones((num_samples, num_horizons), dtype=bool)
    if variant == "triangular":
        for h in range(1, num_horizons + 1):
            drop = min(num_horizons - h, num_samples)
            mask[:drop, h - 1] = False
        return mask
    if variant == "row_trim":
        mask[: min(num_horizons - 1, num_samples), :] = False
        return mask
    raise ValueError(f"Unknown boundary variant '{variant}'.")


def pooled_metric_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    level_groups: list[tuple[str, list[int]]],
    cell_mask: np.ndarray | None,
) -> dict[tuple[str, str], dict]:
    """Pooled WAPE(%) / MAE / RMSE per level x horizon under an optional mask."""
    num_h = y_true.shape[2]
    horizon_slices: list[tuple[str, slice]] = [("all", slice(None))]
    if num_h > 1:
        horizon_slices += [(f"h{h + 1}", slice(h, h + 1)) for h in range(num_h)]

    rows: dict[tuple[str, str], dict] = {}
    for horizon_label, h_slice in horizon_slices:
        yt, yp = y_true[:, :, h_slice], y_pred[:, :, h_slice]
        mask = None if cell_mask is None else cell_mask[:, h_slice]
        for level_name, indices in level_groups:
            if not indices:
                continue
            t, p = yt[:, indices, :], yp[:, indices, :]
            if mask is not None:
                keep = np.broadcast_to(mask[:, None, :], t.shape)
                t, p = t[keep], p[keep]
            if t.size == 0:
                rows[(level_name, horizon_label)] = {"MAE": np.nan, "RMSE": np.nan, "WAPE": np.nan}
                continue
            abs_err = np.abs(t - p)
            rows[(level_name, horizon_label)] = {
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt(np.mean((t - p) ** 2))),
                "WAPE": float(100.0 * abs_err.sum() / np.maximum(np.abs(t).sum(), 1e-12)),
            }
    return rows


def metric_records(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: DatasetMeta,
    *,
    method: str,
    variant: str,
) -> list[dict]:
    level_groups = meta.level_groups()
    cell_mask = boundary_cell_mask(y_true.shape[0], y_true.shape[2], variant)
    pooled = pooled_metric_rows(y_true, y_pred, level_groups, cell_mask)
    mase_rows = mase.compute_mase_report(
        y_true, y_pred, meta.naive_scale, level_groups, cell_mask=cell_mask
    )
    records = []
    for row in mase_rows:
        key = (row["Level"], row["Horizon"])
        records.append({
            "method": method,
            "boundary_variant": variant,
            "level": row["Level"],
            "horizon_label": row["Horizon"],
            "mase": row["MASE"],
            "mase_n_excluded": row["n_excluded"],
            "mase_version": mase.MASE_VERSION,
            "mase_label": mase.MASE_LABEL,
            "mase_seasonal_period": mase.MASE_SEASONAL_PERIOD,
            **{k.lower(): v for k, v in pooled[key].items()},
        })
    return records


# --------------------------------------------------------------------------- #
# Per-run processing
# --------------------------------------------------------------------------- #

def assert_h1_boundary_metrics_equal(records: list[dict], run_id: str) -> None:
    """Assert semantic equality of H=1 variants within floating precision."""
    base = [record for record in records if record["method"] == "base"]
    by_variant: dict[tuple[str, str], dict[str, float]] = {}
    for record in base:
        cell = (record["level"], record["horizon_label"])
        by_variant.setdefault(cell, {})[record["boundary_variant"]] = record["wape"]
    for cell, variants in by_variant.items():
        reference = variants["none"]
        if any(
            not np.isclose(value, reference, rtol=1e-12, atol=1e-12)
            for value in variants.values()
        ):
            raise RuntimeError(
                f"{run_id}: H=1 boundary variants disagree at {cell}: {variants}"
            )


def clamp_provenance(model_name: str, run_timestamp) -> dict:
    """Infer clamp availability from model family and recorded run timestamp.

    Historical artifacts do not store a git commit. The timestamp comparison
    is therefore explicitly labeled as an inference, not exact code
    provenance. It still prevents pre-introduction LAGTCN runs from being
    incorrectly described as clamped.
    """
    upper = str(model_name).upper().replace("-", "_").replace(" ", "_")
    marker = next((name for name in CLAMP_INTRODUCTIONS if name in upper), None)
    if marker is None:
        return {
            "clamp_model_family": False,
            "clamp_expected": False,
            "clamp_provenance": "model_family_not_clamped",
            "clamp_introduced_at": None,
            "clamp_introduced_commit": None,
        }

    intro = CLAMP_INTRODUCTIONS[marker]
    digits = "".join(ch for ch in str(run_timestamp or "") if ch.isdigit())
    parsed = None
    if len(digits) >= 14:
        try:
            parsed = datetime.datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
        except ValueError:
            parsed = None
    introduced_at = datetime.datetime.fromisoformat(intro["introduced_at"])
    if parsed is None:
        expected = None
        source = "unknown_missing_run_commit_and_parseable_timestamp"
    else:
        expected = parsed >= introduced_at
        source = "inferred_from_run_timestamp_vs_repository_commit_time"
    return {
        "clamp_model_family": True,
        "clamp_expected": expected,
        "clamp_provenance": source,
        "clamp_introduced_at": intro["introduced_at"],
        "clamp_introduced_commit": intro["commit"],
    }


def reconciled_archive_path(output_dir: Path, run_id: str) -> Path:
    return (
        output_dir / "reconciled_predictions" / Path(run_id)
        / "reconciled_predictions.npz"
    )


def load_validated_reconciled_archive(
    output_dir: Path,
    *,
    run_id: str,
    expected_shape: tuple[int, ...],
    expected_time_index: np.ndarray,
    expected_node_order: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int,
) -> dict | None:
    """Load an unchanged reconciliation archive after strict identity checks.

    The archive is reusable across metric-only revisions because reconciliation
    depends on the base forecasts and hierarchy, not on the MASE denominator.
    Any missing or incompatible archive returns ``None`` or raises, allowing
    the caller to fall back to a fresh reconciliation.
    """
    archive_path = reconciled_archive_path(output_dir, run_id)
    if not archive_path.is_file():
        return None
    required = {
        "bu", "td_fp", "mint_ols", "time_index", "node_order",
        "archive_version", "postprocess_version", "reconcile_version",
        "source_run_id", "source_prediction_file",
    }
    with np.load(archive_path, allow_pickle=False) as saved:
        missing = required - set(saved.files)
        if missing:
            raise RuntimeError(f"archive missing fields {sorted(missing)}")
        if str(saved["archive_version"].item()) != RECONCILED_ARCHIVE_VERSION:
            raise RuntimeError("archive version mismatch")
        if str(saved["reconcile_version"].item()) != reconcile_ae.RECONCILE_AE_VERSION:
            raise RuntimeError("reconciliation version mismatch")
        if str(saved["source_run_id"].item()) != run_id:
            raise RuntimeError("source_run_id mismatch")
        if not np.array_equal(
            np.asarray(saved["time_index"], dtype=str),
            np.asarray(expected_time_index, dtype=str),
        ):
            raise RuntimeError("time_index mismatch")
        if not np.array_equal(
            np.asarray(saved["node_order"], dtype=str),
            np.asarray(expected_node_order, dtype=str),
        ):
            raise RuntimeError("node_order mismatch")
        forecasts = {
            method: np.asarray(saved[method], dtype=np.float64).copy()
            for method in ("bu", "td_fp", "mint_ols")
        }
        archive_postprocess_version = str(saved["postprocess_version"].item())
        source_prediction_file = str(saved["source_prediction_file"].item())

    validation_stats = {}
    for method, forecast in forecasts.items():
        if forecast.shape != tuple(expected_shape):
            raise RuntimeError(
                f"{method} shape mismatch {forecast.shape} != {expected_shape}"
            )
        if not np.all(np.isfinite(forecast)):
            raise RuntimeError(f"{method} contains non-finite values")
        stats = reconcile_ae.coherence_stats(
            forecast, sum_matrix, bottom_start_idx=bottom_start_idx
        )
        if stats["min_value"] < -1e-10:
            raise RuntimeError(
                f"{method} violates nonnegativity: {stats['min_value']}"
            )
        if stats["coherence_residual_max_abs"] > 1e-8:
            raise RuntimeError(
                f"{method} violates coherence: "
                f"{stats['coherence_residual_max_abs']}"
            )
        validation_stats[method] = stats

    return {
        "forecasts": forecasts,
        "validation_stats": validation_stats,
        "archive_file": str(archive_path.relative_to(output_dir)),
        "archive_postprocess_version": archive_postprocess_version,
        "source_prediction_file": source_prediction_file,
    }


def process_run(run_dir: Path, runs_root: Path, meta_cache: dict, data_root: Path,
                mint_weights: list[str], nnls_workers: int = 1,
                reconciled_cache_root: Path | None = None,
                cached_diagnostics: dict | None = None) -> dict:
    run_id = str(run_dir.relative_to(runs_root))
    config = load_run_config(run_dir)
    known = {p.name for p in data_root.iterdir() if p.is_dir()}
    dataset_name = resolve_dataset_name(config, run_dir, known)
    if dataset_name is None:
        return {"run_id": run_id, "status": "skipped", "reason": "dataset_unresolved"}

    if dataset_name not in meta_cache:
        meta_cache[dataset_name] = load_dataset_meta(data_root / dataset_name)
    meta: DatasetMeta = meta_cache[dataset_name]

    pred_path = _find_artifact(run_dir, PREDICTION_CANDIDATES, LEGACY_PREDICTION_PATTERNS)
    true_path = _find_artifact(run_dir, TRUE_CANDIDATES, LEGACY_TRUE_PATTERNS)
    y_pred, pred_nodes, time_index = parse_prediction_csv(pred_path)
    y_true, true_nodes, true_time_index = parse_prediction_csv(true_path)

    result: dict = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model_name": config.get("model_name", "unknown"),
        "seed": config.get("seed"),
        "num_timesteps_out": int(y_pred.shape[2]),
        "postprocess_version": POSTPROCESS_VERSION,
        "result_tag": RESULT_TAG,
        "hierarchy_source": "dataset_metadata",
        "prediction_file": pred_path.name,
    }
    result.update({key: config.get(key) for key in PROVENANCE_KEYS})

    if y_pred.shape != y_true.shape:
        return {**result, "status": "skipped", "reason": f"shape_mismatch {y_pred.shape} vs {y_true.shape}"}
    if pred_nodes != true_nodes:
        return {**result, "status": "skipped", "reason": "node_order_mismatch_between_pred_and_true"}
    if not time_index.equals(true_time_index):
        return {**result, "status": "quarantined",
                "reason": "time_index_mismatch_between_pred_and_true"}
    if pred_nodes != meta.node_order:
        if set(pred_nodes) != set(meta.node_order):
            return {**result, "status": "skipped", "reason": "node_set_mismatch_with_dataset_metadata"}
        order = [pred_nodes.index(name) for name in meta.node_order]
        y_pred, y_true = y_pred[:, order, :], y_true[:, order, :]

    # D: non-finite isolation (before any metric touches the arrays).
    bad_pred = int((~np.isfinite(y_pred)).sum())
    bad_true = int((~np.isfinite(y_true)).sum())
    if bad_pred or bad_true:
        return {
            **result,
            "status": "quarantined",
            "reason": "non_finite_values",
            "n_nonfinite_pred": bad_pred,
            "n_nonfinite_true": bad_true,
        }

    try:
        mase.assert_unit_stride(time_index)
        result["stride_check"] = "ok"
    except AssertionError as exc:
        return {**result, "status": "quarantined", "reason": f"stride_violation: {exc}"}

    # Metrics for base + reconciled forecasts under all boundary variants.
    # WLS/shrink estimate W from test-period errors, so their outputs are kept
    # in a separate diagnostic stream and never enter the main metric table.
    primary_weights = [w for w in mint_weights if w == "ols"]
    diagnostic_weights = [w for w in mint_weights if w != "ols"]

    methods: dict[str, np.ndarray] = {"base": y_pred}
    diagnostic_methods: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict] = {
        "base": reconcile_ae.coherence_stats(
            y_pred, meta.sum_matrix, bottom_start_idx=meta.bottom_start_idx),
        "true_values": reconcile_ae.coherence_stats(
            y_true, meta.sum_matrix, bottom_start_idx=meta.bottom_start_idx),
    }
    cache_info = None
    cache_error = None
    expected_primary_methods = ["bu", "td_fp"] + [
        f"mint_{w}" for w in primary_weights
    ]
    if reconciled_cache_root is not None and expected_primary_methods == [
        "bu", "td_fp", "mint_ols"
    ]:
        try:
            cache_info = load_validated_reconciled_archive(
                reconciled_cache_root,
                run_id=run_id,
                expected_shape=y_pred.shape,
                expected_time_index=np.asarray(time_index, dtype=str),
                expected_node_order=np.asarray(meta.node_order, dtype=str),
                sum_matrix=meta.sum_matrix,
                bottom_start_idx=meta.bottom_start_idx,
            )
        except Exception as exc:
            cache_error = f"{type(exc).__name__}: {exc}"

    if cache_info is not None:
        methods.update(cache_info["forecasts"])
        old_run_diagnostics = (cached_diagnostics or {}).get(run_id, {})
        for method in expected_primary_methods:
            diag = dict(old_run_diagnostics.get(method, {}))
            stats = cache_info["validation_stats"][method]
            diag.setdefault("method", method)
            diag.setdefault("solver", "validated_reconciled_archive")
            diag.setdefault("n_failures", 0)
            diag.update({
                "reused_reconciled_archive": True,
                "reused_archive_file": cache_info["archive_file"],
                "reused_archive_postprocess_version": cache_info[
                    "archive_postprocess_version"
                ],
                "cache_validation_min_prediction": stats["min_value"],
                "cache_validation_coherence_residual_max_abs": stats[
                    "coherence_residual_max_abs"
                ],
            })
            diagnostics[method] = diag
        result["reconciled_archive_reused"] = True
        result["reconciled_archive_file"] = cache_info["archive_file"]
        result["reconciled_archive_postprocess_version"] = cache_info[
            "archive_postprocess_version"
        ]
    else:
        result["reconciled_archive_reused"] = False
        if cache_error is not None:
            result["reconciled_archive_reuse_error"] = cache_error
        for method in expected_primary_methods:
            y_tilde, diag = reconcile_ae.apply_reconciliation_ae(
                method, y_pred, meta.sum_matrix,
                bottom_start_idx=meta.bottom_start_idx,
                nnls_workers=nnls_workers,
            )
            methods[method] = y_tilde
            diagnostics[method] = diag
    for weight in diagnostic_weights:
        y_tilde, diag = reconcile_ae.apply_reconciliation_ae(
            f"mint_{weight}", y_pred, meta.sum_matrix,
            bottom_start_idx=meta.bottom_start_idx,
            true_values=y_true,
            nnls_workers=nnls_workers,
        )
        diag["weight_estimation_source"] = "test_targets"
        diagnostic_methods[f"mint_{weight}"] = y_tilde
        diagnostics[f"mint_{weight}"] = diag

    records: list[dict] = []
    for method_name, forecast in methods.items():
        for variant in BOUNDARY_VARIANTS:
            records.extend(metric_records(y_true, forecast, meta, method=method_name, variant=variant))
    diagnostic_records: list[dict] = []
    for method_name, forecast in diagnostic_methods.items():
        for variant in BOUNDARY_VARIANTS:
            diagnostic_records.extend(
                metric_records(y_true, forecast, meta, method=method_name, variant=variant))
    for row in diagnostic_records:
        row["weight_estimation_source"] = "test_targets"

    # E sanity: with H == 1 every variant must equal the unmasked metrics.
    if y_pred.shape[2] == 1:
        assert_h1_boundary_metrics_equal(records, run_id)

    # F1: negative base-prediction rates per level.
    negative_rates = []
    for level_name, indices in meta.level_groups():
        block = y_pred[:, indices, :]
        negative_rates.append({
            "level": level_name,
            "negative_fraction": float((block < 0).mean()) if block.size else 0.0,
            "min_prediction": float(block.min()) if block.size else 0.0,
        })

    # F2: retroactive clamp-trigger rates (all models; flag the clamped ones).
    lower_hits = float((np.abs(y_pred) < CLAMP_LOWER_ATOL).mean())
    if meta.clamp_upper_bound is not None:
        upper_hits = float((y_pred >= meta.clamp_upper_bound * (1.0 - CLAMP_UPPER_RTOL)).mean())
    else:
        upper_hits = None
    clamp_report = {
        **clamp_provenance(result["model_name"], config.get("timestamp")),
        "lower_bound": 0.0,
        "lower_hit_fraction": lower_hits,
        "upper_bound": meta.clamp_upper_bound,
        "upper_hit_fraction": upper_hits,
        "lower_atol": CLAMP_LOWER_ATOL,
        "upper_rtol": CLAMP_UPPER_RTOL,
    }

    return {
        **result,
        "status": "ok",
        "n_test_origins": int(y_pred.shape[0]),
        "metrics": records,
        "diagnostic_metrics": diagnostic_records,
        "reconciliation_diagnostics": diagnostics,
        "negative_prediction_rates": negative_rates,
        "clamp_trigger_rates": clamp_report,
        "reconciled_forecasts": {
            name: values for name, values in methods.items() if name != "base"
        },
        "time_index": np.asarray(time_index, dtype=str),
        "node_order": np.asarray(meta.node_order, dtype=str),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def save_reconciled_archive(
    output_dir: Path,
    outcome: dict,
) -> dict:
    """Atomically persist one run's reconciled forecasts outside the run dir."""
    archive_path = reconciled_archive_path(output_dir, outcome["run_id"])
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **outcome["reconciled_forecasts"],
        "time_index": outcome["time_index"],
        "node_order": outcome["node_order"],
        "archive_version": np.asarray(RECONCILED_ARCHIVE_VERSION),
        "postprocess_version": np.asarray(POSTPROCESS_VERSION),
        "reconcile_version": np.asarray(reconcile_ae.RECONCILE_AE_VERSION),
        "source_run_id": np.asarray(outcome["run_id"]),
        "source_prediction_file": np.asarray(outcome["prediction_file"]),
    }
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=archive_path.parent,
        prefix=".reconciled_",
        suffix=".npz",
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        try:
            np.savez_compressed(handle, **payload)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    os.replace(tmp_path, archive_path)

    method_names = sorted(outcome["reconciled_forecasts"])
    first = outcome["reconciled_forecasts"][method_names[0]]
    return {
        "run_id": outcome["run_id"],
        "dataset": outcome["dataset"],
        "archive_file": str(archive_path.relative_to(output_dir)),
        "archive_version": RECONCILED_ARCHIVE_VERSION,
        "methods": ",".join(method_names),
        "shape": "x".join(str(x) for x in first.shape),
        "dtype": str(first.dtype),
        "size_bytes": archive_path.stat().st_size,
        "source_prediction_file": outcome["prediction_file"],
        "archive_reused": False,
        "archive_postprocess_version": POSTPROCESS_VERSION,
    }


def describe_reused_reconciled_archive(output_dir: Path, outcome: dict) -> dict:
    """Describe an already validated archive without recompressing it."""
    archive_path = reconciled_archive_path(output_dir, outcome["run_id"])
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    method_names = sorted(outcome["reconciled_forecasts"])
    first = outcome["reconciled_forecasts"][method_names[0]]
    return {
        "run_id": outcome["run_id"],
        "dataset": outcome["dataset"],
        "archive_file": str(archive_path.relative_to(output_dir)),
        "archive_version": RECONCILED_ARCHIVE_VERSION,
        "methods": ",".join(method_names),
        "shape": "x".join(str(x) for x in first.shape),
        "dtype": str(first.dtype),
        "size_bytes": archive_path.stat().st_size,
        "source_prediction_file": outcome["prediction_file"],
        "archive_reused": True,
        "archive_postprocess_version": outcome[
            "reconciled_archive_postprocess_version"
        ],
    }

def refresh_clamp_provenance_table(runs_root: Path, output_dir: Path) -> dict:
    """Refresh only clamp provenance columns without recomputing forecasts."""
    table_path = output_dir / "clamp_trigger_rates.csv"
    if not table_path.is_file():
        raise FileNotFoundError(f"Missing clamp report: {table_path}")
    table = pd.read_csv(table_path)
    refreshed_at = datetime.datetime.now().isoformat(timespec="seconds")
    status_rows = []
    for _, row in table.iterrows():
        run_dir = runs_root / str(row["run_id"])
        config = load_run_config(run_dir)
        status_rows.append(
            clamp_provenance(row["model_name"], config.get("timestamp"))
        )
    status_table = pd.DataFrame(status_rows)
    for column in status_table.columns:
        table[column] = status_table[column]
    table["clamp_provenance_refreshed_at"] = refreshed_at

    tmp_path = table_path.with_name(f".{table_path.name}.tmp")
    try:
        table.to_csv(tmp_path, index=False)
        os.replace(tmp_path, table_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    expected = table["clamp_expected"]
    return {
        "table": str(table_path),
        "n_rows": int(len(table)),
        "n_model_family": int(table["clamp_model_family"].fillna(False).sum()),
        "n_expected_true": int((expected == True).sum()),
        "n_expected_false": int((expected == False).sum()),
        "n_expected_unknown": int(expected.isna().sum()),
        "refreshed_at": refreshed_at,
    }




def run_postprocess(runs_root: Path, data_root: Path, output_dir: Path,
                    mint_weights: list[str], limit: int | None = None,
                    save_reconciled: bool = False,
                    reuse_reconciled: bool = False,
                    nnls_workers: int = 1,
                    shard_count: int = 1,
                    shard_index: int = 0) -> dict:
    if shard_count < 1:
        raise ValueError(f"shard_count must be >= 1, got {shard_count}.")
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f"shard_index must be in [0, {shard_count}), got {shard_index}."
        )
    runs = discover_runs(runs_root)
    runs = runs[shard_index::shard_count]
    if limit:
        runs = runs[:limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_diagnostics: dict = {}
    if reuse_reconciled:
        diagnostics_path = output_dir / "reconciliation_diagnostics.json"
        if diagnostics_path.is_file():
            existing_diagnostics = json.loads(diagnostics_path.read_text())
    meta_cache: dict[str, DatasetMeta] = {}
    summaries, quarantine, errors = [], [], []
    metric_rows, diagnostic_metric_rows, negative_rows, clamp_rows = [], [], [], []
    diagnostics_all: dict[str, dict] = {}
    archive_rows: list[dict] = []

    for run_dir in runs:
        try:
            outcome = process_run(
                run_dir, runs_root, meta_cache, data_root, mint_weights, nnls_workers,
                reconciled_cache_root=output_dir if reuse_reconciled else None,
                cached_diagnostics=existing_diagnostics,
            )
        except Exception:
            errors.append({"run_id": str(run_dir.relative_to(runs_root)),
                           "traceback": traceback.format_exc()})
            continue

        status = outcome.get("status")
        if status == "quarantined":
            quarantine.append(outcome)
            continue
        if status != "ok":
            summaries.append(outcome)
            continue

        base_info_keys = (
            "run_id", "dataset", "model_name", "seed", "num_timesteps_out",
            *PROVENANCE_KEYS,
        )
        base_info = {key: outcome.get(key) for key in base_info_keys}
        base_info.update({
            "prediction_file": outcome["prediction_file"],
            "hierarchy_source": outcome["hierarchy_source"],
            "reconciled_archive_reused": outcome.get(
                "reconciled_archive_reused", False
            ),
            "reconciled_archive_reuse_error": outcome.get(
                "reconciled_archive_reuse_error"
            ),
            "reconciled_archive_postprocess_version": outcome.get(
                "reconciled_archive_postprocess_version"
            ),
        })
        if save_reconciled:
            try:
                if outcome.get("reconciled_archive_reused"):
                    archive_info = describe_reused_reconciled_archive(
                        output_dir, outcome
                    )
                else:
                    archive_info = save_reconciled_archive(output_dir, outcome)
            except Exception:
                errors.append({
                    "run_id": outcome["run_id"],
                    "stage": "save_reconciled_archive",
                    "traceback": traceback.format_exc(),
                })
                continue
            archive_rows.append({**base_info, **archive_info})
        summaries.append({
            **base_info,
            "status": "ok",
            "n_test_origins": outcome["n_test_origins"],
        })
        for row in outcome["metrics"]:
            metric_rows.append({**base_info, **row})
        for row in outcome["diagnostic_metrics"]:
            diagnostic_metric_rows.append({**base_info, **row})
        for row in outcome["negative_prediction_rates"]:
            negative_rows.append({**base_info, **row})
        clamp_rows.append({**base_info, **outcome["clamp_trigger_rates"]})
        diagnostics_all[outcome["run_id"]] = outcome["reconciliation_diagnostics"]

    def _tagged(rows: list[dict]) -> pd.DataFrame:
        return pd.DataFrame(rows).assign(
            postprocess_version=POSTPROCESS_VERSION, result_tag=RESULT_TAG)

    _tagged(metric_rows).to_csv(output_dir / "phase1_metrics_long.csv", index=False)
    _tagged(negative_rows).to_csv(output_dir / "negative_prediction_rates.csv", index=False)
    _tagged(clamp_rows).to_csv(output_dir / "clamp_trigger_rates.csv", index=False)
    if diagnostic_metric_rows:
        _tagged(diagnostic_metric_rows).to_csv(
            output_dir / "phase1_metrics_diagnostic_weights.csv", index=False)
    if archive_rows:
        _tagged(archive_rows).to_csv(
            output_dir / "reconciled_predictions_manifest.csv", index=False)
    (output_dir / "reconciliation_diagnostics.json").write_text(
        json.dumps(diagnostics_all, indent=2, allow_nan=False))
    (output_dir / "quarantine.json").write_text(
        json.dumps(quarantine, indent=2, allow_nan=False))
    for name, meta in meta_cache.items():
        (output_dir / f"mase_scale_{name}.json").write_text(
            json.dumps(meta.scale_metadata, indent=2, allow_nan=False))

    manifest = {
        "postprocess_version": POSTPROCESS_VERSION,
        "result_tag": RESULT_TAG,
        "mase_version": mase.MASE_VERSION,
        "mase_label": mase.MASE_LABEL,
        "mase_seasonal_period": mase.MASE_SEASONAL_PERIOD,
        "reconcile_version": reconcile_ae.RECONCILE_AE_VERSION,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "runs_root": str(runs_root),
        "mint_weights": mint_weights,
        "nnls_workers": int(nnls_workers),
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "save_reconciled": bool(save_reconciled),
        "reuse_reconciled_requested": bool(reuse_reconciled),
        "reconciled_archive_version": RECONCILED_ARCHIVE_VERSION if save_reconciled else None,
        "n_reconciled_archives": len(archive_rows),
        "n_reconciled_archives_reused": sum(
            bool(row.get("archive_reused")) for row in archive_rows
        ),
        "n_reconciled_archives_computed": sum(
            not bool(row.get("archive_reused")) for row in archive_rows
        ),
        "n_runs_discovered": len(runs),
        "n_runs_ok": sum(1 for s in summaries if s.get("status") == "ok"),
        "n_runs_skipped": sum(1 for s in summaries if s.get("status") == "skipped"),
        "n_runs_quarantined": len(quarantine),
        "n_runs_errored": len(errors),
        "errors": errors,
        "run_summaries": summaries,
    }
    (output_dir / "postprocess_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Directory tree containing run outputs.")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "Data"),
                        help="Directory containing the dataset folders.")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <runs-root>/_postprocess_ae_phase1, "
                             "or ..._smoke when --limit is set).")
    parser.add_argument("--refresh-clamp-provenance-only", action="store_true",
                        help="Only refresh clamp provenance columns in an existing "
                             "clamp_trigger_rates.csv; do not recompute forecasts.")
    parser.add_argument("--mint-weights", default="ols",
                        help="Comma-separated MinT weight modes (ols, wls, shrink).")
    parser.add_argument("--allow-test-weight-diagnostic", action="store_true",
                        help="Required for wls/shrink: they estimate W from test-period "
                             "errors and are written to a separate diagnostic file only.")
    parser.add_argument("--save-reconciled", action="store_true",
                        help="Persist BU/TD-FP/MinT-OLS forecasts as compressed float64 NPZ "
                             "archives under the post-processing output directory.")
    parser.add_argument("--reuse-reconciled", action="store_true",
                        help="Reuse an existing reconciliation NPZ only after validating "
                             "its identity, shape, time/node axes, finite values, "
                             "nonnegativity, and coherence; otherwise recompute it.")
    parser.add_argument("--nnls-workers", type=int, default=8,
                        help="Fork workers for large MinT-NNLS batches (default: 8; "
                             "small batches remain sequential).")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Deterministically split the sorted run list into this many "
                             "disjoint shards (default: 1).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based shard to process; requires 0 <= index < count "
                             "(default: 0).")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N runs.")
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).resolve()
    if not runs_root.is_dir():
        parser.error(f"--runs-root {runs_root} is not a directory.")
    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1.")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count.")
    if args.output:
        output_dir = Path(args.output)
    else:
        suffix = "_postprocess_ae_phase1_smoke" if args.limit else "_postprocess_ae_phase1"
        if args.shard_count > 1:
            suffix += f"_shard{args.shard_index:02d}-of-{args.shard_count:02d}"
        output_dir = runs_root / suffix
    if args.refresh_clamp_provenance_only:
        summary = refresh_clamp_provenance_table(runs_root, output_dir)
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    mint_weights = [w.strip().lower() for w in args.mint_weights.split(",") if w.strip()]
    if any(w != "ols" for w in mint_weights) and not args.allow_test_weight_diagnostic:
        parser.error(
            "wls/shrink MinT weights are estimated from test-period errors; "
            "pass --allow-test-weight-diagnostic to run them as diagnostics."
        )

    manifest = run_postprocess(
        runs_root, Path(args.data_root).resolve(), output_dir, mint_weights,
        limit=args.limit, save_reconciled=args.save_reconciled,
        reuse_reconciled=args.reuse_reconciled,
        nnls_workers=args.nnls_workers,
        shard_count=args.shard_count,
        shard_index=args.shard_index)
    print(json.dumps({k: manifest[k] for k in
                      ["postprocess_version", "n_runs_discovered", "n_runs_ok",
                       "n_runs_skipped", "n_runs_quarantined", "n_runs_errored",
                       "n_reconciled_archives_reused",
                       "n_reconciled_archives_computed"]}, indent=2))
    print(f"Outputs written to {output_dir}")
    return 1 if manifest["n_runs_errored"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
