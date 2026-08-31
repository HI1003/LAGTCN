"""Final Applied Energy seasonal MASE implementation (training-period scale).

For the hourly day-ahead task, node i uses the 24-hour seasonal-naive MAE over
the frozen training segment in original load units,

    d_i = mean_{tau=25..T_tr} |y_{i,tau} - y_{i,tau-24}|,

shared by all horizons and forecasting models. The numerator is the model MAE
over the evaluated origins/horizons. sMASE-24_i = numerator_i / d_i; overall
and level-wise scores macro-average the node-wise ratios. Nodes with d_i below
``min_scale`` are excluded and counted. The seasonal period is determined by
the hourly load cycle, not by the 168-hour input-window length.

This module is the single implementation used by both the Phase 1
post-processing scripts and the Phase 2 evaluation pipeline. It is pure numpy
and testable on synthetic fixtures; it never reads run artifacts itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MASE_SEASONAL_PERIOD = 24
MASE_LABEL = "sMASE-24"
MASE_VERSION = "ae_smase24_trainscale_v1"
DEFAULT_MIN_SCALE = 1e-8


def as_sample_node_horizon(values: np.ndarray) -> np.ndarray:
    """Coerce predictions/targets to float64 [num_samples, num_nodes, num_horizons]."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, got shape {arr.shape}")
    return arr


def compute_naive_scale(
    train_series: np.ndarray,
    *,
    seasonal_period: int = MASE_SEASONAL_PERIOD,
) -> np.ndarray:
    """Per-node seasonal-naive MAE over the training segment.

    Args:
        train_series: [T_tr, num_nodes] original-scale observations covering
            exactly the frozen training segment.
        seasonal_period: Positive lag in data time steps. The formal hourly
            Applied Energy protocol fixes this to 24.
    Returns:
        [num_nodes] float64 scales (may contain exact zeros for degenerate nodes).
    """
    series = np.asarray(train_series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(f"train_series must be [T, N], got shape {series.shape}")
    if isinstance(seasonal_period, bool) or not isinstance(
        seasonal_period, (int, np.integer)
    ):
        raise ValueError("seasonal_period must be a positive integer.")
    seasonal_period = int(seasonal_period)
    if seasonal_period < 1:
        raise ValueError("seasonal_period must be a positive integer.")
    if series.shape[0] <= seasonal_period:
        raise ValueError(
            "train_series needs more timesteps than seasonal_period "
            f"({series.shape[0]} <= {seasonal_period})."
        )
    if not np.all(np.isfinite(series)):
        raise ValueError("train_series contains non-finite values.")
    return np.mean(
        np.abs(series[seasonal_period:] - series[:-seasonal_period]), axis=0
    )


def naive_scale_metadata(
    scale: np.ndarray,
    *,
    min_scale: float = DEFAULT_MIN_SCALE,
    train_length: int | None = None,
    node_names: list[str] | None = None,
    seasonal_period: int = MASE_SEASONAL_PERIOD,
) -> dict:
    """Frozen denominator metadata to persist alongside recomputed metrics."""
    scale = np.asarray(scale, dtype=np.float64)
    degenerate = np.flatnonzero(scale < min_scale)
    valid = scale[scale >= min_scale]
    meta = {
        "mase_version": MASE_VERSION,
        "mase_label": MASE_LABEL,
        "seasonal_period": int(seasonal_period),
        "scale_reference": "training_period_seasonal_naive_mae",
        "min_scale": float(min_scale),
        "num_nodes": int(scale.size),
        "num_degenerate_nodes": int(degenerate.size),
        "degenerate_node_indices": degenerate.tolist(),
        "scale_min": float(valid.min()) if valid.size else None,
        "scale_median": float(np.median(valid)) if valid.size else None,
        "scale_max": float(valid.max()) if valid.size else None,
        "scale_per_node": scale.tolist(),
    }
    if train_length is not None:
        meta["train_length"] = int(train_length)
    if node_names is not None:
        if len(node_names) != scale.size:
            raise ValueError("node_names length does not match scale length.")
        meta["degenerate_node_names"] = [str(node_names[i]) for i in degenerate]
    return meta


def compute_mase_per_node(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
    *,
    min_scale: float = DEFAULT_MIN_SCALE,
    cell_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Node-wise seasonal MASE. Excluded (degenerate-scale) nodes are NaN.

    Args:
        y_true, y_pred: [S, N, H] (or [S, N]) original-scale test blocks.
        scale: [N] frozen training-period naive MAE.
        cell_mask: optional bool [S, H]; True marks origin/horizon cells that
            enter the numerator (used by boundary-trim robustness checks).
    """
    yt = as_sample_node_horizon(y_true)
    yp = as_sample_node_horizon(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (yt.shape[1],):
        raise ValueError(f"scale shape {scale.shape} != ({yt.shape[1]},)")

    abs_err = np.abs(yt - yp)  # [S, N, H]
    if cell_mask is None:
        numerator = abs_err.mean(axis=(0, 2))
    else:
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != (yt.shape[0], yt.shape[2]):
            raise ValueError(f"cell_mask shape {mask.shape} != ({yt.shape[0]}, {yt.shape[2]})")
        kept = mask.sum()
        if kept == 0:
            return np.full(yt.shape[1], np.nan)
        numerator = (abs_err * mask[:, None, :]).sum(axis=(0, 2)) / float(kept)

    mase = np.full(scale.shape, np.nan)
    included = scale >= min_scale
    mase[included] = numerator[included] / scale[included]
    return mase


def macro_average_mase(mase_per_node: np.ndarray, indices: list[int]) -> dict:
    """Macro-average node-wise seasonal MASE over a level, reporting exclusions."""
    values = np.asarray(mase_per_node, dtype=np.float64)[list(indices)]
    finite = values[np.isfinite(values)]
    return {
        "mase": float(finite.mean()) if finite.size else float("nan"),
        "n_nodes": len(indices),
        "n_excluded": int(len(indices) - finite.size),
    }


def build_level_groups(
    num_nodes: int,
    *,
    bottom_start_idx: int,
    num_bottom_nodes: int,
    middle_levels: list[list[int]] | None = None,
    num_top_nodes: int = 1,
) -> list[tuple[str, list[int]]]:
    """Level groups (All / top / middle{i} / bottom) with explicit middle levels.

    ``middle_levels`` comes from dataset metadata (hierarchy_info.json); pass
    None or [] for a 2-level hierarchy. Names match metrics.calculate_level_metrics.
    """
    bottom_start = int(bottom_start_idx)
    num_bottom = int(num_bottom_nodes)
    if bottom_start + num_bottom != num_nodes:
        raise ValueError(
            f"bottom_start_idx({bottom_start}) + num_bottom({num_bottom}) != num_nodes({num_nodes})"
        )
    groups: list[tuple[str, list[int]]] = [
        ("All", list(range(num_nodes))),
        ("top_level", list(range(num_top_nodes))),
    ]
    covered = int(num_top_nodes)
    for i, level in enumerate(middle_levels or [], start=1):
        indices = [int(x) for x in level]
        if any(idx < num_top_nodes or idx >= bottom_start for idx in indices):
            raise ValueError(f"middle level {i} indices {indices} outside middle range.")
        groups.append((f"middle{i}_level", indices))
        covered += len(indices)
    if covered != bottom_start:
        raise ValueError(
            f"top+middle levels cover {covered} nodes but bottom_start_idx={bottom_start}."
        )
    groups.append(("bottom_level", list(range(bottom_start, bottom_start + num_bottom))))
    return groups


def compute_mase_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scale: np.ndarray,
    level_groups: list[tuple[str, list[int]]],
    *,
    min_scale: float = DEFAULT_MIN_SCALE,
    cell_mask: np.ndarray | None = None,
) -> list[dict]:
    """Level x horizon sMASE-24 rows (horizon 'all' plus each step when H > 1)."""
    yt = as_sample_node_horizon(y_true)
    yp = as_sample_node_horizon(y_pred)
    num_horizons = yt.shape[2]

    scale = np.asarray(scale, dtype=np.float64)
    if scale.shape != (yt.shape[1],):
        raise ValueError(f"scale shape {scale.shape} != ({yt.shape[1]},)")
    if yp.shape != yt.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    abs_err = np.abs(yt - yp)
    if cell_mask is None:
        per_horizon_numerator = abs_err.mean(axis=0).T  # [H, N]
        all_numerator = abs_err.mean(axis=(0, 2))
    else:
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != (yt.shape[0], num_horizons):
            raise ValueError(
                f"cell_mask shape {mask.shape} != ({yt.shape[0]}, {num_horizons})"
            )
        kept_per_horizon = mask.sum(axis=0)
        error_sum = np.einsum(
            "snh,sh->hn", abs_err, mask, optimize=True
        )
        per_horizon_numerator = np.full_like(error_sum, np.nan, dtype=np.float64)
        nonempty = kept_per_horizon > 0
        per_horizon_numerator[nonempty] = (
            error_sum[nonempty] / kept_per_horizon[nonempty, None]
        )
        kept_total = int(kept_per_horizon.sum())
        all_numerator = (
            error_sum.sum(axis=0) / float(kept_total)
            if kept_total
            else np.full(yt.shape[1], np.nan)
        )

    included = scale >= min_scale
    numerator_rows: list[tuple[str, np.ndarray]] = [("all", all_numerator)]
    if num_horizons > 1:
        numerator_rows.extend(
            (f"h{h + 1}", per_horizon_numerator[h])
            for h in range(num_horizons)
        )

    rows: list[dict] = []
    for horizon_label, numerator in numerator_rows:
        per_node = np.full(scale.shape, np.nan)
        per_node[included] = numerator[included] / scale[included]
        for level_name, indices in level_groups:
            if not indices:
                continue
            summary = macro_average_mase(per_node, indices)
            rows.append({
                "Level": level_name,
                "Horizon": horizon_label,
                "MASE": summary["mase"],
                "n_nodes": summary["n_nodes"],
                "n_excluded": summary["n_excluded"],
            })
    return rows


def assert_unit_stride(time_index, *, expected_step: str = "1h") -> None:
    """Assert consecutive forecast origins advance by exactly one data time step.

    Equal spacing alone is not enough: a stride-2 window sequence is equally
    spaced but breaks the triangular boundary masks. Datetime-like indices must
    advance by exactly ``expected_step`` (default one hour, the sampling period
    of all Applied Energy datasets); integer indices must advance by exactly 1.
    """
    if len(time_index) < 2:
        return
    arr = np.asarray(time_index)
    if np.issubdtype(arr.dtype, np.integer):
        diffs = np.diff(arr.astype(np.int64))
        expected = 1
    else:
        try:
            idx = pd.to_datetime(time_index)
            diffs = np.diff(idx.asi8)
            expected = int(pd.Timedelta(expected_step).value)
        except (ValueError, TypeError):
            diffs = np.diff(arr.astype(np.int64))
            expected = 1
    if diffs.size and (diffs.min() != expected or diffs.max() != expected):
        raise AssertionError(
            f"Forecast origins are not stride-1 (expected step {expected}): "
            f"observed step range [{diffs.min()}, {diffs.max()}]."
        )
