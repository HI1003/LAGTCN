"""Forecast-accuracy and hierarchy-coherence metrics."""

from __future__ import annotations

import numpy as np


def forecast_metrics(
    true_values: np.ndarray,
    predictions: np.ndarray,
    epsilon: float = 1e-8,
) -> dict[str, float]:
    """Return aggregate metrics for arrays shaped ``[samples,nodes,horizon]``."""
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if true_values.shape != predictions.shape or true_values.ndim != 3:
        raise ValueError(
            "true_values and predictions must share shape [samples,nodes,horizon]"
        )
    if not np.isfinite(true_values).all() or not np.isfinite(predictions).all():
        raise ValueError("metric inputs contain NaN or Inf")
    errors = predictions - true_values
    absolute_errors = np.abs(errors)
    return {
        "mae": float(absolute_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "wape": float(absolute_errors.sum() / max(np.abs(true_values).sum(), epsilon)),
        "smape": float(
            np.mean(
                2.0
                * absolute_errors
                / np.maximum(np.abs(true_values) + np.abs(predictions), epsilon)
            )
        ),
    }

def coherence_metrics(
    predictions: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int | None = None,
) -> dict[str, float]:
    """Measure deviation from ``y = S b`` without modifying predictions."""
    predictions = np.asarray(predictions, dtype=np.float64)
    sum_matrix = np.asarray(sum_matrix, dtype=np.float64)
    if predictions.ndim != 3 or sum_matrix.ndim != 2:
        raise ValueError("predictions must be 3D and sum_matrix must be 2D")
    node_count = predictions.shape[1]
    bottom_count = sum_matrix.shape[1]
    if sum_matrix.shape[0] != node_count:
        raise ValueError("sum_matrix row count does not match predictions")
    bottom_start = (
        node_count - bottom_count
        if bottom_start_idx is None
        else int(bottom_start_idx)
    )
    bottom = predictions[:, bottom_start : bottom_start + bottom_count, :]
    coherent = np.einsum("nb,sbh->snh", sum_matrix, bottom)
    residual = predictions - coherent
    return {
        "coherence_mae": float(np.mean(np.abs(residual))),
        "coherence_rmse": float(np.sqrt(np.mean(residual**2))),
        "coherence_max_abs": float(np.max(np.abs(residual))),
    }
