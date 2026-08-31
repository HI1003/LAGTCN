"""Post-hoc hierarchical reconciliation utilities.

This module provides classic reconciliation methods that can be applied to
saved base forecasts without retraining:
  - Bottom-Up (BU)
  - Top-Down (TD)
  - MinT (OLS / WLS-var / Sample / Shrink)
"""
from __future__ import annotations

import numpy as np

try:
    from sklearn.covariance import LedoitWolf

    _HAS_LEDOIT_WOLF = True
except Exception:  # pragma: no cover - optional dependency fallback
    LedoitWolf = None
    _HAS_LEDOIT_WOLF = False


def _as_3d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected predictions with 2D/3D shape, got {arr.shape}")
    return arr


def _resolve_bottom_start(sum_matrix: np.ndarray, num_nodes: int, bottom_start_idx: int | None) -> int:
    num_bottom = int(sum_matrix.shape[1])
    if bottom_start_idx is None:
        bottom_start_idx = num_nodes - num_bottom
    bottom_start_idx = int(bottom_start_idx)
    if bottom_start_idx < 0 or bottom_start_idx + num_bottom > num_nodes:
        raise ValueError(
            f"Invalid bottom index range [{bottom_start_idx}, {bottom_start_idx + num_bottom}) "
            f"for num_nodes={num_nodes} and num_bottom={num_bottom}."
        )
    return bottom_start_idx


def reconcile_bottom_up(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int | None = None,
    force_nonnegative_bottom: bool = False,
) -> np.ndarray:
    """Bottom-Up reconciliation via y_tilde = S @ y_bottom."""
    y_hat = _as_3d(base_predictions)
    S = np.asarray(sum_matrix, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"sum_matrix must be 2D, got {S.shape}")
    if S.shape[0] != y_hat.shape[1]:
        raise ValueError(f"sum_matrix rows {S.shape[0]} != node count {y_hat.shape[1]}")

    bottom_start = _resolve_bottom_start(S, y_hat.shape[1], bottom_start_idx)
    num_bottom = S.shape[1]
    bottom = y_hat[:, bottom_start:bottom_start + num_bottom, :]
    if force_nonnegative_bottom:
        bottom = np.clip(bottom, 0.0, None)
    return np.einsum("nb,sbh->snh", S, bottom)


def reconcile_top_down(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int | None = None,
    mode: str = "forecast_proportions",
    epsilon: float = 1e-8,
    force_nonnegative_reference: bool = True,
) -> np.ndarray:
    """Top-Down reconciliation.

    Supported modes:
      - forecast_proportions: per-sample/horizon shares from base bottom forecasts.
      - average_proportions: one global share vector averaged over all samples/horizons.
    """
    y_hat = _as_3d(base_predictions)
    S = np.asarray(sum_matrix, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"sum_matrix must be 2D, got {S.shape}")
    if S.shape[0] != y_hat.shape[1]:
        raise ValueError(f"sum_matrix rows {S.shape[0]} != node count {y_hat.shape[1]}")

    bottom_start = _resolve_bottom_start(S, y_hat.shape[1], bottom_start_idx)
    num_bottom = S.shape[1]

    top = y_hat[:, 0:1, :]  # [S,1,H]
    ref_bottom = y_hat[:, bottom_start:bottom_start + num_bottom, :]  # [S,B,H]
    if force_nonnegative_reference:
        ref_bottom = np.clip(ref_bottom, 0.0, None)

    mode = str(mode).lower().strip()
    if mode == "forecast_proportions":
        denom = ref_bottom.sum(axis=1, keepdims=True)  # [S,1,H]
        safe_denom = np.where(np.abs(denom) < epsilon, 1.0, denom)
        p = ref_bottom / safe_denom
        if np.any(np.abs(denom) < epsilon):
            uniform = 1.0 / max(1, num_bottom)
            zero_mask = np.abs(denom) < epsilon
            p = np.where(zero_mask, uniform, p)
        bottom_td = p * top
    elif mode == "average_proportions":
        p_avg = ref_bottom.mean(axis=(0, 2))  # [B]
        p_sum = float(np.sum(p_avg))
        if np.abs(p_sum) < epsilon:
            p_avg = np.full((num_bottom,), 1.0 / max(1, num_bottom), dtype=np.float64)
        else:
            p_avg = p_avg / p_sum
        bottom_td = top * p_avg[None, :, None]
    else:
        raise ValueError(f"Unsupported TD mode '{mode}'.")

    return np.einsum("nb,sbh->snh", S, bottom_td)


def estimate_error_covariance(
    errors: np.ndarray,
    mode: str = "shrink",
    ridge: float = 1e-6,
) -> np.ndarray:
    """Estimate base forecast error covariance W used in MinT.

    Args:
        errors: shape [S,N,H] or [S,N] (forecast errors: y - y_hat).
        mode:
          - identity: W = I
          - diag: W = diag(var_i)
          - sample: empirical covariance
          - shrink: Ledoit-Wolf shrinkage (fallback to fixed shrink)
    """
    e = _as_3d(errors)
    num_nodes = int(e.shape[1])
    mode = str(mode).lower().strip()

    if mode == "identity":
        W = np.eye(num_nodes, dtype=np.float64)
    else:
        flat_e = e.transpose(0, 2, 1).reshape(-1, num_nodes)
        if flat_e.shape[0] <= 1:
            variances = np.ones((num_nodes,), dtype=np.float64)
        else:
            variances = np.var(flat_e, axis=0, ddof=1)
        variances = np.where(variances <= 0.0, 1e-8, variances)

        if mode == "diag":
            W = np.diag(variances)
        elif mode == "sample":
            if flat_e.shape[0] <= 1:
                W = np.diag(variances)
            else:
                W = np.cov(flat_e, rowvar=False)
        elif mode == "shrink":
            if flat_e.shape[0] <= 1:
                W = np.diag(variances)
            elif _HAS_LEDOIT_WOLF:
                W = LedoitWolf(assume_centered=False).fit(flat_e).covariance_
            else:
                sample = np.cov(flat_e, rowvar=False)
                target = np.diag(np.diag(sample))
                lam = 0.5
                W = (1.0 - lam) * sample + lam * target
        else:
            raise ValueError(f"Unsupported covariance mode '{mode}'.")

    W = np.asarray(W, dtype=np.float64)
    if W.shape != (num_nodes, num_nodes):
        raise ValueError(f"Covariance shape {W.shape} != ({num_nodes}, {num_nodes})")

    ridge = float(ridge)
    if ridge > 0:
        scale = float(np.trace(W) / max(1, num_nodes))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        W = W + np.eye(num_nodes, dtype=np.float64) * (ridge * scale)
    return W


def reconcile_mint(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    error_covariance: np.ndarray,
) -> np.ndarray:
    """MinT reconciliation: y_tilde = S G y_hat, G=(S'W^-1S)^-1 S'W^-1."""
    y_hat = _as_3d(base_predictions)
    S = np.asarray(sum_matrix, dtype=np.float64)
    W = np.asarray(error_covariance, dtype=np.float64)

    n = y_hat.shape[1]
    if S.shape[0] != n:
        raise ValueError(f"sum_matrix rows {S.shape[0]} != node count {n}")
    if W.shape != (n, n):
        raise ValueError(f"error_covariance shape {W.shape} != ({n}, {n})")

    W_inv = np.linalg.pinv(W)
    middle = S.T @ W_inv @ S
    G = np.linalg.pinv(middle) @ S.T @ W_inv  # [B,N]

    bottom_tilde = np.einsum("bn,snh->sbh", G, y_hat)  # [S,B,H]
    return np.einsum("nb,sbh->snh", S, bottom_tilde)


def apply_reconciliation(
    method: str,
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    bottom_start_idx: int | None = None,
    td_mode: str = "forecast_proportions",
    mint_cov_mode: str = "shrink",
    true_values: np.ndarray | None = None,
    mint_error_cov: np.ndarray | None = None,
) -> np.ndarray:
    """Unified entry for post-hoc reconciliation methods."""
    method = str(method).lower().strip()
    if method in {"none", "base", "no"}:
        return _as_3d(base_predictions)
    if method == "bu":
        return reconcile_bottom_up(base_predictions, sum_matrix, bottom_start_idx=bottom_start_idx)
    if method == "td":
        return reconcile_top_down(
            base_predictions,
            sum_matrix,
            bottom_start_idx=bottom_start_idx,
            mode=td_mode,
        )
    if method == "mint":
        if mint_error_cov is None:
            if true_values is None:
                if mint_cov_mode == "identity":
                    errors = np.zeros_like(_as_3d(base_predictions))
                else:
                    raise ValueError(
                        "MinT requires true_values (for error covariance estimation) "
                        "or explicit mint_error_cov."
                    )
            else:
                y_true = _as_3d(true_values)
                y_hat = _as_3d(base_predictions)
                if y_true.shape != y_hat.shape:
                    raise ValueError(
                        f"true_values shape {y_true.shape} != base_predictions shape {y_hat.shape}"
                    )
                errors = y_true - y_hat
            mint_error_cov = estimate_error_covariance(errors, mode=mint_cov_mode)
        return reconcile_mint(base_predictions, sum_matrix, mint_error_cov)
    raise ValueError(f"Unsupported reconciliation method '{method}'.")
