"""Classical (post-hoc) hierarchical reconciliation methods.

Implements Bottom-Up, Top-Down (proportions), and MinT (OLS / WLS / SHR)
reconciliation that can be applied to *any* set of base forecasts.

All methods take:
    base_forecasts : np.ndarray  [T, N]  or  [T, N, H]
    sum_matrix     : np.ndarray  [N, B]  (S matrix: maps bottom -> all nodes)
and return reconciled forecasts of the same shape.

References:
    - Wickramasuriya et al. (2019) "Optimal forecast reconciliation ..."
    - Hyndman & Athanasopoulos, "Forecasting: Principles and Practice", Ch 11
"""
from __future__ import annotations

import numpy as np


def _ensure_3d(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x[:, :, np.newaxis]
    return x


def _squeeze_if_needed(x: np.ndarray, was_2d: bool) -> np.ndarray:
    if was_2d:
        return x[:, :, 0]
    return x


# ---------------------------------------------------------------------------
# Bottom-Up (classical)
# ---------------------------------------------------------------------------

def reconcile_bottom_up(
    base_forecasts: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int | None = None,
) -> np.ndarray:
    """Classical bottom-up: use only bottom-level base forecasts, aggregate via S."""
    was_2d = base_forecasts.ndim == 2
    bf = _ensure_3d(base_forecasts)  # [T, N, H]
    S = np.asarray(sum_matrix, dtype=np.float64)  # [N, B]
    B = S.shape[1]
    if bottom_start_idx is None:
        bottom_start_idx = bf.shape[1] - B

    bottom = bf[:, bottom_start_idx:bottom_start_idx + B, :]  # [T, B, H]
    reconciled = np.einsum("nb,tbh->tnh", S, bottom)  # [T, N, H]
    return _squeeze_if_needed(reconciled, was_2d)


# ---------------------------------------------------------------------------
# Top-Down (historical proportions)
# ---------------------------------------------------------------------------

def reconcile_top_down(
    base_forecasts: np.ndarray,
    sum_matrix: np.ndarray,
    train_actuals: np.ndarray,
    bottom_start_idx: int | None = None,
    mid_to_bottom_indices: list | None = None,
) -> np.ndarray:
    """Classical top-down using historical average proportions.

    Args:
        train_actuals: [T_train, N] or [T_train, N, H] — actuals from training set
            used to compute historical proportions.
    """
    was_2d = base_forecasts.ndim == 2
    bf = _ensure_3d(base_forecasts)  # [T, N, H]
    ta = _ensure_3d(train_actuals)
    S = np.asarray(sum_matrix, dtype=np.float64)
    B = S.shape[1]
    N = bf.shape[1]
    if bottom_start_idx is None:
        bottom_start_idx = N - B

    # Compute historical proportions from training data (average over time & horizons)
    train_bottom = ta[:, bottom_start_idx:bottom_start_idx + B, :]  # [T_train, B, H]
    train_top = ta[:, 0:1, :]  # [T_train, 1, H]

    # Average proportions: p_i = mean(bottom_i) / mean(top)
    mean_bottom = train_bottom.mean(axis=(0, 2))  # [B]
    mean_top = train_top.mean(axis=(0, 2))  # scalar
    eps = 1e-8
    proportions = mean_bottom / max(mean_top, eps)  # [B]
    proportions = proportions / max(proportions.sum(), eps)  # normalize to sum=1

    # Apply: reconciled_bottom = top_forecast * proportions
    top_forecast = bf[:, 0:1, :]  # [T, 1, H]
    reconciled_bottom = top_forecast * proportions[np.newaxis, :, np.newaxis]  # [T, B, H]

    # Aggregate via S
    reconciled = np.einsum("nb,tbh->tnh", S, reconciled_bottom)  # [T, N, H]
    return _squeeze_if_needed(reconciled, was_2d)


# ---------------------------------------------------------------------------
# MinT reconciliation
# ---------------------------------------------------------------------------

def _compute_mint_P(
    S: np.ndarray,
    W_inv: np.ndarray,
) -> np.ndarray:
    """Compute the MinT projection matrix P = S (S'W^{-1}S)^{-1} S'W^{-1}.

    S: [N, B], W_inv: [N, N] (inverse of covariance/weight matrix)
    Returns P: [N, N]
    """
    # P = S @ inv(S.T @ W_inv @ S) @ S.T @ W_inv
    StWi = S.T @ W_inv  # [B, N]
    StWiS = StWi @ S  # [B, B]
    try:
        StWiS_inv = np.linalg.inv(StWiS)
    except np.linalg.LinAlgError:
        StWiS_inv = np.linalg.pinv(StWiS)
    P = S @ StWiS_inv @ StWi  # [N, N]
    return P


def reconcile_mint_ols(
    base_forecasts: np.ndarray,
    sum_matrix: np.ndarray,
) -> np.ndarray:
    """MinT-OLS: W = I (identity covariance)."""
    was_2d = base_forecasts.ndim == 2
    bf = _ensure_3d(base_forecasts)  # [T, N, H]
    S = np.asarray(sum_matrix, dtype=np.float64)
    N = S.shape[0]

    W_inv = np.eye(N)
    P = _compute_mint_P(S, W_inv)  # [N, N]

    # Apply: reconciled = P @ base for each (t, h)
    reconciled = np.einsum("nm,tmh->tnh", P, bf)
    return _squeeze_if_needed(reconciled, was_2d)


def reconcile_mint_wls(
    base_forecasts: np.ndarray,
    sum_matrix: np.ndarray,
    residuals: np.ndarray | None = None,
) -> np.ndarray:
    """MinT-WLS: W = diag(var(residuals)) — variance scaling.

    Args:
        residuals: [T_train, N] or [T_train, N, H] — in-sample residuals.
            If None, uses equal weights (falls back to OLS).
    """
    was_2d = base_forecasts.ndim == 2
    bf = _ensure_3d(base_forecasts)
    S = np.asarray(sum_matrix, dtype=np.float64)
    N = S.shape[0]

    if residuals is None:
        return reconcile_mint_ols(base_forecasts, sum_matrix)

    res = _ensure_3d(residuals)
    # Variance per node (across time and horizons)
    var_per_node = res.reshape(-1, N).var(axis=0)  # [N]
    var_per_node = np.maximum(var_per_node, 1e-8)

    W_inv = np.diag(1.0 / var_per_node)  # [N, N]
    P = _compute_mint_P(S, W_inv)

    reconciled = np.einsum("nm,tmh->tnh", P, bf)
    return _squeeze_if_needed(reconciled, was_2d)


def reconcile_mint_shr(
    base_forecasts: np.ndarray,
    sum_matrix: np.ndarray,
    residuals: np.ndarray | None = None,
) -> np.ndarray:
    """MinT-SHR: shrinkage estimator for W.

    Uses the Ledoit-Wolf shrinkage to estimate the covariance matrix
    of the residuals, providing a regularized version of MinT.

    Args:
        residuals: [T_train, N] or [T_train, N, H] — in-sample residuals.
            If None, falls back to OLS.
    """
    was_2d = base_forecasts.ndim == 2
    bf = _ensure_3d(base_forecasts)
    S = np.asarray(sum_matrix, dtype=np.float64)
    N = S.shape[0]

    if residuals is None:
        return reconcile_mint_ols(base_forecasts, sum_matrix)

    res = _ensure_3d(residuals)
    res_flat = res.reshape(-1, N)  # [T*H, N]
    T_eff = res_flat.shape[0]

    # Sample covariance
    sample_cov = np.cov(res_flat, rowvar=False)  # [N, N]

    # Ledoit-Wolf shrinkage target: diagonal of sample_cov
    target = np.diag(np.diag(sample_cov))

    # Estimate optimal shrinkage intensity (simplified Ledoit-Wolf)
    # Using the formula from Ledoit & Wolf (2004)
    X = res_flat - res_flat.mean(axis=0)
    S2 = (X ** 2).T @ (X ** 2) / T_eff - sample_cov ** 2
    delta = np.sum(S2) / T_eff

    # Frobenius norm of (sample_cov - target)
    frob_sq = np.sum((sample_cov - target) ** 2)

    if frob_sq < 1e-12:
        shrinkage = 1.0
    else:
        shrinkage = np.clip(delta / frob_sq, 0.0, 1.0)

    W = shrinkage * target + (1 - shrinkage) * sample_cov

    # Regularize to ensure invertibility
    W += np.eye(N) * 1e-6

    try:
        W_inv = np.linalg.inv(W)
    except np.linalg.LinAlgError:
        W_inv = np.linalg.pinv(W)

    P = _compute_mint_P(S, W_inv)
    reconciled = np.einsum("nm,tmh->tnh", P, bf)
    return _squeeze_if_needed(reconciled, was_2d)


# ---------------------------------------------------------------------------
# Convenience: apply all methods at once
# ---------------------------------------------------------------------------

CLASSICAL_METHODS = {
    "BU_classical": reconcile_bottom_up,
    "TD_classical": reconcile_top_down,
    "MinT_OLS": reconcile_mint_ols,
    "MinT_WLS": reconcile_mint_wls,
    "MinT_SHR": reconcile_mint_shr,
}


def apply_all_classical(
    base_forecasts: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int | None = None,
    train_actuals: np.ndarray | None = None,
    residuals: np.ndarray | None = None,
    mid_to_bottom_indices: list | None = None,
) -> dict[str, np.ndarray]:
    """Apply all classical reconciliation methods and return results dict."""
    results = {}

    results["BU_classical"] = reconcile_bottom_up(
        base_forecasts, sum_matrix, bottom_start_idx
    )

    if train_actuals is not None:
        results["TD_classical"] = reconcile_top_down(
            base_forecasts, sum_matrix, train_actuals,
            bottom_start_idx, mid_to_bottom_indices,
        )

    results["MinT_OLS"] = reconcile_mint_ols(base_forecasts, sum_matrix)

    if residuals is not None:
        results["MinT_WLS"] = reconcile_mint_wls(
            base_forecasts, sum_matrix, residuals
        )
        results["MinT_SHR"] = reconcile_mint_shr(
            base_forecasts, sum_matrix, residuals
        )

    return results
