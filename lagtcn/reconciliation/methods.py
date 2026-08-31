"""Final Applied Energy post-hoc reconciliation operators (nonnegative + coherent).

All three operators impose nonnegativity in bottom space and rebuild every
level through y_tilde = S @ b_tilde. Because S has nonnegative entries and an
identity block for the bottom level, the output is nonnegative and exactly
coherent by construction:

  - BU:      b_tilde = max(b_hat, 0)
  - TD-FP:   p = max(b_hat,0) / sum(max(b_hat,0))  (uniform if the sum is 0),
             b_tilde = p * max(top_hat, 0)
  - MinT:    b_tilde = argmin_{b >= 0} (y_hat - S b)' W^{-1} (y_hat - S b)
             solved per sample/horizon by NNLS; W = I for OLS, diag error
             variances for WLS, Ledoit-Wolf covariance for shrink.

Every operator returns (y_tilde, diagnostics). This module is the single
implementation shared by Phase 1 post-processing and the Phase 2 pipeline; it
is pure numpy/scipy and testable on synthetic fixtures.
"""
from __future__ import annotations

import contextlib
import multiprocessing as mp
import time

import numpy as np

RECONCILE_AE_VERSION = "ae_reconcile_nonneg_v2"

AE_METHODS = ("bu", "td_fp", "mint_ols")

NNLS_PARALLEL_THRESHOLD = 1000
_NNLS_WORKER_STATE = None
_NNLS_WORKER_LIMITER = None


def _init_nnls_worker(
    design: np.ndarray,
    targets: np.ndarray,
    maxiter: int,
    atol: float,
) -> None:
    """Initialize fork workers while pinning each worker's BLAS to one thread."""
    global _NNLS_WORKER_STATE, _NNLS_WORKER_LIMITER
    _NNLS_WORKER_STATE = (design, targets, maxiter, atol)
    try:
        from threadpoolctl import threadpool_limits
        _NNLS_WORKER_LIMITER = threadpool_limits(limits=1, user_api="blas")
    except ImportError:
        _NNLS_WORKER_LIMITER = None


def _solve_nnls_worker(column_index: int) -> tuple[int, np.ndarray | None]:
    """Solve one column in a worker; None preserves the clipped fallback."""
    from scipy.optimize import nnls

    design, targets, maxiter, atol = _NNLS_WORKER_STATE
    try:
        solution, _ = nnls(
            design, targets[column_index], maxiter=maxiter, atol=atol
        )
        return column_index, solution
    except Exception:
        return column_index, None



def _as_3d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Expected predictions with 2D/3D shape, got {arr.shape}")
    return arr


def _validate_structure(y_hat: np.ndarray, sum_matrix: np.ndarray, bottom_start_idx: int | None) -> tuple[np.ndarray, int]:
    S = np.asarray(sum_matrix, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"sum_matrix must be 2D, got {S.shape}")
    num_nodes = y_hat.shape[1]
    num_bottom = S.shape[1]
    if S.shape[0] != num_nodes:
        raise ValueError(f"sum_matrix rows {S.shape[0]} != node count {num_nodes}")
    if np.any(S < 0):
        raise ValueError("sum_matrix must be nonnegative for nonnegativity-by-construction.")
    if bottom_start_idx is None:
        bottom_start_idx = num_nodes - num_bottom
    bottom_start_idx = int(bottom_start_idx)
    if bottom_start_idx < 0 or bottom_start_idx + num_bottom > num_nodes:
        raise ValueError(
            f"Invalid bottom index range [{bottom_start_idx}, {bottom_start_idx + num_bottom}) "
            f"for num_nodes={num_nodes}."
        )
    identity_block = S[bottom_start_idx:bottom_start_idx + num_bottom, :]
    if not np.array_equal(identity_block, np.eye(num_bottom)):
        raise ValueError("sum_matrix bottom block must be the identity (bottom rows map to themselves).")
    return S, bottom_start_idx


def _coherence_stats(y_tilde: np.ndarray, S: np.ndarray, bottom_start: int) -> dict:
    num_bottom = S.shape[1]
    bottom = y_tilde[:, bottom_start:bottom_start + num_bottom, :]
    residual = y_tilde - np.einsum("nb,sbh->snh", S, bottom)
    return {
        "coherence_residual_max_abs": float(np.max(np.abs(residual))) if residual.size else 0.0,
        "coherence_residual_mean_abs": float(np.mean(np.abs(residual))) if residual.size else 0.0,
    }


def coherence_stats(
    predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    bottom_start_idx: int | None = None,
) -> dict:
    """Coherence residual and minimum of an arbitrary forecast block.

    Intended for diagnosing base forecasts (and sanity-checking true values)
    against the same summing matrix used for reconciliation.
    """
    values = _as_3d(predictions)
    S, bottom_start = _validate_structure(values, sum_matrix, bottom_start_idx)
    stats = _coherence_stats(values, S, bottom_start)
    stats["min_value"] = float(values.min()) if values.size else 0.0
    return stats


def _finalize(b_tilde: np.ndarray, S: np.ndarray, bottom_start: int, diag: dict, start_time: float) -> tuple[np.ndarray, dict]:
    y_tilde = np.einsum("nb,sbh->snh", S, b_tilde)
    diag.update(_coherence_stats(y_tilde, S, bottom_start))
    diag["min_prediction"] = float(y_tilde.min()) if y_tilde.size else 0.0
    diag["runtime_sec"] = float(time.perf_counter() - start_time)
    diag["reconcile_version"] = RECONCILE_AE_VERSION
    return y_tilde, diag


def reconcile_bu(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    bottom_start_idx: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Nonnegative Bottom-Up: b_tilde = max(b_hat, 0), y_tilde = S b_tilde."""
    start = time.perf_counter()
    y_hat = _as_3d(base_predictions)
    S, bottom_start = _validate_structure(y_hat, sum_matrix, bottom_start_idx)
    num_bottom = S.shape[1]

    bottom = y_hat[:, bottom_start:bottom_start + num_bottom, :]
    negative = bottom < 0
    b_tilde = np.clip(bottom, 0.0, None)
    diag = {
        "method": "bu",
        "solver": "positive_part",
        "n_samples": int(y_hat.shape[0]),
        "n_horizons": int(y_hat.shape[2]),
        "n_negative_bottom_clipped": int(negative.sum()),
        "negative_bottom_fraction": float(negative.mean()) if negative.size else 0.0,
        "n_failures": 0,
    }
    return _finalize(b_tilde, S, bottom_start, diag, start)


def reconcile_td_fp(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    bottom_start_idx: int | None = None,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, dict]:
    """Nonnegative forecast-proportion Top-Down.

    Proportions come from the positive part of the bottom base forecasts
    (uniform when their sum is zero); the disaggregated total is the positive
    part of the top base forecast.
    """
    start = time.perf_counter()
    y_hat = _as_3d(base_predictions)
    S, bottom_start = _validate_structure(y_hat, sum_matrix, bottom_start_idx)
    num_bottom = S.shape[1]

    top_hat = y_hat[:, 0:1, :]
    negative_top = top_hat < 0
    top_plus = np.clip(top_hat, 0.0, None)  # [S,1,H]

    bottom_plus = np.clip(y_hat[:, bottom_start:bottom_start + num_bottom, :], 0.0, None)
    denom = bottom_plus.sum(axis=1, keepdims=True)  # [S,1,H]
    zero_denom = denom <= epsilon
    safe_denom = np.where(zero_denom, 1.0, denom)
    proportions = np.where(zero_denom, 1.0 / num_bottom, bottom_plus / safe_denom)
    b_tilde = proportions * top_plus

    diag = {
        "method": "td_fp",
        "solver": "forecast_proportions_positive_part",
        "n_samples": int(y_hat.shape[0]),
        "n_horizons": int(y_hat.shape[2]),
        "n_negative_top_clipped": int(negative_top.sum()),
        "n_zero_denominator_columns": int(zero_denom.sum()),
        "epsilon": float(epsilon),
        "n_failures": 0,
    }
    return _finalize(b_tilde, S, bottom_start, diag, start)


def estimate_mint_weight(
    weight_mode: str,
    *,
    base_predictions: np.ndarray | None = None,
    true_values: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return W for the MinT objective, or None for the identity (OLS)."""
    mode = str(weight_mode).lower().strip()
    if mode in {"ols", "identity"}:
        return None
    if base_predictions is None or true_values is None:
        raise ValueError(f"weight_mode='{mode}' requires base_predictions and true_values.")
    y_hat = _as_3d(base_predictions)
    y_true = _as_3d(true_values)
    if y_hat.shape != y_true.shape:
        raise ValueError(f"true_values shape {y_true.shape} != base_predictions shape {y_hat.shape}")
    errors = (y_true - y_hat).transpose(0, 2, 1).reshape(-1, y_hat.shape[1])
    if mode in {"wls", "wls_var", "diag"}:
        variances = np.var(errors, axis=0, ddof=1) if errors.shape[0] > 1 else np.ones(y_hat.shape[1])
        variances = np.where(variances <= 0.0, 1e-8, variances)
        return np.diag(variances)
    if mode in {"shrink", "mint_shrink"}:
        try:
            from sklearn.covariance import LedoitWolf
            W = LedoitWolf(assume_centered=False).fit(errors).covariance_
        except ImportError:
            sample = np.cov(errors, rowvar=False)
            W = 0.5 * sample + 0.5 * np.diag(np.diag(sample))
        return np.asarray(W, dtype=np.float64)
    raise ValueError(f"Unsupported MinT weight_mode '{weight_mode}'.")


def _whitening_factor(W: np.ndarray | None, num_nodes: int, ridge: float = 1e-10) -> np.ndarray | None:
    """L such that L'L = W^{-1}; None means identity."""
    if W is None:
        return None
    W = np.asarray(W, dtype=np.float64)
    if W.shape != (num_nodes, num_nodes):
        raise ValueError(f"W shape {W.shape} != ({num_nodes}, {num_nodes})")
    eigenvalues, eigenvectors = np.linalg.eigh(W)
    floor = max(ridge, ridge * float(eigenvalues.max())) if eigenvalues.size else ridge
    eigenvalues = np.clip(eigenvalues, floor, None)
    return (eigenvectors / np.sqrt(eigenvalues)).T  # rows scaled by 1/sqrt(lambda)


def reconcile_mint_nnls(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    bottom_start_idx: int | None = None,
    weight_mode: str = "ols",
    W: np.ndarray | None = None,
    true_values: np.ndarray | None = None,
    nnls_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """Nonnegative MinT: per sample/horizon NNLS in bottom space.

    A column whose unconstrained MinT solution is already nonnegative keeps
    that solution — it is the exact NNLS optimum there — so NNLS runs only
    where the nonnegativity constraint is active (any strictly negative
    entry). Columns where NNLS fails fall back to the clipped unconstrained
    solution (still nonnegative and coherent) and are counted in
    ``n_failures``.
    """
    from scipy.optimize import nnls

    start = time.perf_counter()
    y_hat = _as_3d(base_predictions)
    S, bottom_start = _validate_structure(y_hat, sum_matrix, bottom_start_idx)
    num_samples, num_nodes, num_horizons = y_hat.shape
    num_bottom = S.shape[1]

    if W is None:
        W = estimate_mint_weight(weight_mode, base_predictions=y_hat, true_values=true_values)
    L = _whitening_factor(W, num_nodes)
    S_w = S if L is None else L @ S
    G = np.linalg.pinv(S_w)  # [B, N]; SVD-based, avoids squaring the condition number
    nnls_maxiter = 3 * num_bottom
    # scipy.optimize.nnls documents this scale-aware value as the typical
    # relaxation for the projected-residual KKT test. Freeze it explicitly so
    # diagnostics are reproducible instead of hiding a solver default.
    nnls_atol = float(
        max(S_w.shape) * np.linalg.norm(S_w, 1) * np.spacing(np.float64(1.0))
    )

    columns = y_hat.transpose(0, 2, 1).reshape(-1, num_nodes)  # [(S*H), N]
    whitened = columns if L is None else columns @ L.T
    b_unconstrained = whitened @ G.T  # [(S*H), B]

    needs_nnls = b_unconstrained.min(axis=1) < 0.0
    b_solution = np.clip(b_unconstrained, 0.0, None)  # exact where needs_nnls is False

    nnls_indices = np.flatnonzero(needs_nnls)
    requested_workers = max(1, int(nnls_workers))
    fork_available = "fork" in mp.get_all_start_methods()
    use_parallel = (
        requested_workers > 1
        and nnls_indices.size >= NNLS_PARALLEL_THRESHOLD
        and fork_available
    )
    effective_workers = min(requested_workers, int(nnls_indices.size)) if use_parallel else 1
    parallel_start_method = "fork" if use_parallel else None
    pool_chunksize = None
    blas_thread_limit = 1

    n_failures = 0
    if use_parallel:
        pool_chunksize = max(
            1, int(nnls_indices.size) // (effective_workers * 32)
        )
        context = mp.get_context("fork")
        with context.Pool(
            processes=effective_workers,
            initializer=_init_nnls_worker,
            initargs=(S_w, whitened, nnls_maxiter, nnls_atol),
        ) as pool:
            for col, solution in pool.imap_unordered(
                _solve_nnls_worker,
                nnls_indices.tolist(),
                chunksize=pool_chunksize,
            ):
                if solution is None:
                    n_failures += 1
                else:
                    b_solution[col] = solution
    else:
        try:
            from threadpoolctl import threadpool_limits
            blas_context = threadpool_limits(limits=1, user_api="blas")
        except ImportError:
            blas_context = contextlib.nullcontext()
            blas_thread_limit = None
        with blas_context:
            for col in nnls_indices:
                try:
                    b_solution[col], _ = nnls(
                        S_w, whitened[col], maxiter=nnls_maxiter, atol=nnls_atol
                    )
                except Exception:
                    n_failures += 1  # keep the clipped unconstrained fallback
    b_tilde = b_solution.reshape(num_samples, num_horizons, num_bottom).transpose(0, 2, 1)

    diag = {
        "method": f"mint_{str(weight_mode).lower().strip()}",
        "solver": "scipy.optimize.nnls",
        "weight_mode": str(weight_mode).lower().strip(),
        "n_samples": int(num_samples),
        "n_horizons": int(num_horizons),
        "n_columns": int(columns.shape[0]),
        "n_nnls_solves": int(needs_nnls.sum()),
        "nnls_maxiter": int(nnls_maxiter),
        "nnls_atol": nnls_atol,
        "blas_thread_limit": blas_thread_limit,
        "nnls_workers_requested": requested_workers,
        "nnls_workers_effective": effective_workers,
        "nnls_parallel_threshold": NNLS_PARALLEL_THRESHOLD,
        "nnls_parallel_start_method": parallel_start_method,
        "nnls_pool_chunksize": pool_chunksize,
        "fork_available": fork_available,
        "unconstrained_feasibility_tolerance": 0.0,
        "n_failures": int(n_failures),
        "failure_fallback": "clipped_unconstrained" if n_failures else None,
    }
    return _finalize(b_tilde, S, bottom_start, diag, start)


def apply_reconciliation_ae(
    method: str,
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    bottom_start_idx: int | None = None,
    true_values: np.ndarray | None = None,
    mint_weight: np.ndarray | None = None,
    nnls_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """Unified Applied Energy entry point.

    Methods: 'bu', 'td_fp', 'mint_ols', 'mint_wls', 'mint_shrink'.
    """
    name = str(method).lower().strip().replace("-", "_")
    if name == "bu":
        return reconcile_bu(base_predictions, sum_matrix, bottom_start_idx=bottom_start_idx)
    if name in {"td_fp", "tdfp", "td"}:
        return reconcile_td_fp(base_predictions, sum_matrix, bottom_start_idx=bottom_start_idx)
    if name.startswith("mint"):
        weight_mode = name.split("_", 1)[1] if "_" in name else "ols"
        return reconcile_mint_nnls(
            base_predictions,
            sum_matrix,
            bottom_start_idx=bottom_start_idx,
            weight_mode=weight_mode,
            W=mint_weight,
            true_values=true_values,
            nnls_workers=nnls_workers,
        )
    raise ValueError(f"Unsupported AE reconciliation method '{method}'.")
