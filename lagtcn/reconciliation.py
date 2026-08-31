"""Final Applied Energy post-hoc reconciliation operators (nonnegative + coherent).

All three operators impose nonnegativity in bottom space and rebuild every
level through y_tilde = S @ b_tilde. Because S has nonnegative entries and an
identity block for the bottom level, the output is nonnegative and exactly
coherent by construction:

  - BU:      b_tilde = max(b_hat, 0)
  - TD-FP:   p = max(b_hat,0) / sum(max(b_hat,0))  (uniform if the sum is 0),
             b_tilde = p * max(top_hat, 0)
  - MinT-SHR: b_tilde = argmin_{b >= 0} (y_hat - S b)' W^{-1} (y_hat - S b),
              solved per sample/horizon by NNLS, where W is a Ledoit-Wolf
              covariance estimated from validation residuals.

Every operator returns (y_tilde, diagnostics). This module is the single
implementation shared by Phase 1 post-processing and the Phase 2 pipeline; it
is pure numpy/scipy and testable on synthetic fixtures.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

RECONCILE_AE_VERSION = "ae_reconcile_nonneg_v2"

AE_METHODS = ("bu", "td_fp", "mint_shrink")

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


def estimate_mint_shrink_weight(
    validation_predictions: np.ndarray,
    validation_true_values: np.ndarray,
) -> np.ndarray:
    """Estimate the MinT-SHR covariance using validation residuals only."""
    y_hat = _as_3d(validation_predictions)
    y_true = _as_3d(validation_true_values)
    if y_hat.shape != y_true.shape:
        raise ValueError(f"true_values shape {y_true.shape} != base_predictions shape {y_hat.shape}")
    errors = (y_true - y_hat).transpose(0, 2, 1).reshape(-1, y_hat.shape[1])
    try:
        from sklearn.covariance import LedoitWolf
        weight = LedoitWolf(assume_centered=False).fit(errors).covariance_
    except ImportError:
        sample = np.cov(errors, rowvar=False)
        weight = 0.5 * sample + 0.5 * np.diag(np.diag(sample))
    return np.asarray(weight, dtype=np.float64)


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


def reconcile_mint_shrink(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    *,
    mint_weight: np.ndarray,
    bottom_start_idx: int | None = None,
    nnls_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """Nonnegative MinT-SHR: per sample/horizon NNLS in bottom space.

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

    L = _whitening_factor(mint_weight, num_nodes)
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
        "method": "mint_shrink",
        "solver": "scipy.optimize.nnls",
        "weight_mode": "ledoit_wolf_shrinkage",
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
    mint_weight: np.ndarray | None = None,
    nnls_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """Apply one of the three reported methods: BU, TD-FP, or MinT-SHR.

    ``mint_weight`` must be estimated from validation residuals with
    :func:`estimate_mint_shrink_weight`; requiring it here prevents accidental
    covariance estimation on test targets.
    """
    name = str(method).lower().strip().replace("-", "_")
    if name == "bu":
        return reconcile_bu(base_predictions, sum_matrix, bottom_start_idx=bottom_start_idx)
    if name in {"td_fp", "tdfp", "td"}:
        return reconcile_td_fp(base_predictions, sum_matrix, bottom_start_idx=bottom_start_idx)
    if name == "mint_shrink":
        if mint_weight is None:
            raise ValueError(
                "mint_shrink requires a covariance matrix estimated from validation data"
            )
        return reconcile_mint_shrink(
            base_predictions,
            sum_matrix,
            mint_weight=mint_weight,
            bottom_start_idx=bottom_start_idx,
            nnls_workers=nnls_workers,
        )
    raise ValueError(f"Unsupported AE reconciliation method '{method}'.")


def _load_prediction_archive(path: Path) -> tuple[np.lib.npyio.NpzFile, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    if "predictions" not in archive.files:
        raise KeyError(f"{path} does not contain a 'predictions' array")
    return archive, archive["predictions"]


def main(argv: list[str] | None = None) -> None:
    """Command-line interface for reconciling a saved LAGTCN forecast."""
    parser = argparse.ArgumentParser(description="Apply BU, TD-FP, or MinT-SHR")
    parser.add_argument("--method", choices=AE_METHODS, required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--sum-matrix", type=Path, required=True)
    parser.add_argument(
        "--validation-archive",
        type=Path,
        help="Required for MinT-SHR; contains validation predictions and true_values.",
    )
    parser.add_argument("--bottom-start-idx", type=int, default=None)
    parser.add_argument("--nnls-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    base_archive, base_predictions = _load_prediction_archive(args.base_archive)
    sum_matrix = np.genfromtxt(args.sum_matrix, delimiter=",", dtype=np.float64)
    if sum_matrix.ndim == 1:
        sum_matrix = sum_matrix.reshape(-1, 1)
    if sum_matrix.ndim == 2:
        sum_matrix = sum_matrix[:, ~np.all(np.isnan(sum_matrix), axis=0)]
    mint_weight = None
    if args.method == "mint_shrink":
        if args.validation_archive is None:
            parser.error("--validation-archive is required for mint_shrink")
        validation_archive, validation_predictions = _load_prediction_archive(
            args.validation_archive
        )
        if "true_values" not in validation_archive.files:
            raise KeyError("validation archive does not contain 'true_values'")
        mint_weight = estimate_mint_shrink_weight(
            validation_predictions,
            validation_archive["true_values"],
        )

    reconciled, diagnostics = apply_reconciliation_ae(
        args.method,
        base_predictions,
        sum_matrix,
        bottom_start_idx=args.bottom_start_idx,
        mint_weight=mint_weight,
        nnls_workers=args.nnls_workers,
    )
    if args.method == "mint_shrink":
        diagnostics["covariance_estimation"] = "validation_residuals_only"
        diagnostics["validation_archive"] = str(args.validation_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"predictions": reconciled.astype(np.float64)}
    for key in ("true_values", "target_timestamps", "node_names"):
        if key in base_archive.files:
            payload[key] = base_archive[key]
    np.savez_compressed(args.output, **payload)
    diagnostics_path = args.output.with_suffix(".json")
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
