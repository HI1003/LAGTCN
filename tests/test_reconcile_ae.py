from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lagtcn.reconciliation import methods as reconcile_ae


def sum_matrix_2level() -> tuple[np.ndarray, int]:
    S = np.vstack([np.ones((1, 3)), np.eye(3)])
    return S, 1


def sum_matrix_3level() -> tuple[np.ndarray, int]:
    S = np.vstack([
        np.ones((1, 3)),
        np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
    ])
    return S, 3


def sum_matrix_4level() -> tuple[np.ndarray, int]:
    S = np.vstack([
        np.ones((1, 4)),
        np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]),
        np.eye(4),
    ])
    return S, 6


ALL_FIXTURES = [sum_matrix_2level, sum_matrix_3level, sum_matrix_4level]


def coherent_input(S: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    """y = S b for bottom [S,B,H]."""
    return np.einsum("nb,sbh->snh", S, bottom)


def assert_nonneg_and_coherent(test: unittest.TestCase, y_tilde: np.ndarray, S: np.ndarray, bottom_start: int) -> None:
    test.assertGreaterEqual(y_tilde.min(), 0.0)
    num_bottom = S.shape[1]
    rebuilt = np.einsum("nb,sbh->snh", S, y_tilde[:, bottom_start:bottom_start + num_bottom, :])
    np.testing.assert_allclose(y_tilde, rebuilt, atol=1e-8)


class BottomUpTest(unittest.TestCase):
    def test_negative_bottom_clipped_all_fixtures(self) -> None:
        rng = np.random.default_rng(0)
        for fixture in ALL_FIXTURES:
            S, bottom_start = fixture()
            n, B = S.shape
            y_hat = rng.normal(0.0, 5.0, size=(4, n, 2))
            y_tilde, diag = reconcile_ae.reconcile_bu(y_hat, S, bottom_start_idx=bottom_start)
            assert_nonneg_and_coherent(self, y_tilde, S, bottom_start)
            expected_bottom = np.clip(y_hat[:, bottom_start:bottom_start + B, :], 0.0, None)
            np.testing.assert_allclose(y_tilde[:, bottom_start:, :], expected_bottom)
            self.assertEqual(diag["n_negative_bottom_clipped"], int((y_hat[:, bottom_start:, :] < 0).sum()))
            self.assertLessEqual(diag["coherence_residual_max_abs"], 1e-8)

    def test_nonnegative_input_unchanged_bottom(self) -> None:
        S, bottom_start = sum_matrix_3level()
        bottom = np.abs(np.random.default_rng(1).normal(2.0, 1.0, size=(3, 3, 1)))
        y_hat = coherent_input(S, bottom)
        y_tilde, diag = reconcile_ae.reconcile_bu(y_hat, S, bottom_start_idx=bottom_start)
        np.testing.assert_allclose(y_tilde, y_hat, atol=1e-10)
        self.assertEqual(diag["n_negative_bottom_clipped"], 0)


class TopDownTest(unittest.TestCase):
    def test_proportions_and_coherence(self) -> None:
        S, bottom_start = sum_matrix_2level()
        y_hat = np.zeros((1, 4, 1))
        y_hat[0, 0, 0] = 12.0                 # top forecast
        y_hat[0, 1:, 0] = [1.0, 2.0, 3.0]     # bottom forecasts -> proportions 1/6, 2/6, 3/6
        y_tilde, diag = reconcile_ae.reconcile_td_fp(y_hat, S, bottom_start_idx=bottom_start)
        np.testing.assert_allclose(y_tilde[0, 1:, 0], [2.0, 4.0, 6.0])
        np.testing.assert_allclose(y_tilde[0, 0, 0], 12.0)
        assert_nonneg_and_coherent(self, y_tilde, S, bottom_start)
        self.assertEqual(diag["n_zero_denominator_columns"], 0)

    def test_negative_top_gives_zero_column(self) -> None:
        S, bottom_start = sum_matrix_3level()
        y_hat = np.ones((2, 6, 2))
        y_hat[0, 0, :] = -5.0
        y_tilde, diag = reconcile_ae.reconcile_td_fp(y_hat, S, bottom_start_idx=bottom_start)
        np.testing.assert_allclose(y_tilde[0], 0.0)
        self.assertGreater(y_tilde[1].max(), 0.0)
        self.assertEqual(diag["n_negative_top_clipped"], 2)
        assert_nonneg_and_coherent(self, y_tilde, S, bottom_start)

    def test_all_nonpositive_bottom_uses_uniform_proportions(self) -> None:
        S, bottom_start = sum_matrix_2level()
        y_hat = np.zeros((1, 4, 1))
        y_hat[0, 0, 0] = 9.0
        y_hat[0, 1:, 0] = [-1.0, -2.0, 0.0]
        y_tilde, diag = reconcile_ae.reconcile_td_fp(y_hat, S, bottom_start_idx=bottom_start)
        np.testing.assert_allclose(y_tilde[0, 1:, 0], [3.0, 3.0, 3.0])
        self.assertEqual(diag["n_zero_denominator_columns"], 1)
        assert_nonneg_and_coherent(self, y_tilde, S, bottom_start)


class MintNnlsTest(unittest.TestCase):
    def test_coherent_nonnegative_input_is_fixed_point(self) -> None:
        for fixture in ALL_FIXTURES:
            S, bottom_start = fixture()
            bottom = np.abs(np.random.default_rng(2).normal(3.0, 1.0, size=(3, S.shape[1], 2)))
            y_hat = coherent_input(S, bottom)
            y_tilde, diag = reconcile_ae.reconcile_mint_nnls(y_hat, S, bottom_start_idx=bottom_start)
            np.testing.assert_allclose(y_tilde, y_hat, atol=1e-7)
            self.assertEqual(diag["n_nnls_solves"], 0)
            self.assertEqual(diag["n_failures"], 0)

    def test_incoherent_input_projected_nonneg_coherent(self) -> None:
        S, bottom_start = sum_matrix_3level()
        rng = np.random.default_rng(3)
        y_hat = rng.normal(0.0, 10.0, size=(5, 6, 3))  # strongly incoherent, signed
        y_tilde, diag = reconcile_ae.reconcile_mint_nnls(y_hat, S, bottom_start_idx=bottom_start)
        assert_nonneg_and_coherent(self, y_tilde, S, bottom_start)
        self.assertGreater(diag["n_nnls_solves"], 0)
        self.assertEqual(diag["n_failures"], 0)
        self.assertGreaterEqual(diag["min_prediction"], 0.0)
        self.assertGreater(diag["nnls_atol"], 0.0)
        self.assertEqual(diag["nnls_maxiter"], 3 * S.shape[1])
        self.assertEqual(diag["unconstrained_feasibility_tolerance"], 0.0)

    def test_nnls_no_worse_than_clipped_projection(self) -> None:
        S, bottom_start = sum_matrix_2level()
        rng = np.random.default_rng(4)
        y_hat = rng.normal(-1.0, 4.0, size=(20, 4, 1))
        y_tilde, _ = reconcile_ae.reconcile_mint_nnls(y_hat, S, bottom_start_idx=bottom_start)

        G = np.linalg.pinv(S.T @ S) @ S.T
        for s in range(y_hat.shape[0]):
            y = y_hat[s, :, 0]
            b_clip = np.clip(G @ y, 0.0, None)
            b_nnls = y_tilde[s, bottom_start:, 0]
            obj_clip = np.sum((y - S @ b_clip) ** 2)
            obj_nnls = np.sum((y - S @ b_nnls) ** 2)
            self.assertLessEqual(obj_nnls, obj_clip + 1e-8)


    def test_parallel_fork_matches_sequential(self) -> None:
        S, bottom_start = sum_matrix_3level()
        y_hat = np.random.default_rng(41).normal(
            -1.0, 4.0, size=(20, S.shape[0], 2))
        sequential, _ = reconcile_ae.reconcile_mint_nnls(
            y_hat, S, bottom_start_idx=bottom_start, nnls_workers=1)

        old_threshold = reconcile_ae.NNLS_PARALLEL_THRESHOLD
        reconcile_ae.NNLS_PARALLEL_THRESHOLD = 1
        try:
            parallel, diag = reconcile_ae.reconcile_mint_nnls(
                y_hat, S, bottom_start_idx=bottom_start, nnls_workers=2)
        finally:
            reconcile_ae.NNLS_PARALLEL_THRESHOLD = old_threshold

        np.testing.assert_allclose(parallel, sequential, rtol=0.0, atol=1e-10)
        if "fork" in reconcile_ae.mp.get_all_start_methods():
            self.assertEqual(diag["nnls_workers_effective"], 2)
            self.assertEqual(diag["nnls_parallel_start_method"], "fork")
        else:
            self.assertEqual(diag["nnls_workers_effective"], 1)

    def test_wls_weight_changes_solution(self) -> None:
        S, bottom_start = sum_matrix_2level()
        rng = np.random.default_rng(5)
        y_hat = rng.normal(2.0, 3.0, size=(6, 4, 2))
        y_true = np.abs(rng.normal(2.0, 3.0, size=(6, 4, 2)))
        y_ols, diag_ols = reconcile_ae.reconcile_mint_nnls(y_hat, S, bottom_start_idx=bottom_start)
        y_wls, diag_wls = reconcile_ae.reconcile_mint_nnls(
            y_hat, S, bottom_start_idx=bottom_start, weight_mode="wls", true_values=y_true
        )
        assert_nonneg_and_coherent(self, y_wls, S, bottom_start)
        self.assertEqual(diag_ols["weight_mode"], "ols")
        self.assertEqual(diag_wls["weight_mode"], "wls")
        self.assertFalse(np.allclose(y_ols, y_wls))

    def test_wls_requires_true_values(self) -> None:
        S, bottom_start = sum_matrix_2level()
        with self.assertRaises(ValueError):
            reconcile_ae.reconcile_mint_nnls(
                np.ones((2, 4, 1)), S, bottom_start_idx=bottom_start, weight_mode="wls"
            )


class UnifiedEntryTest(unittest.TestCase):
    def test_dispatch_and_diagnostics(self) -> None:
        S, bottom_start = sum_matrix_4level()
        rng = np.random.default_rng(6)
        y_hat = rng.normal(1.0, 5.0, size=(4, S.shape[0], 2))
        for method in ["bu", "td_fp", "mint_ols"]:
            y_tilde, diag = reconcile_ae.apply_reconciliation_ae(
                method, y_hat, S, bottom_start_idx=bottom_start
            )
            assert_nonneg_and_coherent(self, y_tilde, S, bottom_start)
            for key in ["method", "solver", "runtime_sec", "min_prediction",
                        "coherence_residual_max_abs", "n_failures", "reconcile_version"]:
                self.assertIn(key, diag, msg=f"{method} missing {key}")
            self.assertEqual(diag["reconcile_version"], reconcile_ae.RECONCILE_AE_VERSION)

    def test_unknown_method_rejected(self) -> None:
        S, bottom_start = sum_matrix_2level()
        with self.assertRaises(ValueError):
            reconcile_ae.apply_reconciliation_ae("ols_projection", np.ones((1, 4, 1)), S)


class StructureValidationTest(unittest.TestCase):
    def test_non_identity_bottom_block_rejected(self) -> None:
        S = np.vstack([np.ones((1, 3)), 2.0 * np.eye(3)])
        with self.assertRaises(ValueError):
            reconcile_ae.reconcile_bu(np.ones((2, 4, 1)), S, bottom_start_idx=1)

    def test_negative_sum_matrix_rejected(self) -> None:
        S = np.vstack([np.ones((1, 3)), np.eye(3)])
        S[0, 0] = -1.0
        with self.assertRaises(ValueError):
            reconcile_ae.reconcile_bu(np.ones((2, 4, 1)), S, bottom_start_idx=1)


if __name__ == "__main__":
    unittest.main()
