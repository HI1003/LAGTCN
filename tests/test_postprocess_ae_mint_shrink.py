from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "code", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from postprocess_ae_mint_shrink import (
    estimate_horizon_shrink_covariances,
    reconcile_horizonwise_mint_shrink,
    strict_validation_slice,
)


class MintShrinkTest(unittest.TestCase):
    def test_validation_slice_keeps_all_target_timestamp_safe_origins(self):
        n_val = 100
        H = 24
        selections = [strict_validation_slice(n_val, H, h) for h in range(H)]
        assert {(selection.start, selection.stop) for selection in selections} == {
            (0, n_val)
        }


    def test_horizon_shrink_covariances_are_positive_definite_and_use_no_edges(self):
        rng = np.random.default_rng(7)
        n_val, n_nodes, H = 80, 5, 6
        truth = rng.normal(size=(n_val, n_nodes, H))
        shared = rng.normal(scale=0.3, size=(n_val, 1, H))
        prediction = truth + shared + rng.normal(
            scale=0.2, size=(n_val, n_nodes, H)
        )

        covariances, diagnostics = estimate_horizon_shrink_covariances(
            prediction, truth
        )

        assert covariances.shape == (H, n_nodes, n_nodes)
        assert len(diagnostics) == H
        assert {row["n_residual_vectors"] for row in diagnostics} == {
            n_val
        }
        for covariance in covariances:
            np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
            assert np.linalg.eigvalsh(covariance).min() > 0.0


    def test_horizonwise_mint_shrink_is_nonnegative_and_coherent(self):
        # total + two bottoms
        S = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        base = np.array(
            [
                [[8.0, 9.0], [7.0, -2.0], [4.0, 8.0]],
                [[5.0, 3.0], [4.0, 2.0], [4.0, 2.0]],
            ]
        )
        covariances = np.stack(
            [
                np.diag([4.0, 1.0, 1.0]),
                np.array(
                    [
                        [3.0, 0.2, 0.1],
                        [0.2, 1.0, 0.0],
                        [0.1, 0.0, 1.5],
                    ]
                ),
            ]
        )

        reconciled, diagnostic = reconcile_horizonwise_mint_shrink(
            base,
            S,
            covariances,
            bottom_start_idx=1,
            nnls_workers=1,
        )

        assert reconciled.shape == base.shape
        assert reconciled.min() >= 0.0
        expected = np.einsum("nb,sbh->snh", S, reconciled[:, 1:, :])
        np.testing.assert_allclose(reconciled, expected, atol=1e-10)
        assert diagnostic["n_failures"] == 0
        assert diagnostic["coherence_residual_max_abs"] <= 1e-10
