from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from lagtcn.core import scaled_error as mase


class NaiveScaleTest(unittest.TestCase):
    def test_known_values(self) -> None:
        train = np.array([[0.0, 1.0], [2.0, 1.0], [6.0, 1.0]])  # diffs: [2,4] and [0,0]
        scale = mase.compute_naive_scale(train, seasonal_period=1)
        np.testing.assert_allclose(scale, [3.0, 0.0])

    def test_default_is_24_hour_seasonal_scale(self) -> None:
        time = np.arange(49, dtype=float)
        train = np.column_stack([time, np.ones_like(time)])
        scale = mase.compute_naive_scale(train)
        np.testing.assert_allclose(scale, [24.0, 0.0])

    def test_rejects_short_or_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            mase.compute_naive_scale(np.ones((1, 3)))
        with self.assertRaises(ValueError):
            mase.compute_naive_scale(np.ones((24, 3)))
        with self.assertRaises(ValueError):
            mase.compute_naive_scale(np.ones((30, 3)), seasonal_period=0)
        bad = np.ones((5, 2))
        bad[2, 1] = np.nan
        with self.assertRaises(ValueError):
            mase.compute_naive_scale(bad, seasonal_period=1)

    def test_metadata_counts_degenerate_nodes(self) -> None:
        meta = mase.naive_scale_metadata(
            np.array([3.0, 0.0, 5.0]), train_length=100, node_names=["a", "b", "c"]
        )
        self.assertEqual(meta["num_degenerate_nodes"], 1)
        self.assertEqual(meta["degenerate_node_indices"], [1])
        self.assertEqual(meta["degenerate_node_names"], ["b"])
        self.assertEqual(meta["scale_min"], 3.0)
        self.assertEqual(meta["train_length"], 100)
        self.assertEqual(meta["mase_version"], mase.MASE_VERSION)
        self.assertEqual(meta["mase_label"], "sMASE-24")
        self.assertEqual(meta["seasonal_period"], 24)
        self.assertEqual(meta["scale_reference"], "training_period_seasonal_naive_mae")


class MasePerNodeTest(unittest.TestCase):
    def test_hand_computed_two_nodes(self) -> None:
        scale = np.array([2.0, 4.0])
        # 2 samples, 2 nodes, 2 horizons; abs errors node0: [1,1,1,1] node1: [2,2,2,2]
        y_true = np.zeros((2, 2, 2))
        y_pred = np.stack([np.full((2, 2), [1.0, 2.0]) for _ in range(2)], axis=2)
        per_node = mase.compute_mase_per_node(y_true, y_pred, scale)
        np.testing.assert_allclose(per_node, [0.5, 0.5])

    def test_degenerate_scale_is_nan_and_excluded(self) -> None:
        scale = np.array([1.0, 0.0])
        y_true = np.zeros((3, 2, 1))
        y_pred = np.ones((3, 2, 1))
        per_node = mase.compute_mase_per_node(y_true, y_pred, scale)
        self.assertTrue(np.isnan(per_node[1]))
        summary = mase.macro_average_mase(per_node, [0, 1])
        self.assertEqual(summary["n_excluded"], 1)
        self.assertAlmostEqual(summary["mase"], 1.0)

    def test_cell_mask_restricts_numerator(self) -> None:
        scale = np.array([1.0])
        y_true = np.zeros((3, 1, 2))
        y_pred = np.zeros((3, 1, 2))
        y_pred[0, 0, :] = 5.0  # error only in the first origin
        mask = np.ones((3, 2), dtype=bool)
        mask[0, :] = False
        per_node = mase.compute_mase_per_node(y_true, y_pred, scale, cell_mask=mask)
        np.testing.assert_allclose(per_node, [0.0])
        per_node_full = mase.compute_mase_per_node(y_true, y_pred, scale)
        self.assertGreater(per_node_full[0], 0.0)


class LevelGroupTest(unittest.TestCase):
    def test_two_level(self) -> None:
        groups = dict(mase.build_level_groups(3, bottom_start_idx=1, num_bottom_nodes=2))
        self.assertEqual(groups["All"], [0, 1, 2])
        self.assertEqual(groups["top_level"], [0])
        self.assertEqual(groups["bottom_level"], [1, 2])
        self.assertNotIn("middle1_level", groups)

    def test_three_level(self) -> None:
        groups = dict(mase.build_level_groups(
            6, bottom_start_idx=3, num_bottom_nodes=3, middle_levels=[[1, 2]]
        ))
        self.assertEqual(groups["middle1_level"], [1, 2])

    def test_four_level(self) -> None:
        groups = dict(mase.build_level_groups(
            7, bottom_start_idx=4, num_bottom_nodes=3, middle_levels=[[1], [2, 3]]
        ))
        self.assertEqual(groups["middle1_level"], [1])
        self.assertEqual(groups["middle2_level"], [2, 3])

    def test_incomplete_middle_coverage_fails(self) -> None:
        with self.assertRaises(ValueError):
            mase.build_level_groups(7, bottom_start_idx=4, num_bottom_nodes=3, middle_levels=[[1]])


class MaseReportTest(unittest.TestCase):
    def test_report_rows_and_horizon_split(self) -> None:
        scale = np.array([2.0, 1.0, 1.0])
        groups = mase.build_level_groups(3, bottom_start_idx=1, num_bottom_nodes=2)
        y_true = np.zeros((4, 3, 2))
        y_pred = np.zeros((4, 3, 2))
        y_pred[:, 0, 0] = 1.0  # top node, horizon 1 only: per-node MASE h1 = 1/2
        rows = mase.compute_mase_report(y_true, y_pred, scale, groups)
        by_key = {(r["Level"], r["Horizon"]): r["MASE"] for r in rows}
        self.assertAlmostEqual(by_key[("top_level", "h1")], 0.5)
        self.assertAlmostEqual(by_key[("top_level", "h2")], 0.0)
        self.assertAlmostEqual(by_key[("top_level", "all")], 0.25)
        self.assertAlmostEqual(by_key[("bottom_level", "all")], 0.0)

    def test_vectorized_masked_report_matches_nodewise_reference(self) -> None:
        rng = np.random.default_rng(19)
        y_true = rng.normal(size=(7, 3, 4))
        y_pred = y_true + rng.normal(size=(7, 3, 4))
        scale = np.array([0.8, 1.7, 0.0])
        groups = mase.build_level_groups(
            3, bottom_start_idx=1, num_bottom_nodes=2
        )
        mask = np.ones((7, 4), dtype=bool)
        mask[:3, 0] = False
        mask[:1, 2] = False
        rows = mase.compute_mase_report(
            y_true, y_pred, scale, groups, cell_mask=mask
        )
        by_key = {(row["Level"], row["Horizon"]): row for row in rows}
        slices = [("all", slice(None))] + [
            (f"h{h + 1}", slice(h, h + 1)) for h in range(4)
        ]
        for label, h_slice in slices:
            reference = mase.compute_mase_per_node(
                y_true[:, :, h_slice],
                y_pred[:, :, h_slice],
                scale,
                cell_mask=mask[:, h_slice],
            )
            for level_name, indices in groups:
                expected = mase.macro_average_mase(reference, indices)
                self.assertAlmostEqual(
                    by_key[(level_name, label)]["MASE"], expected["mase"]
                )
                self.assertEqual(
                    by_key[(level_name, label)]["n_excluded"],
                    expected["n_excluded"],
                )


class StrideAssertionTest(unittest.TestCase):
    def test_hourly_stride_passes(self) -> None:
        idx = np.array(["2017-01-01 00:00", "2017-01-01 01:00", "2017-01-01 02:00"], dtype="datetime64[m]")
        mase.assert_unit_stride(idx)

    def test_gap_fails(self) -> None:
        idx = np.array(["2017-01-01 00:00", "2017-01-01 01:00", "2017-01-01 03:00"], dtype="datetime64[m]")
        with self.assertRaises(AssertionError):
            mase.assert_unit_stride(idx)

    def test_equally_spaced_two_hour_stride_fails(self) -> None:
        idx = np.array(["2017-01-01 00:00", "2017-01-01 02:00", "2017-01-01 04:00"], dtype="datetime64[m]")
        with self.assertRaises(AssertionError):
            mase.assert_unit_stride(idx)
        mase.assert_unit_stride(idx, expected_step="2h")

    def test_integer_index(self) -> None:
        mase.assert_unit_stride([5, 6, 7])
        with self.assertRaises(AssertionError):
            mase.assert_unit_stride([5, 7, 8])

    def test_equally_spaced_integer_stride_two_fails(self) -> None:
        with self.assertRaises(AssertionError):
            mase.assert_unit_stride([0, 2, 4])


if __name__ == "__main__":
    unittest.main()
