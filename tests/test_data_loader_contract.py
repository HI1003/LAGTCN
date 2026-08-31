from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lagtcn.core.data import LoadDatasetLoader


class DataLoaderContractTest(unittest.TestCase):
    def _write_fixture(self, root: Path) -> None:
        timesteps = 20
        node_order = ["Total", "Bottom"]
        values = np.full((timesteps, 2, 1), 0.5, dtype=np.float32)
        np.save(root / "node_values.npy", values)
        np.save(root / "adj_hierarchy.npy", np.eye(2, dtype=np.float32))
        np.save(
            root / "normalization_params.npy",
            {
                "norm_method": "minmax",
                "min": 0.0,
                "max": 1.0,
                "use_log": False,
                "train_T": 16,
            },
        )
        pd.DataFrame([[1.0], [1.0]]).to_csv(
            root / "sum_matrix.csv", header=False, index=False
        )
        frame = pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-01", periods=timesteps, freq="h"),
                "Total": np.ones(timesteps),
                "Bottom": np.ones(timesteps),
            }
        )
        frame.to_csv(root / "load.csv", index=False)
        (root / "hierarchy_info.json").write_text(
            json.dumps(
                {
                    "num_total_nodes": 2,
                    "num_bottom_nodes": 1,
                    "bottom_start_idx": 1,
                    "num_mid_nodes": 0,
                    "node_order": node_order,
                    "middle_levels": [],
                }
            )
        )

    def test_constructor_accepts_main_entrypoint_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            loader = LoadDatasetLoader(
                str(root),
                input_dim=1,
                adj_file="adj_hierarchy.npy",
                value_file="node_values.npy",
                raw_csv_file="load.csv",
            )

            self.assertEqual(loader.expected_input_dim, 1)
            self.assertEqual(loader.adj_file, "adj_hierarchy.npy")
            self.assertEqual(tuple(loader.X.shape), (2, 1, 20))
            self.assertEqual(loader.node_names, ["Total", "Bottom"])
            self.assertEqual(loader.sum_matrix.shape, (2, 1))

    def test_constructor_rejects_feature_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_fixture(root)
            with self.assertRaisesRegex(ValueError, "Feature dim mismatch"):
                LoadDatasetLoader(
                    str(root),
                    input_dim=2,
                    adj_file="adj_hierarchy.npy",
                    value_file="node_values.npy",
                    raw_csv_file="load.csv",
                )


if __name__ == "__main__":
    unittest.main()
