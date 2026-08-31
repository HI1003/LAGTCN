from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lagtcn.core.data import LoadDatasetLoader, TARGET_TIMESTAMP_SPLIT_VERSION


class TargetTimestampSplitTest(unittest.TestCase):
    def test_direct24_targets_do_not_cross_raw_boundaries(self):
        loader = LoadDatasetLoader.__new__(LoadDatasetLoader)
        raw_t, n, t_in, horizon = 300, 3, 24, 24
        loader.X = torch.zeros(n, 1, raw_t)
        loader.time_index = pd.date_range("2020-01-01", periods=raw_t, freq="h")
        loader.norm_params = {"train_T": 240}
        loader.edges = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int64)
        loader.edge_weights = np.ones(3, dtype=np.float32)
        loader.target_position_indices = [
            list(range(i + t_in, i + t_in + horizon))
            for i in range(raw_t - t_in - horizon + 1)
        ]
        loader.target_time_indices = [
            [loader.time_index[pos] for pos in positions]
            for positions in loader.target_position_indices
        ]
        loader.target_time_index = [values[0] for values in loader.target_time_indices]
        loader.task_input_length = t_in
        loader.task_output_length = horizon
        loader.task_stride = 1
        features = [np.zeros((n, 1, t_in), dtype=np.float32) for _ in loader.target_position_indices]
        targets = [np.zeros((n, horizon), dtype=np.float32) for _ in loader.target_position_indices]
        signal = StaticGraphTemporalSignal(loader.edges, loader.edge_weights, features, targets)

        train, validation, test, provenance, timestamps = loader.split_dataset_by_target_timestamp(signal)

        self.assertEqual(len(train.features), 193)
        self.assertEqual(len(validation.features), 7)
        self.assertEqual(len(test.features), 7)
        self.assertEqual(provenance["dropped_boundary_origin_count"], 46)
        self.assertEqual(provenance["split_protocol_version"], TARGET_TIMESTAMP_SPLIT_VERSION)
        self.assertEqual(
            provenance["window_assignment_protocol"],
            "global_stride_1_windows_then_target_timestamp_partition",
        )
        self.assertEqual(
            provenance["input_history_boundary_policy"],
            "may_look_back_across_partition_boundary",
        )
        self.assertEqual(
            provenance["target_boundary_policy"],
            "all_targets_must_lie_within_one_partition",
        )
        self.assertEqual(provenance["input_length"], t_in)
        self.assertEqual(provenance["output_length"], horizon)
        self.assertEqual(provenance["stride"], 1)
        self.assertEqual(provenance["segments"]["validation"]["target_cell_count"], 7 * 24)
        self.assertEqual(len(timestamps["test"]), 7)

    def test_formal_168_to_24_boundary_origins_use_observed_history_only(self):
        loader = LoadDatasetLoader.__new__(LoadDatasetLoader)
        raw_t, n, t_in, horizon = 1000, 1, 168, 24
        loader.X = torch.arange(raw_t, dtype=torch.float32).reshape(n, 1, raw_t)
        loader.time_index = pd.date_range("2020-01-01", periods=raw_t, freq="h")
        loader.norm_params = {"train_T": 800}
        loader.edges = np.array([[0], [0]], dtype=np.int64)
        loader.edge_weights = np.ones(1, dtype=np.float32)
        loader._generate_task(t_in, horizon)
        signal = StaticGraphTemporalSignal(
            loader.edges,
            loader.edge_weights,
            loader.features,
            loader.targets,
        )

        train, validation, test, provenance, _ = loader.split_dataset_by_target_timestamp(signal)

        self.assertEqual(len(train.features), 609)
        self.assertEqual(len(validation.features), 77)
        self.assertEqual(len(test.features), 77)
        self.assertEqual(provenance["dropped_boundary_origin_count"], 46)

        np.testing.assert_array_equal(validation.features[0][0, 0], np.arange(632, 800))
        np.testing.assert_array_equal(validation.targets[0][0], np.arange(800, 824))
        np.testing.assert_array_equal(test.targets[0][0], np.arange(900, 924))
        np.testing.assert_array_equal(test.features[0][0, 0], np.arange(732, 900))

    def test_frozen_normalization_boundary_mismatch_fails(self):
        loader = LoadDatasetLoader.__new__(LoadDatasetLoader)
        loader.X = torch.zeros(2, 1, 300)
        loader.norm_params = {"train_T": 239}
        loader.target_position_indices = [[24]]
        loader.target_time_indices = [[pd.Timestamp("2020-01-02")]]
        with self.assertRaisesRegex(ValueError, "train_T"):
            loader.split_dataset_by_target_timestamp(object())


if __name__ == "__main__":
    unittest.main()
