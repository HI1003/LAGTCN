"""Data loading and leakage-safe window construction for LAGTCN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


RAW_CSV_BY_DATASET = {
    "GEFCom2012_2level": "Load_GEFCom2012_hourly.csv",
    "GEFCom2017QualifyingMatch_3level": "GEFCom2017QualifyingMatchDemand.csv",
    "GEFCom2017FinalMatch_4level": "load_final_filled.csv",
}
TARGET_TIMESTAMP_SPLIT_VERSION = "target_timestamp_80_10_10_v1"


class WindowDataset(Dataset):
    """A lazy set of forecasting windows backed by one time-series array."""

    def __init__(
        self,
        values: np.ndarray,
        starts: list[int],
        input_length: int,
        output_length: int,
        time_index: pd.DatetimeIndex,
    ) -> None:
        self.values = values
        self.starts = [int(start) for start in starts]
        self.input_length = int(input_length)
        self.output_length = int(output_length)
        self.time_index = time_index

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        split = start + self.input_length
        stop = split + self.output_length
        features = self.values[start:split].transpose(1, 2, 0).copy()
        targets = self.values[split:stop, :, 0].T.copy()
        return torch.from_numpy(features), torch.from_numpy(targets)

    @property
    def target_timestamps(self) -> list[list[str]]:
        return [
            [
                str(timestamp)
                for timestamp in self.time_index[
                    start + self.input_length : start + self.input_length + self.output_length
                ]
            ]
            for start in self.starts
        ]


class HierarchicalLoadDataset:
    """Load one preprocessed GEFCom hierarchy.

    Required files are ``node_values.npy``, ``normalization_params.npy``,
    ``sum_matrix.csv``, ``hierarchy_info.json``, the selected adjacency file,
    and the dataset's timestamped CSV.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        value_file: str = "node_values.npy",
        adjacency_file: str = "adj_hierarchy.npy",
        raw_csv_file: str | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        if not self.dataset_dir.is_dir():
            raise FileNotFoundError(f"dataset directory not found: {self.dataset_dir}")

        self.values = np.load(self.dataset_dir / value_file).astype(np.float32)
        if self.values.ndim != 3:
            raise ValueError(
                f"node values must have shape [time,nodes,features], got {self.values.shape}"
            )
        if not np.isfinite(self.values).all():
            raise ValueError("node values contain NaN or Inf")
        self.time_length, self.num_nodes, self.input_dim = self.values.shape

        self.adjacency = np.load(self.dataset_dir / adjacency_file).astype(np.float32)
        if self.adjacency.shape != (self.num_nodes, self.num_nodes):
            raise ValueError("adjacency shape does not match node count")
        if not np.isfinite(self.adjacency).all() or np.any(self.adjacency < 0):
            raise ValueError("adjacency must be finite and nonnegative")

        self.sum_matrix = (
            pd.read_csv(self.dataset_dir / "sum_matrix.csv", header=None)
            .dropna(axis=1, how="all")
            .to_numpy(dtype=np.float32)
        )
        if self.sum_matrix.ndim != 2 or self.sum_matrix.shape[0] != self.num_nodes:
            raise ValueError("sum_matrix row count does not match node count")
        if not np.isfinite(self.sum_matrix).all() or np.any(self.sum_matrix < 0):
            raise ValueError("sum_matrix must be finite and nonnegative")
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.num_nodes - self.num_bottom_nodes
        bottom_block = self.sum_matrix[self.bottom_start_idx :]
        if not np.allclose(
            bottom_block,
            np.eye(self.num_bottom_nodes, dtype=np.float32),
            rtol=0,
            atol=1e-7,
        ):
            raise ValueError("the bottom block of sum_matrix must be an identity matrix")

        hierarchy_path = self.dataset_dir / "hierarchy_info.json"
        self.hierarchy_info = (
            json.loads(hierarchy_path.read_text(encoding="utf-8"))
            if hierarchy_path.exists()
            else {}
        )
        self.middle_levels = self.hierarchy_info.get("middle_levels")

        params = np.load(
            self.dataset_dir / "normalization_params.npy", allow_pickle=True
        ).item()
        if not isinstance(params, dict):
            raise ValueError("normalization_params.npy must contain a dictionary")
        self.norm_params = params
        self._configure_normalization(params)

        dataset_name = self.dataset_dir.name
        csv_name = raw_csv_file or RAW_CSV_BY_DATASET.get(dataset_name)
        if not csv_name:
            raise ValueError(
                f"unknown dataset {dataset_name!r}; pass raw_csv_file explicitly"
            )
        frame = pd.read_csv(self.dataset_dir / csv_name)
        if len(frame) != self.time_length:
            raise ValueError("timestamp CSV length does not match node_values.npy")
        self.time_index = pd.to_datetime(frame.iloc[:, 0])
        if (
            self.time_index.hasnans
            or not self.time_index.is_unique
            or not self.time_index.is_monotonic_increasing
        ):
            raise ValueError("timestamps must be finite, unique, and increasing")

        declared_order = self.hierarchy_info.get("node_order")
        if declared_order is not None:
            self.node_names = [str(name) for name in declared_order]
        else:
            self.node_names = [str(name) for name in frame.columns[1 : self.num_nodes + 1]]
        if len(self.node_names) != self.num_nodes or len(set(self.node_names)) != self.num_nodes:
            raise ValueError("node order must contain one unique name per node")

    def _configure_normalization(self, params: dict) -> None:
        method = params.get("norm_method") or params.get("method")
        if method is None:
            method = "zscore" if "mean" in params and "std" in params else "minmax"
        self.norm_method = str(method).lower()
        self.use_log = bool(
            params.get(
                "use_log",
                params.get("mode") == "log" or "global_min" in params,
            )
        )
        self.log_offset = float(params.get("log_offset", 1.0)) if self.use_log else 0.0
        if self.norm_method == "zscore":
            self.norm_mean = float(params["mean"])
            self.norm_std = float(params["std"]) or 1.0
            self.norm_min = None
            self.norm_max = None
            self.global_min = self.norm_mean
            self.global_max = self.norm_mean + self.norm_std
        else:
            minimum = params.get("min", params.get("global_min"))
            maximum = params.get("max", params.get("global_max"))
            if minimum is None or maximum is None:
                raise KeyError("min-max normalization requires min/max values")
            self.norm_min = float(minimum)
            self.norm_max = float(maximum)
            if self.norm_max <= self.norm_min:
                raise ValueError("normalization maximum must exceed minimum")
            self.norm_mean = None
            self.norm_std = None
            self.global_min = self.norm_min
            self.global_max = self.norm_max

    def to_original_scale(self, normalized: np.ndarray) -> np.ndarray:
        normalized = np.asarray(normalized)
        if self.norm_method == "zscore":
            transformed = normalized * self.norm_std + self.norm_mean
        else:
            transformed = normalized * (self.norm_max - self.norm_min) + self.norm_min
        return np.exp(transformed) - self.log_offset if self.use_log else transformed

    @property
    def training_end(self) -> int:
        expected = int(self.time_length * 0.8)
        frozen = self.norm_params.get("train_T")
        if frozen is not None and int(frozen) != expected:
            raise ValueError(
                f"normalization train_T={frozen} differs from 80% boundary={expected}"
            )
        return expected

    def training_target_values(self, *, original_scale: bool = False) -> np.ndarray:
        values = self.values[: self.training_end, :, 0]
        return self.to_original_scale(values) if original_scale else values

    def make_splits(
        self,
        input_length: int = 168,
        output_length: int = 24,
    ) -> tuple[WindowDataset, WindowDataset, WindowDataset, dict]:
        """Build global stride-1 windows and split by all target timestamps."""
        input_length = int(input_length)
        output_length = int(output_length)
        if input_length < 1 or output_length < 1:
            raise ValueError("input_length and output_length must be positive")
        last_start = self.time_length - input_length - output_length
        if last_start < 0:
            raise ValueError("time series is shorter than one complete window")

        train_end = self.training_end
        validation_end = int(self.time_length * 0.9)
        starts = {"train": [], "validation": [], "test": []}
        dropped = []
        for start in range(last_start + 1):
            first_target = start + input_length
            last_target = first_target + output_length - 1
            if last_target < train_end:
                starts["train"].append(start)
            elif train_end <= first_target and last_target < validation_end:
                starts["validation"].append(start)
            elif validation_end <= first_target and last_target < self.time_length:
                starts["test"].append(start)
            else:
                dropped.append(start)
        if any(not indices for indices in starts.values()):
            raise ValueError("the requested window lengths produce an empty split")

        datasets = {
            name: WindowDataset(
                self.values,
                indices,
                input_length,
                output_length,
                self.time_index,
            )
            for name, indices in starts.items()
        }
        provenance = {
            "version": TARGET_TIMESTAMP_SPLIT_VERSION,
            "input_length": input_length,
            "output_length": output_length,
            "stride": 1,
            "train_end_exclusive": train_end,
            "validation_end_exclusive": validation_end,
            "origins": {name: len(indices) for name, indices in starts.items()},
            "dropped_boundary_origins": len(dropped),
        }
        return (
            datasets["train"],
            datasets["validation"],
            datasets["test"],
            provenance,
        )
