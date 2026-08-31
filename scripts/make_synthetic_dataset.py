#!/usr/bin/env python3
"""Create a deterministic coherent hierarchy for a local software smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET = "GEFCom2012_2level"
RAW_CSV = "Load_GEFCom2012_hourly.csv"


def build_dataset(output_root: Path, timesteps: int, seed: int) -> Path:
    if timesteps < 120:
        raise ValueError("--timesteps must be at least 120 for non-empty 80/10/10 splits.")

    rng = np.random.default_rng(seed)
    time = np.arange(timesteps, dtype=np.float64)
    daily = 1.0 + 0.18 * np.sin(2.0 * np.pi * time / 24.0)
    weekly = 1.0 + 0.07 * np.sin(2.0 * np.pi * time / (24.0 * 7.0))

    bottom_count = 20
    bottoms = []
    for index in range(bottom_count):
        scale = 8.0 + 0.65 * index
        phase = 2.0 * np.pi * index / bottom_count
        local = 1.0 + 0.04 * np.sin(2.0 * np.pi * time / 24.0 + phase)
        noise = rng.normal(loc=0.0, scale=0.12, size=timesteps)
        bottoms.append(np.maximum(scale * daily * weekly * local + noise, 0.1))
    bottom_values = np.stack(bottoms, axis=1)
    total = bottom_values.sum(axis=1, keepdims=True)
    values = np.concatenate([total, bottom_values], axis=1).astype(np.float32)

    train_t = int(timesteps * 0.8)
    train_values = values[:train_t]
    mean = float(train_values.mean())
    std = float(train_values.std()) or 1.0
    normalized = ((values - mean) / std)[..., None].astype(np.float32)

    node_order = ["Total"] + [f"Zone_{index:02d}" for index in range(1, 21)]
    sum_matrix = np.zeros((21, 20), dtype=np.float32)
    sum_matrix[0, :] = 1.0
    sum_matrix[1:, :] = np.eye(20, dtype=np.float32)

    adjacency = np.zeros((21, 21), dtype=np.float32)
    adjacency[0, 1:] = 1.0
    adjacency[1:, 0] = 1.0
    expanded_adjacency = adjacency.copy()
    expanded_adjacency[1:, 1:] = 1.0 - np.eye(20, dtype=np.float32)

    dataset_dir = output_root / DATASET
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "node_values.npy", normalized)
    np.save(dataset_dir / "adj_hierarchy.npy", adjacency)
    np.save(dataset_dir / "adj_HGNN.npy", expanded_adjacency)
    np.save(
        dataset_dir / "normalization_params.npy",
        {
            "norm_method": "zscore",
            "use_log": False,
            "mean": mean,
            "std": std,
            "train_T": train_t,
            "synthetic": True,
            "seed": seed,
        },
    )
    pd.DataFrame(sum_matrix).to_csv(dataset_dir / "sum_matrix.csv", header=False, index=False)

    timestamps = pd.date_range("2020-01-01", periods=timesteps, freq="h")
    raw_frame = pd.DataFrame(values, columns=node_order)
    raw_frame.insert(0, "timestamp", timestamps)
    raw_frame.to_csv(dataset_dir / RAW_CSV, index=False)

    hierarchy_info = {
        "num_total_nodes": 21,
        "num_bottom_nodes": 20,
        "bottom_start_idx": 1,
        "num_mid_nodes": 0,
        "node_order": node_order,
        "middle_levels": [],
        "mid_to_bottom_indices": [],
        "synthetic": True,
    }
    (dataset_dir / "hierarchy_info.json").write_text(
        json.dumps(hierarchy_info, indent=2) + "\n", encoding="utf-8"
    )
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("Data"))
    parser.add_argument("--timesteps", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset_dir = build_dataset(args.output_root, args.timesteps, args.seed)
    print(f"Created synthetic smoke-test dataset at {dataset_dir}")


if __name__ == "__main__":
    main()

