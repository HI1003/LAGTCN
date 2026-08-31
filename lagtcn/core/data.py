import json
import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


TARGET_TIMESTAMP_SPLIT_VERSION = "target_timestamp_80_10_10_v1"


class LoadDatasetLoader(object):
    """
    Loads:
      - adjacency: adj_hierarchy.npy (default; configurable)
      - node values: configurable filename (log/normalize settings follow normalization_params.npy)
      - normalization_params.npy (supports log/minmax/zscore parameters)
      - sum_matrix.csv: shape [N, nbottom] for bottom-up aggregation.
      - original CSV (to fetch time index + node names):
          configurable filename (default names; adjust if needed)

    Also builds:
      - mid_to_bottom_indices: one list per middle node containing the contributing
        bottom-level global node indices.
      - target_time_index: list of timestamps aligned to each sample window (same as TGLP.py)

    Optional:
      - hierarchy_info.json with precomputed hierarchy mapping for arbitrary datasets.
    """

    def __init__(
        self,
        raw_data_dir: str,
        input_dim: int = 1,
        adj_file: str = "adj_hierarchy.npy",
        value_file: str = "node_values.npy",
        raw_csv_file: str = "Load_GEFCom2012_hourly.csv",
        norm_file: str = "normalization_params.npy",
    ):
        super(LoadDatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self.expected_input_dim = int(input_dim)
        if self.expected_input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        self.adj_file = adj_file
        self.value_file = value_file
        self.raw_csv_file = raw_csv_file
        self.norm_file = norm_file

        norm_path = self._resolve_path(self.norm_file)
        norm_params = np.load(norm_path, allow_pickle=True).item()
        self._init_norm_config(norm_params)

        self.A = None
        self.X = None
        self.edges = None
        self.edge_weights = None
        self.features = None
        self.targets = None
        self.time_index = None
        self.node_names = None
        self.target_time_index = None
        self.target_time_indices = None
        self.target_position_indices = None
        self.hierarchy_info = self._load_hierarchy_info()

        self._read_web_data()

        # 加载求和矩阵
        self.sum_matrix = pd.read_csv(
            os.path.join(self.raw_data_dir, 'sum_matrix.csv'),
            header=None
        ).dropna(axis=1, how="all").to_numpy().astype(np.float32)
        if self.sum_matrix.ndim != 2 or self.sum_matrix.shape[0] != self.X.shape[0]:
            raise ValueError(
                f"sum_matrix shape {self.sum_matrix.shape} is incompatible with "
                f"X nodes={self.X.shape[0]}."
            )
        if not np.isfinite(self.sum_matrix).all():
            raise ValueError("sum_matrix contains NaN or Inf.")
        if np.any(self.sum_matrix < 0):
            raise ValueError("sum_matrix must be nonnegative.")

        # === 基于 sum_matrix / X 推导层次结构信息 ===
        # 总节点数 = X 的节点数
        self.num_total_nodes = int(self.X.shape[0])
        # 底层节点数 = sum_matrix 的列数
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        # 底层起始索引（假定只有 1 个顶层 + 若干中间层）
        self.bottom_start_idx = self.num_total_nodes - self.num_bottom_nodes  # 123-118=5
        # 中间层节点数（不含顶层）
        self.num_mid_nodes = self.bottom_start_idx - 1  # 5-1=4

        self._apply_hierarchy_info()

        # 创建分解矩阵 - 基于历史数据的比例（只用训练期）
        self.decomposition_matrix = self._create_decomposition_matrix()

        # 创建中间层到底层的索引映射
        self.mid_to_bottom_indices = self._create_mid_to_bottom_indices()

    def _resolve_path(self, filename: str | os.PathLike) -> str:
        path = Path(filename)
        if path.is_absolute():
            return str(path)
        return os.path.join(self.raw_data_dir, str(filename))

    def _init_norm_config(self, norm_params: dict) -> None:
        if not isinstance(norm_params, dict):
            raise ValueError("normalization_params.npy must contain a dict.")
        self.norm_params = norm_params

        method = norm_params.get("norm_method") or norm_params.get("method")
        if method is None:
            if "mean" in norm_params and "std" in norm_params:
                method = "zscore"
            else:
                method = "minmax"
        self.norm_method = str(method).lower()

        use_log = norm_params.get("use_log")
        if use_log is None:
            use_log = norm_params.get("mode") == "log" or "global_min" in norm_params
        self.use_log = bool(use_log)

        log_offset = norm_params.get("log_offset", 1.0)
        self.log_offset = float(log_offset) if self.use_log else 0.0

        if self.norm_method == "zscore":
            if "mean" not in norm_params or "std" not in norm_params:
                raise KeyError("normalization_params.npy missing mean/std for zscore normalization.")
            self.norm_mean = float(norm_params["mean"])
            self.norm_std = float(norm_params["std"])
            if not np.isfinite(self.norm_mean) or not np.isfinite(self.norm_std):
                raise ValueError("normalization mean/std must be finite.")
            if self.norm_std == 0:
                self.norm_std = 1.0
            self.norm_min = None
            self.norm_max = None
            self.global_min = self.norm_mean
            self.global_max = self.norm_mean + self.norm_std
        else:
            min_val = norm_params.get("min", norm_params.get("global_min"))
            max_val = norm_params.get("max", norm_params.get("global_max"))
            if min_val is None or max_val is None:
                raise KeyError("normalization_params.npy missing min/max (or global_min/global_max).")
            self.norm_min = float(min_val)
            self.norm_max = float(max_val)
            if (
                not np.isfinite(self.norm_min)
                or not np.isfinite(self.norm_max)
                or self.norm_max <= self.norm_min
            ):
                raise ValueError("normalization min/max must be finite with max > min.")
            self.norm_mean = None
            self.norm_std = None
            self.global_min = self.norm_min
            self.global_max = self.norm_max

    def _create_decomposition_matrix(self):
        """
        创建分解矩阵，用于自上而下的预测分解。

        思路：
        - 使用历史数据中“底层节点 / 顶层节点”的平均比例；
        - 只在训练期时间步上计算，避免使用 test 信息；
        - 输出形状为 [num_bottom_nodes, 1]，每个底层节点一个权重，且和为 1。
        """
        # 根据层次信息得到索引
        bottom_start = self.bottom_start_idx
        num_bottom = self.num_bottom_nodes
        total_node_index = 0  # 顶层节点索引始终为 0

        # 提取底层节点和总节点的时间序列（仍在归一化 + log 空间）
        item_series = self.X[bottom_start:bottom_start + num_bottom, 0, :]  # [num_bottom, T]
        total_series = self.X[total_node_index, 0, :]  # [T]

        # 反归一化和逆对数变换，回到原始尺度
        item_series = self.inverse_log_transform(self.denormalize_data(item_series))
        total_series = self.inverse_log_transform(self.denormalize_data(total_series))

        # === 只用“训练期时间步”来估计比例，避免泄漏 test ===
        T = total_series.shape[0]
        # 与 raw target-timestamp split 的冻结训练边界保持一致。
        train_ratio = 0.8
        train_T = int(self.norm_params.get("train_T") or max(1, int(T * train_ratio)))
        expected_train_t = max(1, int(T * train_ratio))
        if train_T != expected_train_t:
            raise ValueError(
                f"normalization train_T={train_T} differs from 80% boundary={expected_train_t}."
            )

        item_series = item_series[:, :train_T]  # [num_bottom, train_T]
        total_series = total_series[:train_T]  # [train_T]

        # 避免除零
        total_series_safe = np.where(total_series == 0, 1e-6, total_series)

        # 计算每个底层节点在总量中的比例
        proportions = item_series / total_series_safe  # [num_bottom, train_T]

        # 时间平均比例
        mean_proportions = proportions.mean(axis=1)  # [num_bottom]

        # 归一化使其和为 1
        normalized_proportions = mean_proportions / np.maximum(mean_proportions.sum(), 1e-8)

        # 创建分解矩阵 [num_bottom, 1]
        decomposition_matrix = np.zeros((num_bottom, 1), dtype=np.float32)
        for i in range(num_bottom):
            decomposition_matrix[i, 0] = normalized_proportions[i]

        return decomposition_matrix

    def _create_mid_to_bottom_indices(self):
        """创建中间层到底层的索引映射"""
        if getattr(self, "num_mid_nodes", 0) <= 0:
            return []
        if self.hierarchy_info and "mid_to_bottom_indices" in self.hierarchy_info:
            mid_to_bottom = self.hierarchy_info.get("mid_to_bottom_indices")
            if not mid_to_bottom:
                return []
            return [list(map(int, indices)) for indices in mid_to_bottom]
        # 定义层次结构映射
        # 品牌到商品的映射关系
        brand_to_items = {
            1: list(range(5, 47)),  # B1下42个商品
            2: list(range(47, 92)),  # B2下45个商品
            3: list(range(92, 113)),  # B3下21个商品
            4: list(range(113, 123))  # B4下10个商品
        }

        # 转换为中间层到底层的索引映射
        mid_to_bottom_indices = []
        for i in range(1, 5):
            mid_to_bottom_indices.append(brand_to_items[i])

        return mid_to_bottom_indices

    def _load_hierarchy_info(self):
        info_path = os.path.join(self.raw_data_dir, "hierarchy_info.json")
        if not os.path.exists(info_path):
            return None
        try:
            with open(info_path, "r") as f:
                data = json.load(f)
            logging.info("Loaded hierarchy_info.json from %s", info_path)
            return data
        except Exception as exc:
            raise ValueError(
                f"Failed to read hierarchy_info.json at {info_path}: {exc}"
            ) from exc

    def _apply_hierarchy_info(self):
        inferred = {
            "num_total_nodes": int(self.X.shape[0]),
            "num_bottom_nodes": int(self.sum_matrix.shape[1]),
            "bottom_start_idx": int(self.X.shape[0] - self.sum_matrix.shape[1]),
        }
        inferred["num_mid_nodes"] = inferred["bottom_start_idx"] - 1
        if self.hierarchy_info:
            for key, expected in inferred.items():
                declared = self.hierarchy_info.get(key)
                if declared is not None and int(declared) != expected:
                    raise ValueError(
                        f"Hierarchy metadata {key}={declared} differs from inferred {expected}."
                    )
            declared_order = self.hierarchy_info.get("node_order")
            if declared_order is not None:
                declared_order = [str(value) for value in declared_order]
                if declared_order != [str(value) for value in self.node_names]:
                    raise ValueError(
                        "hierarchy_info.node_order differs from the loaded target-node order."
                    )
            middle_levels = self.hierarchy_info.get("middle_levels")
            if middle_levels is not None:
                flattened = [int(index) for level in middle_levels for index in level]
                expected_middle = list(range(1, inferred["bottom_start_idx"]))
                if sorted(flattened) != expected_middle or len(flattened) != len(set(flattened)):
                    raise ValueError(
                        "hierarchy_info.middle_levels must partition every middle-node index exactly once."
                    )

        self.num_total_nodes = inferred["num_total_nodes"]
        self.num_bottom_nodes = inferred["num_bottom_nodes"]
        self.bottom_start_idx = inferred["bottom_start_idx"]
        self.num_mid_nodes = inferred["num_mid_nodes"]
        bottom_block = self.sum_matrix[
            self.bottom_start_idx:self.bottom_start_idx + self.num_bottom_nodes
        ]
        identity = np.eye(self.num_bottom_nodes, dtype=np.float32)
        if bottom_block.shape != identity.shape or not np.allclose(
            bottom_block, identity, rtol=0.0, atol=1e-7
        ):
            raise ValueError(
                "sum_matrix bottom block must be an identity matrix in node_order."
            )


    def _read_web_data(self):
        # 1) 读取邻接矩阵
        self.A = torch.from_numpy(
            np.load(os.path.join(self.raw_data_dir, self.adj_file)).astype(np.float32)
        )

        # 2) 读取节点时间序列（log 空间的 numpy）
        value_path = self._resolve_path(self.value_file)
        X_np = np.load(value_path).astype(np.float32)  # [T, N, F]
        # 转成 [N, F, T]
        self.X = torch.from_numpy(X_np.transpose((1, 2, 0)))  # [N, F, T]
        N, F, T = self.X.shape
        if not bool(torch.isfinite(self.X).all()):
            raise ValueError("Node-value tensor contains NaN or Inf.")
        if tuple(self.A.shape) != (N, N):
            raise ValueError(
                f"Adjacency shape {tuple(self.A.shape)} does not match node count N={N}."
            )
        if not bool(torch.isfinite(self.A).all()):
            raise ValueError("Adjacency contains NaN or Inf.")
        if bool((self.A < 0).any()):
            raise ValueError("Adjacency must be nonnegative.")

        # Ensure data feature dimension matches config input_dim
        if F != self.expected_input_dim:
            raise ValueError(
                f"[Feature dim mismatch] Data feature dim F={F} but expected input_dim={self.expected_input_dim}. "
                f"Check node_values_*.npy or config['input_dim']."
            )

        # 3) 从原始 CSV 读取时间索引和节点名称
        csv_path = self._resolve_path(self.raw_csv_file)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"找不到原始数据 CSV 文件: {csv_path}")

        df = pd.read_csv(csv_path)

        # 假设第一列是日期列：df.columns[0]
        time_col = df.columns[0]
        time_index_raw = df[time_col].values
        if len(time_index_raw) != T:
            raise ValueError(
                f"CSV 中时间长度 {len(time_index_raw)} 与 X 的时间长度 T={T} 不一致，请检查预处理。"
            )
        # 转成 DatetimeIndex
        self.time_index = pd.to_datetime(time_index_raw)
        if (
            self.time_index.hasnans
            or not self.time_index.is_unique
            or not self.time_index.is_monotonic_increasing
        ):
            raise ValueError("Time index must be finite, unique, and monotonically increasing.")

        # Node labels must match hierarchy metadata or the N target columns.
        node_cols_all = list(df.columns[1:])
        node_cols_all_str = [str(c) for c in node_cols_all]
        metadata_node_order = None
        if self.hierarchy_info and isinstance(self.hierarchy_info.get("node_order"), list):
            candidate_order = [str(c) for c in self.hierarchy_info.get("node_order", [])]
            if len(candidate_order) == N and set(candidate_order).issubset(set(node_cols_all_str)):
                metadata_node_order = candidate_order

        if len(node_cols_all) < N:
            raise ValueError(
                f"CSV has {len(node_cols_all)} node columns but the value tensor has N={N}."
            )

        if metadata_node_order is not None:
            self.node_names = metadata_node_order
            if len(node_cols_all) != N:
                extra_cols = [c for c in node_cols_all_str if c not in set(metadata_node_order)]
                preview = extra_cols[:10]
                suffix = "" if len(extra_cols) <= len(preview) else f" ... (+{len(extra_cols) - len(preview)} more)"
                logging.warning(
                    "CSV has %s columns after the time column but X has N=%s nodes. "
                    "Using hierarchy_info.node_order and ignoring %s extra columns for naming: %s%s",
                    len(node_cols_all), N, len(extra_cols), preview, suffix,
                )
        elif len(node_cols_all) == N:
            self.node_names = node_cols_all
        else:
            raise ValueError(
                f"CSV has {len(node_cols_all)} columns but X has N={N}; check the column order."
            )
        node_names_as_text = [str(name) for name in self.node_names]
        if len(node_names_as_text) != N or len(set(node_names_as_text)) != N:
            raise ValueError("Target node names must contain exactly N unique labels.")
        self.node_names = node_names_as_text

        logging.info(
            f"A shape: {self.A.shape}, X shape: {self.X.shape}, "
            f"time_index length: {len(self.time_index)}, num_nodes: {len(self.node_names)}"
        )

    def denormalize_data(self, normalized_data: np.ndarray) -> np.ndarray:
        """反归一化数据"""
        if self.norm_method == "zscore":
            return normalized_data * self.norm_std + self.norm_mean
        return normalized_data * (self.norm_max - self.norm_min) + self.norm_min

    def inverse_log_transform(self, log_data: np.ndarray) -> np.ndarray:
        """逆对数变换函数"""
        if not self.use_log:
            return log_data
        return np.exp(log_data) - self.log_offset

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        self.edges = edge_indices.numpy()
        self.edge_weights = values.numpy()

    def _generate_task(self, num_timesteps_in: int = 7, num_timesteps_out: int = 1):
        """生成预测任务的特征和目标"""
        self.task_input_length = int(num_timesteps_in)
        self.task_output_length = int(num_timesteps_out)
        self.task_stride = 1
        total_timesteps = self.X.shape[2]
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(total_timesteps - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # 每个样本对应的预测时间索引（窗口右端第一个点）
        if self.time_index is None:
            raise RuntimeError("time_index 为空，请检查 _read_web_data 是否正确读取了 CSV。")
        self.target_position_indices = [
            list(range(i + num_timesteps_in, j)) for (i, j) in indices
        ]
        self.target_time_indices = [
            [self.time_index[pos] for pos in positions]
            for positions in self.target_position_indices
        ]
        self.target_time_index = [timestamps[0] for timestamps in self.target_time_indices]

        # 所有节点作为输入特征，形状: (N, F, T_in)
        self.features = [
            self.X[:, :, i:i + num_timesteps_in].numpy()
            for i, _ in indices
        ]
        # 目标：所有节点的第一个特征作为预测目标，形状: (N, T_out)
        self.targets = [
            self.X[:, 0, i + num_timesteps_in: j].numpy()
            for i, j in indices
        ]

    def split_dataset_by_target_timestamp(
        self,
        dataset: StaticGraphTemporalSignal,
        train_ratio: float = 0.8,
        validation_ratio: float = 0.1,
    ):
        """Split origins only when every target timestamp belongs to one segment.

        Boundaries are defined on the original raw time axis. Input windows may
        look back across a boundary, but forecast targets that straddle a
        boundary are excluded instead of leaking into either adjacent segment.
        """
        train_ratio = float(train_ratio)
        validation_ratio = float(validation_ratio)
        if not 0 < train_ratio < 1:
            raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
        if validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
            raise ValueError(
                "validation_ratio must be positive and train_ratio + validation_ratio < 1"
            )
        if self.target_position_indices is None or self.target_time_indices is None:
            raise RuntimeError("Call get_dataset() before target-timestamp splitting.")

        raw_length = int(self.X.shape[2])
        train_end = int(raw_length * train_ratio)
        validation_end = int(raw_length * (train_ratio + validation_ratio))
        if not 0 < train_end < validation_end < raw_length:
            raise ValueError(
                f"Invalid raw boundaries train_end={train_end}, validation_end={validation_end}, "
                f"raw_length={raw_length}"
            )
        frozen_train_t = self.norm_params.get("train_T")
        if frozen_train_t is not None and int(frozen_train_t) != train_end:
            raise ValueError(
                "normalization_params train_T does not match target split boundary: "
                f"train_T={frozen_train_t}, expected={train_end}."
            )
        sample_count = len(self.target_position_indices)
        observed_counts = {
            "dataset.features": len(dataset.features),
            "dataset.targets": len(dataset.targets),
            "target_time_indices": len(self.target_time_indices),
            "target_time_index": len(self.target_time_index),
        }
        if any(count != sample_count for count in observed_counts.values()):
            raise ValueError(
                "Window metadata and dataset snapshot counts differ: "
                f"positions={sample_count}, observed={observed_counts}."
            )


        segment_indices = {"train": [], "validation": [], "test": []}
        dropped = []
        for sample_idx, positions in enumerate(self.target_position_indices):
            if not positions:
                raise ValueError(f"Sample {sample_idx} has no target positions.")
            expected_output_length = int(self.task_output_length)
            expected_positions = list(
                range(int(positions[0]), int(positions[0]) + expected_output_length)
            )
            if [int(value) for value in positions] != expected_positions:
                raise ValueError(
                    f"Sample {sample_idx} target positions are not one contiguous "
                    f"length-{expected_output_length} block."
                )
            first, last = int(positions[0]), int(positions[-1])
            if 0 <= first and last < train_end:
                segment_indices["train"].append(sample_idx)
            elif train_end <= first and last < validation_end:
                segment_indices["validation"].append(sample_idx)
            elif validation_end <= first and last < raw_length:
                segment_indices["test"].append(sample_idx)
            else:
                dropped.append(sample_idx)

        for name, indices in segment_indices.items():
            if not indices:
                raise ValueError(f"Target-timestamp split produced an empty {name} segment.")

        def subset(indices):
            return StaticGraphTemporalSignal(
                self.edges,
                self.edge_weights,
                [dataset.features[i] for i in indices],
                [dataset.targets[i] for i in indices],
            )

        signals = tuple(subset(segment_indices[name]) for name in ("train", "validation", "test"))

        def segment_metadata(name):
            indices = segment_indices[name]
            target_lists = [self.target_time_indices[i] for i in indices]
            return {
                "origin_count": int(len(indices)),
                "target_cell_count": int(sum(len(v) for v in target_lists)),
                "first_target_timestamp": str(target_lists[0][0]),
                "last_target_timestamp": str(target_lists[-1][-1]),
                "first_sample_index": int(indices[0]),
                "last_sample_index": int(indices[-1]),
            }

        provenance = {
            "split_protocol_version": TARGET_TIMESTAMP_SPLIT_VERSION,
            "window_assignment_protocol": (
                "global_stride_1_windows_then_target_timestamp_partition"
            ),
            "input_history_boundary_policy": (
                "may_look_back_across_partition_boundary"
            ),
            "target_boundary_policy": "all_targets_must_lie_within_one_partition",
            "raw_time_length": raw_length,
            "train_end_position_exclusive": train_end,
            "validation_end_position_exclusive": validation_end,
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": 1.0 - train_ratio - validation_ratio,
            "input_length": int(self.task_input_length),
            "output_length": int(self.task_output_length),
            "stride": int(self.task_stride),
            "dropped_boundary_origin_count": int(len(dropped)),
            "dropped_boundary_sample_indices": [int(i) for i in dropped],
            "segments": {
                name: segment_metadata(name)
                for name in ("train", "validation", "test")
            },
        }
        timestamp_indices = {
            name: [self.target_time_index[i] for i in indices]
            for name, indices in segment_indices.items()
        }
        return (*signals, provenance, timestamp_indices)

    def get_dataset(self, num_timesteps_in: int = 7, num_timesteps_out: int = 1) -> StaticGraphTemporalSignal:
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        return StaticGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
