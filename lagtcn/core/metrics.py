import logging
from lagtcn.core import scaled_error as ae_mase
import numpy as np
import pandas as pd
from lagtcn.core.protocol import is_formal_ae_stage
from lagtcn.core.naming import artifact_filename
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_mase(y_true, y_pred, num_timesteps_in: int = 7, epsilon: float = 1e-8):
    """
    计算 MASE（Mean Absolute Scaled Error）

    定义（按你的描述）：
    - 测试集用 num_timesteps_in 个历史值做一步预测，因此从第 num_timesteps_in+1 个位置开始计入指标；
    - 分子：从索引 num_timesteps_in 开始，模型预测的 MAE；
    - 分母：从索引 num_timesteps_in 开始，Naive 预测（y_{t-1}）的 MAE；
    - MASE = 分子 / 分母。

    参数
    ----
    y_true : np.ndarray
        真实值，形状可以是 [T], [T, N] 或 [T, N, 1]
    y_pred : np.ndarray
        预测值，形状需与 y_true 一致
    num_timesteps_in : int
        所用历史步长，从第 num_timesteps_in 个索引开始计算
    epsilon : float
        防止分母为 0 的小常数

    返回
    ----
    float : MASE 值
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    # 如果是 [T, N, 1]，先去掉最后一维
    if y_true.ndim == 3 and y_true.shape[-1] == 1:
        y_true = y_true[..., 0]
        y_pred = y_pred[..., 0]

    # 至少要有 num_timesteps_in+1 个时间步，才能计算 y_t 和 y_{t-1}
    T = y_true.shape[0]
    if T <= num_timesteps_in:
        return np.nan

    # 展平除时间维以外的维度，统一处理
    y_true_flat = y_true.reshape(T, -1)   # [T, K]
    y_pred_flat = y_pred.reshape(T, -1)   # [T, K]

    # 分子：模型误差，从 t = num_timesteps_in 开始
    model_errors = np.abs(
        y_true_flat[num_timesteps_in:, :] - y_pred_flat[num_timesteps_in:, :]
    )  # [T - num_timesteps_in, K]

    # 分母：Naive 误差，从 t = num_timesteps_in 开始，预测为前一个真实值
    naive_errors = np.abs(
        y_true_flat[num_timesteps_in:, :] - y_true_flat[num_timesteps_in - 1:-1, :]
    )  # [T - num_timesteps_in, K]

    numerator = model_errors.mean()
    denominator = naive_errors.mean() + epsilon

    return float(numerator / denominator)


def compute_coherency_violation(
    predictions: np.ndarray,
    sum_matrix: np.ndarray,
    bottom_start_idx: int | None = None,
    epsilon: float = 1e-8,
) -> dict:
    """
    计算层级一致性违约度（Coherency Violation）。

    给定预测值 y_hat（全节点）和求和矩阵 S：
      y_hat_coherent = S @ y_hat_bottom
    使用 residual = y_hat - y_hat_coherent 评估一致性违约。
    """
    preds = np.asarray(predictions)
    if preds.ndim == 2:
        preds = preds[..., None]
    if preds.ndim != 3:
        raise ValueError(f"predictions must be 2D/3D, got shape {preds.shape}")

    S = np.asarray(sum_matrix, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"sum_matrix must be 2D, got shape {S.shape}")

    num_nodes = preds.shape[1]
    num_bottom = S.shape[1]
    if S.shape[0] != num_nodes:
        raise ValueError(
            f"sum_matrix rows ({S.shape[0]}) must match prediction nodes ({num_nodes})."
        )

    if bottom_start_idx is None:
        bottom_start_idx = num_nodes - num_bottom
    bottom_start_idx = int(bottom_start_idx)
    bottom_end_idx = bottom_start_idx + num_bottom
    if bottom_start_idx < 0 or bottom_end_idx > num_nodes:
        raise ValueError(
            f"Invalid bottom index range [{bottom_start_idx}, {bottom_end_idx}) for num_nodes={num_nodes}."
        )

    bottom_preds = preds[:, bottom_start_idx:bottom_end_idx, :]  # [S, B, H]
    coherent_preds = np.einsum("nb,sbh->snh", S, bottom_preds)   # [S, N, H]
    residual = preds - coherent_preds

    violation_mae = float(np.mean(np.abs(residual)))
    violation_rmse = float(np.sqrt(np.mean(residual ** 2)))
    scale = float(np.mean(np.abs(preds)))
    violation_nmae = float(violation_mae / max(scale, epsilon) * 100.0)

    return {
        "coherency_mae": violation_mae,
        "coherency_rmse": violation_rmse,
        "coherency_nmae_pct": violation_nmae,
    }


def _build_level_indices(config: dict, num_nodes: int):
    bottom_start = config.get('bottom_start_idx')
    num_bottom = config.get('num_bottom_nodes')
    if bottom_start is None:
        if num_bottom is not None:
            bottom_start = num_nodes - int(num_bottom)
        else:
            bottom_start = 1
    bottom_start = int(bottom_start)
    if num_bottom is None:
        num_bottom = max(0, num_nodes - bottom_start)
    num_bottom = int(num_bottom)

    middle_levels = []
    middle_level_indices = config.get("middle_levels")
    if middle_level_indices:
        for level in middle_level_indices:
            if level:
                middle_levels.append([int(i) for i in level])
    else:
        middle_level_sizes = config.get("middle_level_sizes")
        if middle_level_sizes:
            start = 1
            for size in middle_level_sizes:
                size = int(size)
                if size <= 0:
                    continue
                middle_levels.append(list(range(start, start + size)))
                start += size

    if not middle_levels:
        num_mid = int(config.get('num_mid_nodes', bottom_start - 1))
        if num_mid > 0:
            middle_levels.append(list(range(1, 1 + num_mid)))

    top_level = [0] if num_nodes > 0 else []
    bottom_level = list(range(bottom_start, bottom_start + num_bottom)) if num_bottom > 0 else []
    return top_level, middle_levels, bottom_level


def calculate_level_metrics(predictions: np.ndarray, true_values: np.ndarray, config: dict):
    """Compute All/level metrics and save the canonical level CSV.

    MAE, RMSE and WAPE pool cells within a level. For formal Applied Energy
    runs, sMASE uses the frozen training-period 24-hour seasonal-naive scale,
    computes a ratio per node, then macro-averages those node ratios. Legacy
    callers without ``_mase_scale`` retain their historical diagnostic MASE.
    """
    predictions = np.asarray(predictions)
    true_values = np.asarray(true_values)
    if predictions.shape != true_values.shape:
        raise ValueError(
            f"predictions shape {predictions.shape} != true_values shape {true_values.shape}"
        )
    if predictions.ndim == 2:
        predictions = predictions[..., None]
        true_values = true_values[..., None]

    num_nodes = predictions.shape[1]
    num_horizons = predictions.shape[-1]
    num_timesteps_in = int(config.get("num_timesteps_in", 7))
    frozen_scale = config.get("_mase_scale")
    if frozen_scale is None and is_formal_ae_stage(config.get("experiment_stage")):
        raise ValueError(
            "Formal Applied Energy level metrics require the frozen training-period "
            "lag-24 sMASE scale; refusing legacy MASE fallback."
        )
    if frozen_scale is not None:
        frozen_scale = np.asarray(frozen_scale, dtype=np.float64)
        if frozen_scale.shape != (num_nodes,):
            raise ValueError(
                f"_mase_scale shape {frozen_scale.shape} != ({num_nodes},)"
            )

    top_level, middle_levels, bottom_level = _build_level_indices(config, num_nodes)
    level_groups = [("All", list(range(num_nodes))), ("top_level", top_level)]
    for i, indices in enumerate(middle_levels, start=1):
        level_groups.append((f"middle{i}_level", indices))
    level_groups.append(("bottom_level", bottom_level))

    rows = []

    def _append_metric_row(level_name, horizon_label, level_indices, y_t, y_p):
        y_t_flat = y_t.reshape(-1)
        y_p_flat = y_p.reshape(-1)
        mae = mean_absolute_error(y_t_flat, y_p_flat)
        wape = np.sum(np.abs(y_t_flat - y_p_flat)) / np.maximum(
            np.sum(np.abs(y_t_flat)), 1e-3
        )
        rmse = np.sqrt(mean_squared_error(y_t_flat, y_p_flat))
        if frozen_scale is None:
            mase_value = compute_mase(
                y_t, y_p, num_timesteps_in=num_timesteps_in
            )
            n_excluded = 0
            mase_version = "legacy_test_window_one_step_diagnostic"
        else:
            level_scale = frozen_scale[level_indices]
            per_node = ae_mase.compute_mase_per_node(y_t, y_p, level_scale)
            summary = ae_mase.macro_average_mase(
                per_node, list(range(len(level_indices)))
            )
            mase_value = summary["mase"]
            n_excluded = summary["n_excluded"]
            mase_version = ae_mase.MASE_VERSION
        rows.append({
            "Level": level_name,
            "Horizon": horizon_label,
            "MAE": mae,
            "RMSE": rmse,
            "MASE": mase_value,
            "MASE_n_excluded": n_excluded,
            "MASE_version": mase_version,
            "WAPE": wape * 100,
        })

    for level_name, level_indices in level_groups:
        if not level_indices:
            logging.info("Skipping %s: no nodes.", level_name)
            continue
        level_pred = predictions[:, level_indices, :]
        level_true = true_values[:, level_indices, :]
        _append_metric_row(
            level_name, "all", level_indices, level_true, level_pred
        )
        if num_horizons > 1:
            for h in range(num_horizons):
                _append_metric_row(
                    level_name, f"h{h + 1}", level_indices,
                    level_true[:, :, h:h + 1],
                    level_pred[:, :, h:h + 1],
                )

    level_metrics_df = pd.DataFrame(rows)
    metrics_path = f"{config['output_dir']}/{artifact_filename('level_metrics')}"
    level_metrics_df.to_csv(metrics_path, index=False)
    logging.info("Level metrics saved to %s", metrics_path)
    return level_metrics_df
