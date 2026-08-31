import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_base import BaseGCNGRUModel, build_mlp


DEFAULT_RECONCILE_MLP_HIDDEN = (128, 64)
DEFAULT_RECONCILE_MLP_DROPOUT = 0.2
DEFAULT_RECONCILE_MLP_LAYER_NORM = True
DEFAULT_RECONCILE_MLP_ACTIVATION = "relu"
DEFAULT_MIXING_MLP_HIDDEN = (64, 32)
DEFAULT_MIXING_MLP_DROPOUT = 0.2


def _map_nodes_per_horizon(mapping: nn.Module, values: torch.Tensor) -> torch.Tensor:
    """Apply a node-wise mapper on each forecast horizon independently.

    Args:
        mapping: module that maps node_dim -> target_node_dim.
        values: tensor with shape [S, N, H].
    Returns:
        Tensor with shape [S, target_node_dim, H].
    """
    mapped = mapping(values.permute(0, 2, 1).contiguous())  # [S, H, target_node_dim]
    return mapped.permute(0, 2, 1).contiguous()  # [S, target_node_dim, H]


def _aggregate_middle_from_bottom(
    bottom_values: torch.Tensor,
    mid_to_bottom_indices: list,
    bottom_start_idx: int,
    num_mid_nodes: int,
) -> torch.Tensor:
    """Aggregate bottom predictions to middle levels for every horizon.

    Args:
        bottom_values: [S, num_bottom_nodes, H]
    Returns:
        Middle-level predictions with shape [S, num_mid_nodes, H].
    """
    ssz, _, horizon = bottom_values.shape
    mid_values = bottom_values.new_zeros(ssz, num_mid_nodes, horizon)
    for i in range(num_mid_nodes):
        global_indices = mid_to_bottom_indices[i] if i < len(mid_to_bottom_indices) else []
        if not global_indices:
            continue
        local_indices = [idx - bottom_start_idx for idx in global_indices]
        mid_values[:, i, :] = bottom_values[:, local_indices, :].sum(dim=1)
    return mid_values


class BottomUpDirectGCGRU(BaseGCNGRUModel):
    """
    自下而上直接调和的GCN-GRU模型 (GCN-GRU-LP-BUD)
    设计思想：
      - 使用基础预测中底层节点的原始预测值；
      - 在原始空间用求和矩阵 S 聚合得到所有节点预测。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix: np.ndarray,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
    ):
        super(BottomUpDirectGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )
        # 保存求和矩阵
        self.register_buffer('sum_matrix', torch.tensor(sum_matrix, dtype=torch.float32))  # [N, num_bottom]

        # 基于 S 推导层次信息
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        # 基础预测（归一化+log 空间）
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, 123, 1]

        # 取底层节点预测
        bottom_predictions = base_predictions[:, self.bottom_start_idx:, :]  # [S, 118, 1]

        # 回到原始空间
        bottom_denorm = self.denormalize(bottom_predictions)
        bottom_delog = self.inverse_log(bottom_denorm)   # [S, 118, 1]

        # 原始空间用 S 矩阵构建完整预测
        hierarchical_predictions = torch.matmul(self.sum_matrix, bottom_delog)  # [S, 123, 1]

        return hierarchical_predictions


class NoReconcileGCGRU(BaseGCNGRUModel):
    """
    无调和模型 (GCN-GRU-LP-NO)
    设计思想：
      - 仅输出基础预测；
      - 不做层次调和，仅将预测回到原始尺度。
    """

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]
        base_denorm = self.denormalize(base_predictions)
        base_delog = self.inverse_log(base_denorm)
        return base_delog


class BottomUpLinearGCGRU(BaseGCNGRUModel):
    """
    自下而上线性调和的GCN-GRU模型 (GCN-GRU-LP-BUL)
    设计思想：
      - 在log+归一化空间用线性层从全图映射到底层节点；
      - 与底层原始预测做残差线性混合；
      - 在原始空间用 S 矩阵构建所有节点预测。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix: np.ndarray,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
    ):
        super(BottomUpLinearGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        # 保存求和矩阵
        self.register_buffer('sum_matrix', torch.tensor(sum_matrix, dtype=torch.float32))  # [N, num_bottom]

        # 基于 S 推导层次信息
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes

        # 从全图预测线性映射到底层节点（123 -> 118）
        self.bottom_mapping = nn.Linear(node_num, self.num_bottom_nodes)

        # 残差混合比例（全局，可学习）
        self.residual_ratio = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, 123, 1]

        # (1) 取log+归一化空间的底层预测
        base_bottom_pred = base_predictions[:, self.bottom_start_idx:, :]  # [S, 118, 1]

        # (2) 用全图预测线性映射生成“模型视角下”的底层预测
        mapped_bottom = _map_nodes_per_horizon(self.bottom_mapping, base_predictions)  # [S, 118, H]

        # (3) 残差线性混合（仍在归一化+log 空间中）
        alpha = torch.sigmoid(self.residual_ratio)
        bottom_pred = alpha * base_bottom_pred + (1 - alpha) * mapped_bottom  # [S, 118, 1]

        # (4) 回到原始空间
        bottom_denorm = self.denormalize(bottom_pred)
        bottom_delog = self.inverse_log(bottom_denorm)  # [S, 118, 1]

        # (5) 用 S 矩阵生成全部节点预测
        hierarchical_predictions = torch.matmul(self.sum_matrix, bottom_delog)  # [S, 123, H]

        return hierarchical_predictions


class BottomUpNonlinearGCGRU(BaseGCNGRUModel):
    """
    自下而上非线性调和的 GCN-GRU 模型 (GCN-GRU-LP-BUN)

    设计思想：
      1. BaseGCNGRUModel.forward 输出 base_predictions（log+归一化空间）；
      2. 在 log+归一化空间中：
         - 取底层基础预测 base_bottom；
         - 用一个 MLP 从全图预测 (123 维) 非线性映射到底层 mapped_bottom；
         - 用全局残差权重 alpha 做非线性残差混合得到 bottom_mixed；
      3. 仅在最后一步将 bottom_mixed 反归一化 + 逆 log 回到原始空间；
      4. 在原始空间用 S 矩阵从底层构造所有节点预测。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix: np.ndarray,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
        mlp_hidden_dims: list[int] | tuple[int, ...] = DEFAULT_RECONCILE_MLP_HIDDEN,
        mlp_dropout: float = DEFAULT_RECONCILE_MLP_DROPOUT,
        mlp_layer_norm: bool = DEFAULT_RECONCILE_MLP_LAYER_NORM,
        mlp_activation: str = DEFAULT_RECONCILE_MLP_ACTIVATION,
    ):
        super(BottomUpNonlinearGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        # S 矩阵：从底层节点求和构造各层节点
        self.register_buffer('sum_matrix', torch.tensor(sum_matrix, dtype=torch.float32))
        # 基于 S 推导层次信息
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes

        # 在 log+归一化空间的非线性 MLP: 全图预测 -> 底层预测
        self.bottom_mapping = build_mlp(
            in_dim=node_num,
            out_dim=self.num_bottom_nodes,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
            layer_norm=mlp_layer_norm,
            activation=mlp_activation,
        )

        # 全局残差混合比例（可训练，越接近 1 越信任 base_bottom）
        self.residual_ratio = nn.Parameter(torch.tensor(0.6), requires_grad=True)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        # 基础预测：log+归一化空间
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)      # [S, 123, 1]

        # (1) log+归一化空间的底层基础预测
        base_bottom = base_predictions[:, self.bottom_start_idx:, :]                  # [S, 118, 1]

        # (2) 用 MLP 从全图预测（log+norm 空间）映射到底层
        mapped_bottom = _map_nodes_per_horizon(self.bottom_mapping, base_predictions)  # [S, 118, H]

        # (3) 残差非线性混合（仍在 log+归一化空间）
        # 对 [S,118,1] 的 base_bottom 求一个 per-snapshot scalar
        sparsity_score = (base_bottom.abs() < 1e-3).float().mean(dim=(1, 2), keepdim=True)  # [S,1,1]
        alpha = torch.sigmoid(self.residual_ratio)  # 标量
        adaptive_alpha = alpha + (1 - alpha) * sparsity_score  # [S,1,1]
        bottom_mixed = adaptive_alpha * base_bottom + (1 - adaptive_alpha) * mapped_bottom

        # (4) 将 bottom_mixed 反归一化 + 逆 log 回到原始空间
        bottom_denorm = self.denormalize(bottom_mixed)            # 反归一化（log 空间）
        bottom_delog = self.inverse_log(bottom_denorm)            # [S, 118, 1] 原始空间

        # (5) 在原始空间用 S 矩阵构建全层次预测
        hierarchical_predictions = torch.matmul(self.sum_matrix, bottom_delog)  # [S, 123, H]

        return hierarchical_predictions


class TopDownDirectGCGRU(BaseGCNGRUModel):
    """
    自上而下直接调和的GCN-GRU模型 (GCN-GRU-LP-TDD)

    改进后的设计思想：
      - 仍然只使用基础模型的顶层 + 底层预测；
      - 先将基础预测中的底层预测在“原始空间”求和，得到每个样本的底层总量；
      - 用每个底层预测 / 底层总量 得到该节点的“模型比例”（model ratio）；
      - 用 顶层基础预测 × 模型比例 分配顶层总量到底层，得到调和后的底层预测；
      - 再对调和后的底层预测按 brand 求和，构造中间层调和预测；
      - 顶层保持为基础预测的顶层（回到原始空间），从而保证层次一致：顶层 = 所有底层之和。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        decomposition_matrix: np.ndarray,
        mid_to_bottom_indices: list,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
    ):
        super(TopDownDirectGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        # 仍然注册分解矩阵以便推导底层节点数（本类中不再直接使用该矩阵做比例）
        self.register_buffer(
            'decomposition_matrix',
            torch.tensor(decomposition_matrix, dtype=torch.float32)  # [num_bottom, 1]
        )
        self.mid_to_bottom_indices = mid_to_bottom_indices

        # 基于分解矩阵推导层次信息（底层起点 / 底层数 / 中间层数）
        self.num_bottom_nodes = int(self.decomposition_matrix.shape[0])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes  # 例如 123-118=5
        self.num_mid_nodes = self.bottom_start_idx - 1                 # 例如 5-1=4

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [S, N, F, T_in]
        返回值: 分层调和后的预测（原始尺度） [S, N, 1]
        """
        # 基础 GCN-GRU 预测（仍在归一化 + log 空间）
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]

        # 顶层节点基础预测（索引0）
        top_prediction = base_predictions[:, 0:1, :]  # [S, 1, 1]
        # 底层节点基础预测（连续的一段）
        bottom_base = base_predictions[:, self.bottom_start_idx :, :]  # [S, num_bottom, 1]

        # 回到原始空间（先反归一化，再逆 log）
        top_denorm = self.denormalize(top_prediction)
        top_orig = self.inverse_log(top_denorm)              # [S, 1, 1]

        bottom_denorm = self.denormalize(bottom_base)
        bottom_orig = self.inverse_log(bottom_denorm)        # [S, num_bottom, 1]

        # === 用基础底层预测构造“模型比例” ===
        # 每个样本底层总量
        bottom_sum = bottom_orig.sum(dim=1, keepdim=True)    # [S, 1, 1]

        # 为避免除零，设置一个极小值
        eps = 1e-6
        # 基本比例：bottom_i / sum_bottom
        base_ratio = bottom_orig / (bottom_sum + eps)        # [S, num_bottom, 1]

        # 若某个样本 bottom_sum≈0，则用均匀分布代替
        # mask: [S, 1, 1] -> [S, num_bottom, 1]
        zero_mask = (bottom_sum <= eps)
        if zero_mask.any():
            zero_mask_expanded = zero_mask.expand(-1, self.num_bottom_nodes, -1)
            uniform_ratio = torch.full_like(bottom_orig, 1.0 / self.num_bottom_nodes)
            model_ratio = torch.where(zero_mask_expanded, uniform_ratio, base_ratio)
        else:
            model_ratio = base_ratio  # [S, num_bottom, 1]

        # === 用顶层基础预测 × 模型比例 分配到底层 ===
        # 顶层扩展到每个底层节点
        top_expanded = top_orig.expand(-1, self.num_bottom_nodes, -1)  # [S, num_bottom, 1]
        bottom_reconciled = top_expanded * model_ratio                 # [S, num_bottom, 1]

        # === 按品牌层结构求和，得到中间层调和预测 ===
        mid_predictions = _aggregate_middle_from_bottom(
            bottom_reconciled,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        # === 组合最终预测 ===
        hierarchical_predictions = torch.cat([
            top_orig,          # 顶层 [S, 1, 1]
            mid_predictions,   # 中间层 [S, num_mid_nodes, 1]
            bottom_reconciled  # 底层 [S, num_bottom_nodes, 1]
        ], dim=1)  # [S, node_num, 1]

        return hierarchical_predictions


class TopDownLinearGCGRU(BaseGCNGRUModel):
    """
    自上而下线性调和的GCN-GRU模型 (GCN-GRU-LP-TDL)

    新设计思想：
      - 先用基础预测的底层节点在原始空间构造“模型比例”（model ratio）；
      - 再用一个线性映射（ratio_mapping）从全图基础预测中学习对比例的残差修正；
      - 通过 log-space 残差：log r_i' ∝ log r_i + Δ_i，并用 softmax 保证 r_i' ≥ 0 且和为1；
      - 最后用 顶层基础预测 × 修正后的比例 分配到底层，再按品牌求和得到中间层。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        decomposition_matrix: np.ndarray,
        mid_to_bottom_indices: list,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
    ):
        super(TopDownLinearGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        # 仍然注册分解矩阵以推导层次信息（本类中不再直接用它做比例）
        self.register_buffer(
            'decomposition_matrix',
            torch.tensor(decomposition_matrix, dtype=torch.float32)  # [num_bottom, 1]
        )
        self.mid_to_bottom_indices = mid_to_bottom_indices

        # 基于分解矩阵推导层次信息（底层起点 / 底层数 / 中间层数）
        self.num_bottom_nodes = int(self.decomposition_matrix.shape[0])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes  # 例如 123-118=5
        self.num_mid_nodes = self.bottom_start_idx - 1                 # 例如 5-1=4

        # 线性映射：从全图基础预测映射到“比例残差”Δ_i（对每个底层节点一个残差）
        self.ratio_mapping = nn.Linear(node_num, self.num_bottom_nodes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [S, N, F, T_in]
        返回：层次调和后的预测（原始尺度） [S, N, 1]
        """
        # 基础 GCN-GRU 预测（归一化 + log 空间）
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]

        # 顶层与底层基础预测
        top_prediction = base_predictions[:, 0:1, :]                     # [S, 1, 1]
        bottom_base = base_predictions[:, self.bottom_start_idx :, :]    # [S, num_bottom, 1]

        # 回到原始空间
        top_denorm = self.denormalize(top_prediction)
        top_orig = self.inverse_log(top_denorm)                          # [S, 1, 1]

        bottom_denorm = self.denormalize(bottom_base)
        bottom_orig = self.inverse_log(bottom_denorm)                    # [S, num_bottom, 1]

        # 1) 将底层负值裁剪到 >=0
        bottom_orig = torch.clamp(bottom_orig, min=0.0)

        # === 第一步：用基础底层预测构造“模型比例” r_i ===
        bottom_sum = bottom_orig.sum(dim=1, keepdim=True)                # [S, 1, 1]
        eps = 1e-6
        # 3) 若 bottom_sum==0 → 使用均匀比例
        zero_mask = (bottom_sum < eps)  # [S,1,1]
        if zero_mask.any():
            uniform_ratio = torch.full_like(bottom_orig, 1.0 / bottom_orig.size(1))
            base_ratio = torch.where(zero_mask, uniform_ratio, bottom_orig / (bottom_sum + eps))
        else:
            base_ratio = bottom_orig / (bottom_sum + 1e-6)

            # === 第二步：线性残差修正比例 log-space 残差 ===
        # Δ_i = Linear(flat_predictions)，对每个 horizon 独立映射
        delta = _map_nodes_per_horizon(self.ratio_mapping, base_predictions)  # [S, num_bottom, H]

        # log r_i' ∝ log r_i + Δ_i ；用 softmax 归一化
        log_base = torch.log(base_ratio + eps)                           # [S, num_bottom, H]
        log_corrected = log_base + delta                                 # [S, num_bottom, H]
        corrected_ratio = torch.softmax(log_corrected, dim=1)            # [S, num_bottom, H]

        # === 第三步：用顶层基础预测 × 修正后的比例 分配到底层 ===
        top_expanded = top_orig.expand(-1, self.num_bottom_nodes, -1)    # [S, num_bottom, 1]
        bottom_reconciled = top_expanded * corrected_ratio               # [S, num_bottom, 1]

        # === 第四步：按品牌结构求和得到中间层 ===
        mid_predictions = _aggregate_middle_from_bottom(
            bottom_reconciled,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        # === 最终组合 ===
        hierarchical_predictions = torch.cat([
            top_orig,          # [S, 1, 1]
            mid_predictions,   # [S, num_mid_nodes, 1]
            bottom_reconciled  # [S, num_bottom_nodes, 1]
        ], dim=1)  # [S, node_num, 1]

        return hierarchical_predictions


class TopDownNonlinearGCGRU(BaseGCNGRUModel):
    """
    自上而下非线性调和的GCN-GRU模型 (GCN-GRU-LP-TDN)

    新设计思想：
      - 同样先用基础底层预测构造“模型比例”；
      - 再用一个非线性 MLP（ratio_mapping）从全图基础预测中学习对比例的残差修正；
      - 使用 log-space 残差：log r_i' ∝ log r_i + Δ_i，并用 softmax 得到修正后比例；
      - 最后用 顶层基础预测 × 修正后的比例 分配到底层，再构建中间层。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        decomposition_matrix: np.ndarray,
        mid_to_bottom_indices: list,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
        mlp_hidden_dims: list[int] | tuple[int, ...] = DEFAULT_RECONCILE_MLP_HIDDEN,
        mlp_dropout: float = DEFAULT_RECONCILE_MLP_DROPOUT,
        mlp_layer_norm: bool = DEFAULT_RECONCILE_MLP_LAYER_NORM,
        mlp_activation: str = DEFAULT_RECONCILE_MLP_ACTIVATION,
    ):
        super(TopDownNonlinearGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        # 仍然注册分解矩阵以推导层次信息
        self.register_buffer(
            'decomposition_matrix',
            torch.tensor(decomposition_matrix, dtype=torch.float32)  # [num_bottom, 1]
        )
        self.mid_to_bottom_indices = mid_to_bottom_indices

        # 基于分解矩阵推导层次信息（底层起点 / 底层数 / 中间层数）
        self.num_bottom_nodes = int(self.decomposition_matrix.shape[0])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes
        self.num_mid_nodes = self.bottom_start_idx - 1

        # 非线性映射：从全图基础预测映射到“比例残差”Δ_i（对每个底层节点一个残差）
        self.ratio_mapping = build_mlp(
            in_dim=node_num,
            out_dim=self.num_bottom_nodes,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
            layer_norm=mlp_layer_norm,
            activation=mlp_activation,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [S, N, F, T_in]
        返回：层次调和后的预测（原始尺度） [S, N, 1]
        """
        # 基础 GCN-GRU 预测（归一化 + log 空间）
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]

        # 顶层与底层基础预测
        top_prediction = base_predictions[:, 0:1, :]                     # [S, 1, 1]
        bottom_base = base_predictions[:, self.bottom_start_idx :, :]    # [S, num_bottom, 1]

        # 回到原始空间
        top_denorm = self.denormalize(top_prediction)
        top_orig = self.inverse_log(top_denorm)                          # [S, 1, 1]

        bottom_denorm = self.denormalize(bottom_base)
        bottom_orig = self.inverse_log(bottom_denorm)                    # [S, num_bottom, 1]

        bottom_orig = torch.clamp(bottom_orig, min=0.0)

        # === 第一步：基础模型比例 r_i ===
        bottom_sum = bottom_orig.sum(dim=1, keepdim=True)                # [S, 1, 1]
        eps = 1e-6
        zero_mask = (bottom_sum <= eps)  # [S,1,1]
        if zero_mask.any():
            # 均匀比例兜底
            uniform_ratio = torch.full_like(bottom_orig, 1.0 / bottom_orig.size(1))
            base_ratio = torch.where(
                zero_mask.expand_as(bottom_orig),
                uniform_ratio,
                bottom_orig / (bottom_sum + eps)
            )
        else:
            base_ratio = bottom_orig / (bottom_sum + eps)

        # === 第二步：非线性残差修正比例（log-space） ===
        delta = _map_nodes_per_horizon(self.ratio_mapping, base_predictions)  # [S, num_bottom, H]

        log_base = torch.log(base_ratio + eps)                           # [S, num_bottom, H]
        log_corrected = log_base + delta                                 # [S, num_bottom, H]
        log_corrected = torch.clamp(log_corrected, -20.0, 20.0)
        corrected_ratio = torch.softmax(log_corrected, dim=1)            # [S, num_bottom, H]

        # === 第三步：顶层 × 修正比例 → 底层调和预测 ===
        top_expanded = top_orig.expand(-1, self.num_bottom_nodes, -1)    # [S, num_bottom, 1]
        bottom_reconciled = top_expanded * corrected_ratio               # [S, num_bottom, 1]

        # === 第四步：按品牌求和得到中间层 ===
        mid_predictions = _aggregate_middle_from_bottom(
            bottom_reconciled,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        # === 最终组合 ===
        hierarchical_predictions = torch.cat([
            top_orig,
            mid_predictions,
            bottom_reconciled
        ], dim=1)  # [S, node_num, 1]

        return hierarchical_predictions


class HybridDirectGCGRU(BaseGCNGRUModel):
    """
    直接混合调和的GCN-GRU模型 (GCN-GRU-LP-HYBD)

    设计思想（对称 BU/TD）：
      - 用基础底层预测在原始空间构造“模型比例” base_ratio；
      - TD 分支：用基础顶层预测 top_orig × base_ratio 分配到底层，构造 TD 层次预测；
      - BU 分支：用基础底层总和 bottom_sum × base_ratio 分配到底层，构造 BU 层次预测；
      - 两套预测都满足层次一致（top = sum(mid) = sum(bottom)）；
      - 最后按 alpha 在原始空间线性混合。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix: np.ndarray,
        decomposition_matrix: np.ndarray,
        mid_to_bottom_indices: list,
        alpha: float = 0.5,
        alpha_trainable: bool = True,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
    ):
        super(HybridDirectGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        # 保存矩阵和索引映射（本类不再直接用 sum_matrix / decomposition_matrix 做分配）
        self.register_buffer('sum_matrix', torch.tensor(sum_matrix, dtype=torch.float32))  # [N, num_bottom]
        self.register_buffer('decomposition_matrix',
                             torch.tensor(decomposition_matrix, dtype=torch.float32))      # [num_bottom, 1]
        self.mid_to_bottom_indices = mid_to_bottom_indices

        # 基于 sum_matrix / decomposition_matrix 推导层次信息
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes
        self.num_mid_nodes = self.bottom_start_idx - 1

        # 混合权重参数
        if alpha_trainable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha  # 固定混合权重

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [S, N, F, T_in]
        返回: 混合后的层次预测（原始尺度） [S, N, 1]
        """
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]

        # 顶层与底层基础预测（归一化 + log 空间）
        top_prediction = base_predictions[:, 0:1, :]                  # [S, 1, 1]
        bottom_base = base_predictions[:, self.bottom_start_idx :, :] # [S, num_bottom, 1]

        # 回到原始空间
        top_denorm = self.denormalize(top_prediction)
        top_orig = self.inverse_log(top_denorm)                       # [S, 1, 1]

        bottom_denorm = self.denormalize(bottom_base)
        bottom_orig = self.inverse_log(bottom_denorm)                 # [S, num_bottom, 1]

        # === 构造“模型比例” base_ratio（两分支共用，无残差） ===
        bottom_sum = bottom_orig.sum(dim=1, keepdim=True)             # [S, 1, 1]
        eps = 1e-6
        base_ratio = bottom_orig / (bottom_sum + eps)                 # [S, num_bottom, 1]

        # 底层总量≈0 时用均匀比例兜底
        zero_mask = (bottom_sum <= eps)
        if zero_mask.any():
            zero_mask_expanded = zero_mask.expand(-1, self.num_bottom_nodes, -1)
            uniform_ratio = torch.full_like(bottom_orig, 1.0 / self.num_bottom_nodes)
            base_ratio = torch.where(zero_mask_expanded, uniform_ratio, base_ratio)

        # ----- TD 分支：top_orig × base_ratio -----
        top_TD = top_orig                                              # [S, 1, 1]
        top_TD_expanded = top_TD.expand(-1, self.num_bottom_nodes, -1) # [S, num_bottom, 1]
        bottom_TD = top_TD_expanded * base_ratio                       # [S, num_bottom, 1]

        mid_TD = _aggregate_middle_from_bottom(
            bottom_TD,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        td_predictions = torch.cat([
            top_TD,
            mid_TD,
            bottom_TD
        ], dim=1)                                                      # [S, N, 1]

        # ----- BU 分支：bottom_sum × base_ratio -----
        top_BU = bottom_sum                                            # [S, 1, 1]
        top_BU_expanded = top_BU.expand(-1, self.num_bottom_nodes, -1) # [S, num_bottom, 1]
        bottom_BU = top_BU_expanded * base_ratio                       # [S, num_bottom, 1]

        mid_BU = _aggregate_middle_from_bottom(
            bottom_BU,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        bu_predictions = torch.cat([
            top_BU,
            mid_BU,
            bottom_BU
        ], dim=1)                                                      # [S, N, 1]

        # ----- 混合 -----
        alpha_sigmoid = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
        hybrid_predictions = alpha_sigmoid * td_predictions + (1 - alpha_sigmoid) * bu_predictions

        return hybrid_predictions


class HybridLinearGCGRU(BaseGCNGRUModel):
    """
    线性混合调和的GCN-GRU模型 (GCN-GRU-LP-HYBL)
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix: np.ndarray,
        decomposition_matrix: np.ndarray,
        mid_to_bottom_indices: list,
        alpha: float = 0.5,
        alpha_trainable: bool = True,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
    ):
        super(HybridLinearGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        self.register_buffer('sum_matrix', torch.tensor(sum_matrix, dtype=torch.float32))  # [N, num_bottom]
        self.register_buffer('decomposition_matrix',
                             torch.tensor(decomposition_matrix, dtype=torch.float32))      # [num_bottom, 1]
        self.mid_to_bottom_indices = mid_to_bottom_indices

        # 基于 sum_matrix 推导层次信息
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes
        self.num_mid_nodes = self.bottom_start_idx - 1

        # 混合权重参数
        if alpha_trainable:
            self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = alpha

        # 线性映射：从全图基础预测映射到“比例残差” Δ_i
        self.ratio_mapping = nn.Linear(node_num, self.num_bottom_nodes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [S, N, F, T_in]
        返回: 混合后的层次预测 [S, N, 1]
        """
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]
        eps = 1e-6

        # 顶层与底层基础预测（仍在归一化 + log 空间）
        top_prediction    = base_predictions[:, 0:1, :]                    # [S, 1, 1]
        bottom_base       = base_predictions[:, self.bottom_start_idx:, :] # [S, num_bottom, 1]

        # 回到原始空间（真实尺度）
        top_denorm        = self.denormalize(top_prediction)
        bottom_denorm     = self.denormalize(bottom_base)

        top_orig          = self.inverse_log(top_denorm)                   # [S, 1, 1]
        bottom_orig       = self.inverse_log(bottom_denorm)                # [S, num_bottom, 1]

        # === 关键修改：仅用于构造比例的部分做平滑非负 ===
        # 使用 softplus(x) >= 0，且对负数仍然有梯度，比 clamp 更温和
        bottom_pos = F.softplus(bottom_orig)                               # [S, num_bottom, 1]

        # 第一步：基础比例 base_ratio（基于 bottom_pos）
        bottom_sum = bottom_pos.sum(dim=1, keepdim=True)                   # [S, 1, 1]

        base_ratio = bottom_pos / (bottom_sum + eps)                       # [S, num_bottom, 1]

        # 若总量过小，用均匀比例兜底
        zero_mask = (bottom_sum <= eps)                                    # [S,1,1]
        if zero_mask.any():
            zero_mask_expanded = zero_mask.expand(-1, self.num_bottom_nodes, -1)
            uniform_ratio      = torch.full_like(bottom_pos, 1.0 / self.num_bottom_nodes)
            base_ratio         = torch.where(zero_mask_expanded, uniform_ratio, base_ratio)

        # 第二步：线性残差 Δ_i，log-space 校正比例
        delta = _map_nodes_per_horizon(self.ratio_mapping, base_predictions)  # [S, num_bottom, H]

        base_ratio_safe = torch.clamp(base_ratio, min=eps)                 # 防止 log(0)

        log_base      = torch.log(base_ratio_safe)                         # [S, num_bottom, H]
        log_corrected = log_base + delta                                   # [S, num_bottom, H]
        log_corrected = torch.clamp(log_corrected, min=-20.0, max=20.0)    # 防止 softmax 溢出

        corrected_ratio = torch.softmax(log_corrected, dim=1)              # [S, num_bottom, H]

        # ----- TD 分支：top_orig × corrected_ratio -----
        top_TD          = top_orig                                         # [S, 1, 1]（保持真实顶层预测，不再 clamp）
        top_TD_expanded = top_TD.expand(-1, self.num_bottom_nodes, -1)
        bottom_TD       = top_TD_expanded * corrected_ratio                # [S, num_bottom, 1]

        mid_TD = _aggregate_middle_from_bottom(
            bottom_TD,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        td_predictions = torch.cat([top_TD, mid_TD, bottom_TD], dim=1)     # [S, N, 1]

        # ----- BU 分支：bottom_sum × corrected_ratio -----
        # 这里的 bottom_sum 来自 bottom_pos，保证 >=0
        top_BU          = bottom_sum                                       # [S, 1, 1]
        top_BU_expanded = top_BU.expand(-1, self.num_bottom_nodes, -1)
        bottom_BU       = top_BU_expanded * corrected_ratio                # [S, num_bottom, 1]

        mid_BU = _aggregate_middle_from_bottom(
            bottom_BU,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        bu_predictions = torch.cat([top_BU, mid_BU, bottom_BU], dim=1)     # [S, N, 1]

        # ----- 线性混合 -----
        alpha_sigmoid = torch.sigmoid(self.alpha) if isinstance(self.alpha, nn.Parameter) else self.alpha
        hybrid_predictions = alpha_sigmoid * td_predictions + (1 - alpha_sigmoid) * bu_predictions

        return hybrid_predictions


class HybridNonlinearGCGRU(BaseGCNGRUModel):
    """
    非线性混合调和的GCN-GRU模型 (GCN-GRU-LP-HYBN)

    设计思想：
      - 用基础底层预测构造 base_ratio；
      - 用非线性 MLP ratio_mapping(flat_predictions) 得到残差 Δ_i；
      - 得到 corrected_ratio（log-space + softmax）；
      - TD 分支：top_orig × corrected_ratio → 层次预测；
      - BU 分支：bottom_sum × corrected_ratio → 层次预测；
      - mixing_network 为每个节点生成 [0,1] 混合权重，对 TD / BU 两套预测做节点级非线性混合。
    """

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix: np.ndarray,
        decomposition_matrix: np.ndarray,
        mid_to_bottom_indices: list,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
        mlp_hidden_dims: list[int] | tuple[int, ...] = DEFAULT_RECONCILE_MLP_HIDDEN,
        mlp_dropout: float = DEFAULT_RECONCILE_MLP_DROPOUT,
        mlp_layer_norm: bool = DEFAULT_RECONCILE_MLP_LAYER_NORM,
        mlp_activation: str = DEFAULT_RECONCILE_MLP_ACTIVATION,
        mixing_hidden_dims: list[int] | tuple[int, ...] = DEFAULT_MIXING_MLP_HIDDEN,
        mixing_dropout: float = DEFAULT_MIXING_MLP_DROPOUT,
    ):
        super(HybridNonlinearGCGRU, self).__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
            gnn_type=gnn_type,
            temporal_type=temporal_type,
        )

        self.register_buffer('sum_matrix', torch.tensor(sum_matrix, dtype=torch.float32))
        self.register_buffer('decomposition_matrix',
                             torch.tensor(decomposition_matrix, dtype=torch.float32))
        self.mid_to_bottom_indices = mid_to_bottom_indices

        # 基于 sum_matrix 推导层次信息
        self.num_bottom_nodes = int(self.sum_matrix.shape[1])
        self.bottom_start_idx = self.node_num - self.num_bottom_nodes
        self.num_mid_nodes = self.bottom_start_idx - 1

        # 非线性映射：从全图基础预测映射到“比例残差” Δ_i
        self.ratio_mapping = build_mlp(
            in_dim=node_num,
            out_dim=self.num_bottom_nodes,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout,
            layer_norm=mlp_layer_norm,
            activation=mlp_activation,
        )

        # 非线性混合网络：根据 (TD, BU) 两套预测为每个节点生成混合权重
        self.mixing_network = build_mlp(
            in_dim=2,
            out_dim=1,
            hidden_dims=mixing_hidden_dims,
            dropout=mixing_dropout,
            layer_norm=mlp_layer_norm,
            activation=mlp_activation,
            final_activation=nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [S, N, F, T_in]
        返回: 混合后的层次预测 [S, N, 1]
        """
        _, base_predictions = super().forward(x, edge_index, edge_weight=edge_weight)  # [S, N, 1]
        eps = 1e-6  # 数值稳定小常数

        # 顶层与底层基础预测
        top_prediction = base_predictions[:, 0:1, :]                  # [S, 1, 1]
        bottom_base = base_predictions[:, self.bottom_start_idx :, :] # [S, num_bottom, 1]

        # 回到原始空间
        top_denorm = self.denormalize(top_prediction)
        bottom_denorm = self.denormalize(bottom_base)

        top_orig = self.inverse_log(top_denorm)                       # [S, 1, 1]
        bottom_orig = self.inverse_log(bottom_denorm)                 # [S, num_bottom, 1]

        # 只对底层做“平滑非负”，保留梯度
        bottom_orig = F.softplus(bottom_orig)  # 始终 >= 0，且在 0 左右梯度不为 0

        # === 第一步：基础比例 base_ratio ===
        bottom_sum = bottom_orig.sum(dim=1, keepdim=True)             # [S, 1, 1]

        base_ratio = bottom_orig / (bottom_sum + eps)                 # [S, num_bottom, 1]

        # 当总和过小（全 0 或接近 0）时，用均匀比例兜底
        zero_mask = (bottom_sum <= eps)
        if zero_mask.any():
            zero_mask_expanded = zero_mask.expand(-1, self.num_bottom_nodes, -1)
            uniform_ratio = torch.full_like(bottom_orig, 1.0 / self.num_bottom_nodes)
            base_ratio = torch.where(zero_mask_expanded, uniform_ratio, base_ratio)

        # === 第二步：非线性残差 Δ_i，log-space 校正比例 ===
        delta = _map_nodes_per_horizon(self.ratio_mapping, base_predictions)  # [S, num_bottom, H]

        # === [稳定性防护 2] 比例在 log 前强制 >= eps，彻底杜绝 log(<=0) ===
        base_ratio_safe = torch.clamp(base_ratio, min=eps)

        log_base = torch.log(base_ratio_safe)                         # [S, num_bottom, H]
        log_corrected = log_base + delta                              # [S, num_bottom, H]
        corrected_ratio = torch.softmax(log_corrected, dim=1)         # [S, num_bottom, H]

        # ----- TD 分支：top_orig × corrected_ratio -----
        top_TD = top_orig                                             # [S, 1, 1]
        top_TD_expanded = top_TD.expand(-1, self.num_bottom_nodes, -1)
        bottom_TD = top_TD_expanded * corrected_ratio                 # [S, num_bottom, 1]

        mid_TD = _aggregate_middle_from_bottom(
            bottom_TD,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        td_predictions = torch.cat([
            top_TD,
            mid_TD,
            bottom_TD
        ], dim=1)                                                     # [S, N, 1]

        # ----- BU 分支：bottom_sum × corrected_ratio -----
        # bottom_sum 由 bottom_orig>=0 得到，因此非负
        top_BU = bottom_sum                                           # [S, 1, 1]
        top_BU_expanded = top_BU.expand(-1, self.num_bottom_nodes, -1)
        bottom_BU = top_BU_expanded * corrected_ratio                 # [S, num_bottom, 1]

        mid_BU = _aggregate_middle_from_bottom(
            bottom_BU,
            self.mid_to_bottom_indices,
            self.bottom_start_idx,
            self.num_mid_nodes,
        )

        bu_predictions = torch.cat([
            top_BU,
            mid_BU,
            bottom_BU
        ], dim=1)                                                     # [S, N, 1]

        # ----- 非线性节点级混合 -----
        # 对每个节点拼接 (TD, BU) 两个标量，喂入 mixing_network 得到混合权重
        combined = torch.stack([td_predictions, bu_predictions], dim=-1)  # [S, N, H, 2]
        mixing_weights = self.mixing_network(combined).squeeze(-1)        # [S, N, H]

        hybrid_predictions = mixing_weights * td_predictions + \
                             (1 - mixing_weights) * bu_predictions

        return hybrid_predictions
