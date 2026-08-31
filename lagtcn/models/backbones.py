import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, TransformerConv
from torch_geometric.utils import dense_to_sparse
from lagtcn.core.graphs import (
    FINAL_GRAPH_SOURCE_POLICY,
    build_topk_similarity_adj_torch,
)


def _require_finite_transform(value: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        finite = value.detach()[torch.isfinite(value)]
        summary = {
            "shape": tuple(value.shape),
            "finite_min": float(finite.min()) if finite.numel() else None,
            "finite_max": float(finite.max()) if finite.numel() else None,
            "nonfinite_count": int((~torch.isfinite(value)).sum()),
        }
        raise FloatingPointError(f"Non-finite {name}: {summary}")
    return value


def _choose_nhead(hidden_dim: int) -> int:
    for nhead in (8, 4, 2, 1):
        if hidden_dim % nhead == 0:
            return nhead
    return 1


_ACTIVATION_REGISTRY = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def build_mlp(
    in_dim: int,
    out_dim: int,
    hidden_dims: list[int] | tuple[int, ...] | None,
    dropout: float = 0.0,
    layer_norm: bool = True,
    activation: str = "relu",
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Generic MLP builder used by reconciliation heads.

    Each hidden layer follows: Linear -> [LayerNorm] -> activation -> [Dropout].
    The final projection is a bare Linear (plus optional `final_activation`).
    """
    act_key = str(activation).lower()
    if act_key not in _ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation '{activation}'. Choose from {sorted(_ACTIVATION_REGISTRY)}."
        )
    act_cls = _ACTIVATION_REGISTRY[act_key]
    dims = list(hidden_dims) if hidden_dims else []
    dropout = float(dropout)
    use_ln = bool(layer_norm)

    layers: list[nn.Module] = []
    prev = int(in_dim)
    for h in dims:
        h = int(h)
        layers.append(nn.Linear(prev, h))
        if use_ln:
            layers.append(nn.LayerNorm(h))
        layers.append(act_cls())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, int(out_dim)))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class SeriesDecomp(nn.Module):
    """Moving-average based trend/seasonal decomposition.

    Returns (seasonal, trend) for an input of shape [B, T, H].
    Uses replicate padding so output length equals input length.
    """

    def __init__(self, kernel_size: int = 13):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T = x.shape[1]
        k = self.kernel_size
        if T < 3:
            return x, torch.zeros_like(x)
        if k > T:
            k = T if T % 2 == 1 else T - 1
        pad = k // 2
        x_t = x.transpose(1, 2)                        # [B, H, T]
        padded = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=k, stride=1).transpose(1, 2)
        trend = trend[:, :T, :]                         # safety
        seasonal = x - trend
        return seasonal, trend


class _TimeMixerBlock(nn.Module):
    """One Past-Decomposable-Mixing block.

    Seasonal components mix bottom-up (fine -> coarse).
    Trend components mix top-down (coarse -> fine).
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.seasonal_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.trend_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, seasonal_scales, trend_scales, down_factor):
        mixed_seas = [seasonal_scales[0]]
        for i in range(1, len(seasonal_scales)):
            T_cur = seasonal_scales[i].shape[1]
            prev_ds = F.avg_pool1d(
                mixed_seas[-1].transpose(1, 2), kernel_size=down_factor,
            ).transpose(1, 2)
            if prev_ds.shape[1] > T_cur:
                prev_ds = prev_ds[:, :T_cur]
            elif prev_ds.shape[1] < T_cur:
                prev_ds = F.pad(prev_ds, (0, 0, 0, T_cur - prev_ds.shape[1]))
            mixed_seas.append(seasonal_scales[i] + self.seasonal_mlp(prev_ds))

        mixed_tr = [None] * len(trend_scales)
        mixed_tr[-1] = trend_scales[-1]
        for i in range(len(trend_scales) - 2, -1, -1):
            T_cur = trend_scales[i].shape[1]
            prev_us = F.interpolate(
                mixed_tr[i + 1].transpose(1, 2),
                size=T_cur, mode="linear", align_corners=False,
            ).transpose(1, 2)
            mixed_tr[i] = trend_scales[i] + self.trend_mlp(prev_us)
        return mixed_seas, mixed_tr


class TimeMixerEncoder(nn.Module):
    """Simplified TimeMixer temporal encoder (Wang et al., ICLR 2024).

    Multi-scale downsampling + series decomposition. Seasonal components mix
    bottom-up (fine -> coarse); trend components mix top-down (coarse -> fine).
    Interface matches the other temporal encoders in this file:
        Input  [B, T, H] -> Output [B, H] (representation of last timestep).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        num_scales: int = 3,
        down_factor: int = 2,
        moving_avg: int = 13,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_scales = max(1, int(num_scales))
        self.down_factor = max(2, int(down_factor))
        self.decomp = SeriesDecomp(moving_avg)
        self.blocks = nn.ModuleList(
            [_TimeMixerBlock(hidden_dim, dropout) for _ in range(max(1, int(num_layers)))]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _multi_scale_inputs(self, x: torch.Tensor) -> list[torch.Tensor]:
        scales = [x]
        for _ in range(self.num_scales - 1):
            prev = scales[-1]
            if prev.shape[1] < self.down_factor * 2:
                break
            down = F.avg_pool1d(prev.transpose(1, 2), kernel_size=self.down_factor).transpose(1, 2)
            scales.append(down)
        return scales

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scales = self._multi_scale_inputs(x)
        seasonal_scales, trend_scales = [], []
        for s in scales:
            seas, tr = self.decomp(s)
            seasonal_scales.append(seas)
            trend_scales.append(tr)
        for block in self.blocks:
            seasonal_scales, trend_scales = block(seasonal_scales, trend_scales, self.down_factor)
        out = seasonal_scales[0] + trend_scales[0]
        out = self.norm(out + x)
        return out[:, -1, :]


class PatchTSTEncoder(nn.Module):
    """Patch-based Transformer temporal encoder (Nie et al., ICLR 2023).

    Splits the time dimension into overlapping patches, linearly embeds each
    patch into a token, and runs a Transformer encoder over the patch tokens.
    Interface:
        Input  [B, T, H] -> Output [B, H] (mean pooled over patch tokens).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        patch_len: int = 12,
        stride: int | None = None,
        dropout: float = 0.2,
        max_patches: int = 512,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_len = max(1, int(patch_len))
        self.stride = max(1, int(stride) if stride else max(1, self.patch_len // 2))
        self.patch_embed = nn.Linear(self.patch_len * hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nhead = _choose_nhead(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))

        # Precomputed sinusoidal positional encoding for up to max_patches tokens.
        pe = torch.zeros(1, int(max_patches), hidden_dim)
        pos = torch.arange(int(max_patches), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[0, :, 0::2] = torch.sin(pos * div_term)
        pe[0, :, 1::2] = torch.cos(pos * div_term)
        self.register_buffer("pos_pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, H = x.shape
        if T < self.patch_len:
            x = F.pad(x, (0, 0, 0, self.patch_len - T))
            T = self.patch_len
        num_patches = (T - self.patch_len) // self.stride + 1
        required_len = (num_patches - 1) * self.stride + self.patch_len
        if required_len < T:
            x = x[:, :required_len]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # unfold -> [B, num_patches, H, patch_len]
        patches = patches.permute(0, 1, 3, 2).contiguous()
        patches = patches.reshape(B, num_patches, self.patch_len * H)
        tokens = self.patch_embed(patches)
        tokens = self.dropout(tokens + self.pos_pe[:, :num_patches, :])
        out = self.transformer(tokens)
        return out.mean(dim=1)


class DLinearTemporalEncoder(nn.Module):
    """DLinear-style temporal encoder for graph-enhanced base forecasters.

    This module keeps the BaseGCNGRUModel contract:
        Input  [B, T, H] -> Output [B, H]
    It applies a moving-average decomposition over time and learns one
    linear projection from the full input window to a hidden summary for each
    hidden channel.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_timesteps_in: int,
        moving_avg_window: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_timesteps_in = int(num_timesteps_in)
        kernel = max(1, int(moving_avg_window))
        if kernel % 2 == 0:
            kernel += 1
        self.moving_avg_window = kernel
        self.trend_linear = nn.Linear(self.num_timesteps_in, 1)
        self.seasonal_linear = nn.Linear(self.num_timesteps_in, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.constant_(self.trend_linear.weight, 1.0 / self.num_timesteps_in)
        nn.init.constant_(self.seasonal_linear.weight, 1.0 / self.num_timesteps_in)
        nn.init.zeros_(self.trend_linear.bias)
        nn.init.zeros_(self.seasonal_linear.bias)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, T]
        pad = self.moving_avg_window // 2
        x_pad = F.pad(x, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.moving_avg_window, stride=1)
        return trend[:, :, : x.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        B, T, H = x.shape
        if T != self.num_timesteps_in:
            raise ValueError(f"DLinear temporal encoder expected T_in={self.num_timesteps_in}, got {T}")
        x_ch = x.transpose(1, 2)  # [B, H, T]
        trend = self._moving_average(x_ch)
        seasonal = x_ch - trend
        hidden = self.trend_linear(trend).squeeze(-1) + self.seasonal_linear(seasonal).squeeze(-1)
        return self.norm(self.dropout(hidden))


class ITransformerTemporalEncoder(nn.Module):
    """iTransformer-style temporal encoder for graph-enhanced forecasters.

    After graph propagation, each hidden channel is treated as an inverted
    token whose features are the values over the input window. Attention is
    applied over hidden-channel tokens, then the diagonal token summaries are
    returned as the node-level temporal representation.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_timesteps_in: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_timesteps_in = int(num_timesteps_in)
        self.token_proj = nn.Linear(self.num_timesteps_in, self.hidden_dim)
        nhead = _choose_nhead(self.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=nhead,
            dim_feedforward=self.hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        B, T, H = x.shape
        if T != self.num_timesteps_in:
            raise ValueError(f"iTransformer temporal encoder expected T_in={self.num_timesteps_in}, got {T}")
        tokens = self.token_proj(x.transpose(1, 2))  # [B, H, H]
        encoded = self.encoder(tokens)
        # The diagonal keeps one attended token summary per hidden channel.
        hidden = torch.diagonal(encoded, dim1=1, dim2=2)
        return self.norm(hidden)


class BaseGCNGRUModel(nn.Module):
    """Base spatio-temporal model with pluggable GNN + temporal encoders."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        gnn_type: str = "gcn",
        temporal_type: str = "gru",
        num_timesteps_in: int = 168,
    ):
        super(BaseGCNGRUModel, self).__init__()

        self.node_num = node_num
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.global_min = global_min
        self.global_max = global_max
        self.num_timesteps_in = int(num_timesteps_in)
        self.norm_params = None
        self.norm_method = "minmax"
        self.norm_min = global_min
        self.norm_max = global_max
        self.norm_mean = None
        self.norm_std = None
        self.use_log = True
        self.log_offset = 1.0

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.register_buffer(
            "hier_level_ids",
            torch.zeros(node_num, dtype=torch.long),
            persistent=False,
        )

        self.graph_config = None
        self.base_adj = None
        self.base_edge_index = None
        self.base_edge_weight = None
        self.adaptive_emb = None
        self.use_adaptive = False
        self.use_dynamic = False
        self.supports_data_driven_graph_sources = False

        self.gnn_type = "gcn"
        self.temporal_type = "gru"
        self.gnn_layer = None
        self.node_encoder = None
        self.temporal_encoder = None
        self.temporal_pos_dropout = nn.Dropout(0.1)
        self.gcn = None
        self.gru = None
        self.set_backbone(gnn_type=gnn_type, temporal_type=temporal_type)

    def set_backbone(self, gnn_type: str, temporal_type: str) -> None:
        gnn_type = str(gnn_type).lower()
        temporal_type = str(temporal_type).lower()
        if gnn_type not in {"none", "gcn", "gatv2", "graphsage", "transformer"}:
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")
        if temporal_type not in {"gru", "transformer", "tcn", "timemixer", "patchtst", "dlinear", "itransformer"}:
            raise ValueError(f"Unsupported temporal_type: {temporal_type}")

        self.gnn_type = gnn_type
        self.temporal_type = temporal_type

        if gnn_type == "none":
            self.node_encoder = nn.Linear(self.input_dim, self.hidden_dim)
            self.gnn_layer = None
        elif gnn_type == "gcn":
            self.node_encoder = None
            self.gnn_layer = GCNConv(in_channels=self.input_dim, out_channels=self.hidden_dim)
        elif gnn_type == "gatv2":
            self.node_encoder = None
            self.gnn_layer = GATv2Conv(in_channels=self.input_dim, out_channels=self.hidden_dim, heads=1)
        elif gnn_type == "graphsage":
            self.node_encoder = None
            self.gnn_layer = SAGEConv(in_channels=self.input_dim, out_channels=self.hidden_dim)
        else:
            self.node_encoder = None
            self.gnn_layer = TransformerConv(in_channels=self.input_dim, out_channels=self.hidden_dim, heads=1)

        if temporal_type == "gru":
            self.temporal_encoder = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif temporal_type == "tcn":
            layers = []
            for layer_idx in range(max(1, self.num_layers)):
                dilation = 2 ** layer_idx
                layers.append(
                    nn.Conv1d(
                        in_channels=self.hidden_dim,
                        out_channels=self.hidden_dim,
                        kernel_size=3,
                        dilation=dilation,
                        padding=dilation,
                    )
                )
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
            self.temporal_encoder = nn.Sequential(*layers)
        elif temporal_type == "timemixer":
            self.temporal_encoder = TimeMixerEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
        elif temporal_type == "patchtst":
            self.temporal_encoder = PatchTSTEncoder(
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
            )
        elif temporal_type == "dlinear":
            self.temporal_encoder = DLinearTemporalEncoder(
                hidden_dim=self.hidden_dim,
                num_timesteps_in=self.num_timesteps_in,
            )
        elif temporal_type == "itransformer":
            self.temporal_encoder = ITransformerTemporalEncoder(
                hidden_dim=self.hidden_dim,
                num_timesteps_in=self.num_timesteps_in,
                num_layers=self.num_layers,
            )
        else:
            nhead = _choose_nhead(self.hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=nhead,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.2,
                activation="gelu",
                batch_first=True,
            )
            self.temporal_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=max(1, self.num_layers),
            )

        # Backward compatibility aliases used elsewhere in this project.
        self.gcn = self.gnn_layer
        self.gru = self.temporal_encoder if temporal_type == "gru" else None

    def set_hierarchy_metadata(
        self,
        sum_matrix,
        middle_levels=None,
        bottom_start_idx: int | None = None,
    ) -> None:
        """Attach actual hierarchy-depth ids, with a legacy coverage fallback."""
        if sum_matrix is None:
            return
        matrix = torch.as_tensor(sum_matrix, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != self.node_num:
            raise ValueError(
                "sum_matrix must be two-dimensional with one row per model node: "
                f"expected {self.node_num} rows, got {tuple(matrix.shape)}."
            )

        if middle_levels is not None and bottom_start_idx is not None:
            bottom_start = int(bottom_start_idx)
            num_bottom = int(matrix.shape[1])
            if bottom_start < 1 or bottom_start + num_bottom != self.node_num:
                raise ValueError(
                    "Explicit hierarchy levels require one top node followed by "
                    "middle nodes and a final contiguous bottom block."
                )
            groups = [[int(index) for index in group] for group in middle_levels]
            flattened = [index for group in groups for index in group]
            expected_middle = list(range(1, bottom_start))
            if sorted(flattened) != expected_middle or len(flattened) != len(set(flattened)):
                raise ValueError(
                    "middle_levels must partition exactly the non-top, non-bottom nodes: "
                    f"expected {expected_middle}, got {groups}."
                )
            level_ids = torch.full(
                (self.node_num,), len(groups) + 1, dtype=torch.long
            )
            level_ids[0] = 0
            for level, group in enumerate(groups, start=1):
                if group:
                    level_ids[torch.as_tensor(group, dtype=torch.long)] = level
            self.hierarchy_level_encoding_version = "explicit_middle_levels_v1"
            self.hier_level_ids = level_ids.to(next(self.parameters()).device)
            return

        # Backward-compatible exploratory path for datasets without explicit
        # middle-level metadata. Formal Applied Energy runs never use it.
        coverage = torch.round(matrix.abs().sum(dim=1)).long().clamp_min(1)
        unique_counts = torch.sort(torch.unique(coverage), descending=True).values
        level_ids = torch.zeros(self.node_num, dtype=torch.long)
        for level, count in enumerate(unique_counts):
            level_ids[coverage == count] = min(level, 15)
        self.hierarchy_level_encoding_version = "row_coverage_legacy_v1"
        self.hier_level_ids = level_ids.to(next(self.parameters()).device)

    def set_norm_params(self, norm_params: dict | None) -> None:
        if not norm_params:
            return
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
            mean = norm_params.get("mean")
            std = norm_params.get("std")
            if mean is None or std is None:
                raise KeyError("normalization params missing mean/std for zscore.")
            self.norm_mean = float(mean)
            self.norm_std = float(std) if float(std) != 0 else 1.0
            self.norm_min = None
            self.norm_max = None
            self.global_min = self.norm_mean
            self.global_max = self.norm_mean + self.norm_std
        else:
            min_val = norm_params.get("min", norm_params.get("global_min"))
            max_val = norm_params.get("max", norm_params.get("global_max"))
            if min_val is None or max_val is None:
                raise KeyError("normalization params missing min/max (or global_min/global_max).")
            self.norm_min = float(min_val)
            self.norm_max = float(max_val)
            self.norm_mean = None
            self.norm_std = None
            self.global_min = self.norm_min
            self.global_max = self.norm_max

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization with finite guards."""
        _require_finite_transform(x, "normalized_value_before_inverse_transform")
        if self.norm_method == "zscore":
            mean = self.norm_mean if self.norm_mean is not None else 0.0
            std = self.norm_std if self.norm_std not in (None, 0.0) else 1.0
            return _require_finite_transform(
                x * std + mean, "denormalized_value"
            )
        return _require_finite_transform(
            x * (self.norm_max - self.norm_min) + self.norm_min,
            "denormalized_value",
        )

    def inverse_log(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse log transform."""
        _require_finite_transform(x, "value_before_inverse_log")
        if not self.use_log:
            return x
        return _require_finite_transform(
            torch.exp(x) - self.log_offset, "original_scale_value"
        )

    def transform_target(self, y: torch.Tensor) -> torch.Tensor:
        """Transform training targets back to the original scale."""
        y_denorm = self.denormalize(y)
        y_delog = self.inverse_log(y_denorm)
        return y_delog

    def prediction_from_normalized(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Map a normalized-log forecast to the original target scale."""
        return self.inverse_log(self.denormalize(pred_norm))

    def set_graph_config(
        self,
        graph_config: dict,
        base_adj=None,
        base_edge_index=None,
        base_edge_weight=None,
    ) -> None:
        policy = graph_config.get("graph_sparsity_policy", FINAL_GRAPH_SOURCE_POLICY)
        if policy != FINAL_GRAPH_SOURCE_POLICY:
            raise ValueError(
                f"Unsupported graph_sparsity_policy={policy!r}; "
                f"the current protocol requires {FINAL_GRAPH_SOURCE_POLICY!r}."
            )
        graph_mode = str(graph_config.get("graph_mode", "H"))
        graph_tokens = set(graph_mode.split("+"))
        uses_data_driven_source = bool(graph_tokens.intersection({"S", "A", "D"}))
        if uses_data_driven_source and not self.supports_data_driven_graph_sources:
            raise ValueError(
                "Data-driven S/A/D graph sources are owned by LAGTCN's independent "
                "source-propagation path; generic adjacency fusion has been removed."
            )
        self.graph_config = {
            "graph_mode": graph_mode,
            "sim_type": graph_config.get("sim_type", "cosine"),
            "adaptive_sim_type": graph_config.get("adaptive_sim_type", graph_config.get("sim_type", "cosine")),
            "dynamic_sim_type": graph_config.get("dynamic_sim_type", graph_config.get("sim_type", "cosine")),
            "adaptive_top_k": graph_config.get("adaptive_top_k"),
            "graph_sparsity_policy": policy,
            "adaptive_emb_dim": graph_config.get("adaptive_emb_dim", 16),
            "dynamic_threshold": graph_config.get("dynamic_threshold"),
            "static_threshold": graph_config.get("static_threshold"),
            "include_self_loops": graph_config.get("include_self_loops", True),
        }
        self.use_adaptive = "A" in graph_tokens
        self.use_dynamic = "D" in graph_tokens

        if base_adj is not None:
            self.base_adj = torch.as_tensor(base_adj, dtype=torch.float32)
        if base_edge_index is not None:
            self.base_edge_index = torch.as_tensor(base_edge_index, dtype=torch.long)
        if base_edge_weight is not None:
            self.base_edge_weight = torch.as_tensor(base_edge_weight, dtype=torch.float32)

        if self.use_adaptive and self.adaptive_emb is None:
            emb_dim = int(self.graph_config["adaptive_emb_dim"])
            device = next(self.parameters()).device
            self.adaptive_emb = nn.Parameter(torch.randn(self.node_num, emb_dim, device=device) * 0.01)

        if self.base_adj is not None and self.base_edge_index is None:
            edge_index, edge_weight = dense_to_sparse(self.base_adj)
            self.base_edge_index = edge_index
            self.base_edge_weight = edge_weight

    def _compute_similarity_matrix(
        self,
        node_repr: torch.Tensor,
        sim_type: str,
        use_abs: bool,
    ) -> torch.Tensor:
        if node_repr.dim() not in (2, 3):
            raise ValueError(
                "node_repr must have shape [N,F] or [B,N,F], "
                f"got {tuple(node_repr.shape)}"
            )
        if sim_type.lower() == "pearson":
            centered = node_repr - node_repr.mean(dim=-1, keepdim=True)
            cov = centered @ centered.transpose(-1, -2)
            var = (centered ** 2).sum(dim=-1, keepdim=True)
            denom = torch.sqrt(var) @ torch.sqrt(var).transpose(-1, -2)
            sim = cov / (denom + node_repr.new_tensor(1e-8))
            if use_abs:
                sim = sim.abs()
            sim = torch.nan_to_num(sim, nan=0.0)
            sim.diagonal(dim1=-2, dim2=-1).fill_(1.0)
            return sim

        norms = torch.norm(node_repr, dim=-1, keepdim=True)
        norms = torch.where(norms == 0, node_repr.new_tensor(1e-8), norms)
        x_norm = node_repr / norms
        sim = x_norm @ x_norm.transpose(-1, -2)
        sim = torch.clamp(sim, -1.0, 1.0)
        if use_abs:
            sim = sim.abs()
        sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=0.0)
        sim.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return sim

    def _compute_adaptive_adj(self) -> torch.Tensor:
        config = self.graph_config
        top_k = config.get("adaptive_top_k")
        if top_k is None:
            raise ValueError(
                f"{FINAL_GRAPH_SOURCE_POLICY} requires adaptive_top_k for graph modes using A."
            )
        sim = self._compute_similarity_matrix(
            self.adaptive_emb,
            config["adaptive_sim_type"],
            use_abs=False,
        )
        return build_topk_similarity_adj_torch(
            sim,
            top_k=int(top_k),
            include_self_loops=config["include_self_loops"],
            use_weights=True,
        )


    def _resolve_graph_edges(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Resolve the one fixed graph used by non-LAGTCN model families."""
        if not self.graph_config or self.base_adj is None or self.base_edge_index is None:
            return edge_index, edge_weight
        edge_index = self.base_edge_index.to(x.device)
        edge_weight = (
            self.base_edge_weight.to(x.device)
            if self.base_edge_weight is not None
            else edge_weight
        )
        return edge_index, edge_weight

    def _apply_gnn(
        self,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.gnn_type == "none":
            return self.node_encoder(x_t)
        if self.gnn_type == "gcn":
            return self.gnn_layer(x_t, edge_index, edge_weight=edge_weight)
        return self.gnn_layer(x_t, edge_index)

    def _expand_edges_for_batch(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        batch_size: int,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if batch_size <= 1:
            return edge_index, edge_weight

        num_edges = edge_index.shape[1]
        offsets = (
            torch.arange(batch_size, device=edge_index.device, dtype=edge_index.dtype) * num_nodes
        ).repeat_interleave(num_edges)
        batched_edge_index = edge_index.repeat(1, batch_size) + offsets.unsqueeze(0)
        batched_edge_weight = edge_weight.repeat(batch_size) if edge_weight is not None else None
        return batched_edge_index, batched_edge_weight

    def _build_sinusoidal_positional_encoding(self, length: int, dim: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(1, length, dim, device=device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _run_temporal_encoder(self, spatial_features: torch.Tensor, ssz: int, n_nodes: int) -> torch.Tensor:
        if self.temporal_type == "gru":
            h0 = torch.zeros(self.num_layers, ssz * n_nodes, self.hidden_dim, device=spatial_features.device)
            gru_out, _ = self.temporal_encoder(spatial_features, h0)
            return gru_out[:, -1, :]

        if self.temporal_type == "tcn":
            # [S*N, T, H] -> [S*N, H, T] for Conv1d
            tcn_in = spatial_features.transpose(1, 2)
            tcn_out = self.temporal_encoder(tcn_in)
            tcn_out = tcn_out.transpose(1, 2)
            return tcn_out[:, -1, :]

        if self.temporal_type in ("timemixer", "patchtst", "dlinear", "itransformer"):
            return self.temporal_encoder(spatial_features)

        seq_len = spatial_features.shape[1]
        pos_enc = self._build_sinusoidal_positional_encoding(
            length=seq_len,
            dim=self.hidden_dim,
            device=spatial_features.device,
        )
        tf_in = self.temporal_pos_dropout(spatial_features + pos_enc)
        tf_out = self.temporal_encoder(tf_in)
        return tf_out[:, -1, :]

    def set_st_mode(self, st_mode: str = "sequential") -> None:
        """Set spatio-temporal interaction mode.

        Args:
            st_mode: 'sequential' (default, GNN per timestep then temporal)
                     'alternating' (GNN->Temporal->GNN->Temporal stacked blocks)
                     or 'hier_fusion' (hierarchy-aware temporal + node-token fusion)
        """
        self.st_mode = st_mode.lower()
        if self.st_mode in {"hierarchy_fusion", "hierarchical_fusion"}:
            self.st_mode = "hier_fusion"
        device = next(self.parameters()).device
        if self.st_mode == "alternating" and not hasattr(self, "_alt_gnn_layer"):
            # Create a second GNN layer for the alternating block
            self._alt_gnn_layer = type(self.gnn_layer)(
                in_channels=self.hidden_dim, out_channels=self.hidden_dim,
                **({"heads": 1} if hasattr(self.gnn_layer, "heads") else {}),
            ) if self.gnn_layer is not None else None
            self._alt_layer_norm = nn.LayerNorm(self.hidden_dim)
            if self._alt_gnn_layer is not None:
                self._alt_gnn_layer.to(device)
            self._alt_layer_norm.to(device)
        if self.st_mode == "hier_fusion" and not hasattr(self, "_hier_node_emb"):
            nhead = _choose_nhead(self.hidden_dim)
            self._hier_node_emb = nn.Embedding(self.node_num, self.hidden_dim)
            self._hier_level_emb = nn.Embedding(16, self.hidden_dim)
            self._hier_temporal_score = nn.Linear(self.hidden_dim, 1)
            self._hier_temporal_fusion = nn.Linear(self.hidden_dim * 4, self.hidden_dim)
            self._hier_node_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=nhead,
                dropout=0.1,
                batch_first=True,
            )
            self._hier_graph_gate = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
            self._hier_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )
            self._hier_input_norm = nn.LayerNorm(self.hidden_dim)
            self._hier_temporal_norm = nn.LayerNorm(self.hidden_dim)
            self._hier_node_norm = nn.LayerNorm(self.hidden_dim)
            self._hier_graph_norm = nn.LayerNorm(self.hidden_dim)
            self._hier_output_norm = nn.LayerNorm(self.hidden_dim)
            for module in (
                self._hier_node_emb,
                self._hier_level_emb,
                self._hier_temporal_score,
                self._hier_temporal_fusion,
                self._hier_node_attn,
                self._hier_graph_gate,
                self._hier_ffn,
                self._hier_input_norm,
                self._hier_temporal_norm,
                self._hier_node_norm,
                self._hier_graph_norm,
                self._hier_output_norm,
            ):
                module.to(device)

    def _forward_sequential(
        self,
        x: torch.Tensor,
        edge_index_batch: torch.Tensor,
        edge_weight_batch: torch.Tensor | None,
        ssz: int,
        n_nodes: int,
        feat_dim: int,
        t_in: int,
    ) -> torch.Tensor:
        """Original sequential mode: GNN per timestep -> Temporal encoder."""
        spatial_features = []
        for t in range(t_in):
            x_t = x[:, :, :, t].reshape(ssz * n_nodes, feat_dim)
            gnn_out = self._apply_gnn(x_t, edge_index_batch, edge_weight_batch)
            gnn_out = self.layer_norm(gnn_out)
            gnn_out = self.dropout(gnn_out)
            spatial_features.append(gnn_out)

        spatial_features = torch.stack(spatial_features, dim=1)  # [S*N, T_in, H]
        return self._run_temporal_encoder(spatial_features, ssz=ssz, n_nodes=n_nodes)

    def _forward_alternating(
        self,
        x: torch.Tensor,
        edge_index_batch: torch.Tensor,
        edge_weight_batch: torch.Tensor | None,
        ssz: int,
        n_nodes: int,
        feat_dim: int,
        t_in: int,
    ) -> torch.Tensor:
        """Alternating mode: GNN -> Temporal -> GNN -> Temporal with residual."""
        # Block 1: First GNN pass per timestep
        spatial_features = []
        for t in range(t_in):
            x_t = x[:, :, :, t].reshape(ssz * n_nodes, feat_dim)
            gnn_out = self._apply_gnn(x_t, edge_index_batch, edge_weight_batch)
            gnn_out = self.layer_norm(gnn_out)
            gnn_out = self.dropout(gnn_out)
            spatial_features.append(gnn_out)
        spatial_features = torch.stack(spatial_features, dim=1)  # [S*N, T, H]

        # Block 1: First temporal pass
        temporal_out = self._run_temporal_encoder(spatial_features, ssz=ssz, n_nodes=n_nodes)
        # Expand back to sequence: repeat last hidden state for each timestep
        temporal_seq = temporal_out.unsqueeze(1).expand(-1, t_in, -1)  # [S*N, T, H]
        residual = spatial_features + temporal_seq  # residual connection

        # Block 2: Second GNN pass on temporally-enriched features
        if self._alt_gnn_layer is not None:
            refined_features = []
            for t in range(t_in):
                h_t = residual[:, t, :]  # [S*N, H]
                if self.gnn_type == "gcn":
                    gnn_out2 = self._alt_gnn_layer(h_t, edge_index_batch, edge_weight=edge_weight_batch)
                else:
                    gnn_out2 = self._alt_gnn_layer(h_t, edge_index_batch)
                gnn_out2 = self._alt_layer_norm(gnn_out2)
                gnn_out2 = self.dropout(gnn_out2)
                refined_features.append(gnn_out2)
            refined_features = torch.stack(refined_features, dim=1)  # [S*N, T, H]
            refined_features = refined_features + residual  # residual connection
        else:
            refined_features = residual

        # Block 2: Second temporal pass
        return self._run_temporal_encoder(refined_features, ssz=ssz, n_nodes=n_nodes)

    def _dense_normalized_adj(
        self,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        n_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        adj = torch.zeros(n_nodes, n_nodes, device=device, dtype=dtype)
        if edge_index is not None and edge_index.numel() > 0:
            src = edge_index[0].to(device=device, dtype=torch.long)
            dst = edge_index[1].to(device=device, dtype=torch.long)
            keep = (src >= 0) & (src < n_nodes) & (dst >= 0) & (dst < n_nodes)
            src = src[keep]
            dst = dst[keep]
            if edge_weight is None:
                values = torch.ones_like(src, dtype=dtype, device=device)
            else:
                values = edge_weight.to(device=device, dtype=dtype)[keep]
            adj.index_put_((src, dst), values, accumulate=True)
        adj.fill_diagonal_(1.0)
        row_sum = adj.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return adj / row_sum

    def _hier_level_embedding(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        if self.hier_level_ids.numel() == n_nodes:
            level_ids = self.hier_level_ids.to(device=device)
        else:
            level_ids = torch.zeros(n_nodes, dtype=torch.long, device=device)
        level_ids = level_ids.clamp(min=0, max=self._hier_level_emb.num_embeddings - 1)
        return self._hier_level_emb(level_ids)

    def _forward_hier_fusion(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        edge_index_batch: torch.Tensor,
        edge_weight_batch: torch.Tensor | None,
        ssz: int,
        n_nodes: int,
        feat_dim: int,
        t_in: int,
    ) -> torch.Tensor:
        """Hierarchy-aware fusion mode: temporal summary + node-token graph fusion."""
        spatial_features = []
        for t in range(t_in):
            x_t = x[:, :, :, t].reshape(ssz * n_nodes, feat_dim)
            gnn_out = self._apply_gnn(x_t, edge_index_batch, edge_weight_batch)
            gnn_out = self.layer_norm(gnn_out)
            gnn_out = self.dropout(gnn_out)
            spatial_features.append(gnn_out)

        sequence = torch.stack(spatial_features, dim=1)  # [S*N, T, H]
        tokens = sequence.view(ssz, n_nodes, t_in, self.hidden_dim)

        node_ids = torch.arange(n_nodes, device=x.device)
        node_emb = self._hier_node_emb(node_ids).view(1, n_nodes, 1, self.hidden_dim)
        level_emb = self._hier_level_embedding(n_nodes, x.device).view(1, n_nodes, 1, self.hidden_dim)
        time_emb = self._build_sinusoidal_positional_encoding(
            length=t_in,
            dim=self.hidden_dim,
            device=x.device,
        ).view(1, 1, t_in, self.hidden_dim)
        tokens = self._hier_input_norm(tokens + node_emb + level_emb + time_emb)
        sequence = tokens.reshape(ssz * n_nodes, t_in, self.hidden_dim)

        backbone_summary = self._run_temporal_encoder(sequence, ssz=ssz, n_nodes=n_nodes)
        backbone_summary = backbone_summary.view(ssz, n_nodes, self.hidden_dim)

        score = self._hier_temporal_score(tokens).squeeze(-1)  # [S, N, T]
        weights = torch.softmax(score, dim=-1)
        attentive_summary = torch.sum(tokens * weights.unsqueeze(-1), dim=2)
        mean_summary = tokens.mean(dim=2)
        recent_window = min(t_in, 24)
        recent_summary = tokens[:, :, -recent_window:, :].mean(dim=2)
        temporal_summary = self._hier_temporal_fusion(
            torch.cat(
                [backbone_summary, attentive_summary, mean_summary, recent_summary],
                dim=-1,
            )
        )
        temporal_summary = self._hier_temporal_norm(backbone_summary + self.dropout(temporal_summary))

        node_attn, _ = self._hier_node_attn(
            temporal_summary,
            temporal_summary,
            temporal_summary,
            need_weights=False,
        )
        node_summary = self._hier_node_norm(temporal_summary + self.dropout(node_attn))

        adj = self._dense_normalized_adj(edge_index, edge_weight, n_nodes, x.device, node_summary.dtype)
        graph_context = torch.einsum("ij,bjh->bih", adj, node_summary)
        graph_gate = torch.sigmoid(self._hier_graph_gate(torch.cat([node_summary, graph_context], dim=-1)))
        fused = self._hier_graph_norm(node_summary + graph_gate * graph_context)

        fused_ffn = self._hier_ffn(fused)
        fused = self._hier_output_norm(fused + self.dropout(fused_ffn))
        return fused.reshape(ssz * n_nodes, self.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate base predictions before reconciliation."""
        # x shape: [S, N, Fe, T_in]
        ssz, n_nodes, feat_dim, t_in = x.shape

        edge_index_use, edge_weight_use = self._resolve_graph_edges(x, edge_index, edge_weight)
        edge_index_batch, edge_weight_batch = self._expand_edges_for_batch(
            edge_index_use,
            edge_weight_use,
            batch_size=ssz,
            num_nodes=n_nodes,
        )

        st_mode = getattr(self, "st_mode", "sequential")
        if st_mode == "hier_fusion" and hasattr(self, "_hier_node_emb"):
            temporal_last = self._forward_hier_fusion(
                x,
                edge_index_use,
                edge_weight_use,
                edge_index_batch,
                edge_weight_batch,
                ssz,
                n_nodes,
                feat_dim,
                t_in,
            )
        elif st_mode == "alternating" and hasattr(self, "_alt_gnn_layer"):
            temporal_last = self._forward_alternating(
                x, edge_index_batch, edge_weight_batch, ssz, n_nodes, feat_dim, t_in
            )
        else:
            temporal_last = self._forward_sequential(
                x, edge_index_batch, edge_weight_batch, ssz, n_nodes, feat_dim, t_in
            )

        final_hidden = temporal_last.view(ssz, n_nodes, -1)
        base_predictions = self.projection(final_hidden)
        return final_hidden, base_predictions
