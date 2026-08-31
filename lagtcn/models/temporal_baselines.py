import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _choose_nhead(hidden_dim: int) -> int:
    for nhead in (16, 8, 4, 2, 1):
        if hidden_dim % nhead == 0:
            return nhead
    return 1


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


class TemporalBaselineBase(nn.Module):
    """Shared normalization/log utilities for temporal-only baselines."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
    ):
        super().__init__()
        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.global_min = float(global_min)
        self.global_max = float(global_max)

        self.norm_params = None
        self.norm_method = "minmax"
        self.norm_min = self.global_min
        self.norm_max = self.global_max
        self.norm_mean = None
        self.norm_std = None
        self.use_log = True
        self.log_offset = 1.0

        self.graph_config = {}

    def set_graph_config(
        self,
        graph_config: dict,
        base_adj=None,
        base_edge_index=None,
        base_edge_weight=None,
    ) -> None:
        # Temporal-only baseline does not use graph edges, but we keep this
        # method for compatibility with the existing training pipeline.
        self.graph_config = dict(graph_config or {})

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
        _require_finite_transform(x, "value_before_inverse_log")
        if not self.use_log:
            return x
        return _require_finite_transform(
            torch.exp(x) - self.log_offset, "original_scale_value"
        )

    def transform_target(self, y: torch.Tensor) -> torch.Tensor:
        y_denorm = self.denormalize(y)
        y_delog = self.inverse_log(y_denorm)
        return y_delog

    def prediction_from_normalized(self, pred_norm: torch.Tensor) -> torch.Tensor:
        """Map a normalized-log forecast to the original target scale."""
        return self.inverse_log(self.denormalize(pred_norm))

    @staticmethod
    def _extract_target_series(x: torch.Tensor) -> torch.Tensor:
        # x: [S, N, F, T], use target channel only (F index 0).
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [S,N,F,T], got {tuple(x.shape)}")
        return x[:, :, 0, :]


class DLinearBaseline(TemporalBaselineBase):
    """Original shared DLinear architecture adapted to the project I/O."""

    architecture_version = "dlinear_official_shared_v1"

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        num_timesteps_in: int,
        moving_avg_window: int = 25,
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        kernel = max(1, int(moving_avg_window))
        if kernel % 2 == 0:
            kernel += 1
        self.moving_avg_window = kernel

        # This is the shared (individual=False) DLinear formulation. Keep
        # PyTorch's default Linear initialization, matching the released code;
        # the moving-average initialization shown in that repository is
        # commented out and is not part of the forecasting model.
        self.trend_linear = nn.Linear(self.num_timesteps_in, self.output_dim)
        self.seasonal_linear = nn.Linear(self.num_timesteps_in, self.output_dim)

    def _moving_average(self, series: torch.Tensor) -> torch.Tensor:
        pad = self.moving_avg_window // 2
        x = series.unsqueeze(1)
        x_pad = F.pad(x, (pad, pad), mode="replicate")
        return F.avg_pool1d(
            x_pad, kernel_size=self.moving_avg_window, stride=1
        ).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        target_series = self._extract_target_series(x)
        ssz, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"DLinear expected T_in={self.num_timesteps_in}, got {t_in}")

        flat = target_series.reshape(ssz * n_nodes, t_in)
        trend = self._moving_average(flat)
        seasonal = flat - trend
        pred_norm = self.trend_linear(trend) + self.seasonal_linear(seasonal)
        pred_norm = pred_norm.view(ssz, n_nodes, self.output_dim)
        return self.inverse_log(self.denormalize(pred_norm))


class _RevIN(nn.Module):
    """Reversible instance normalization used by supervised PatchTST."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(self.num_features))
            self.bias = nn.Parameter(torch.zeros(self.num_features))

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.dim() != 3 or x.shape[1] != self.num_features:
            raise ValueError(
                f"RevIN expected [B,{self.num_features},T], got {tuple(x.shape)}"
            )
        if self.subtract_last:
            center = x[:, :, -1:].detach()
        else:
            center = x.mean(dim=-1, keepdim=True).detach()
        stdev = torch.sqrt(
            x.var(dim=-1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        normalized = (x - center) / stdev
        if self.affine:
            normalized = (
                normalized * self.weight.view(1, -1, 1)
                + self.bias.view(1, -1, 1)
            )
        return normalized, center, stdev

    def denormalize(
        self,
        x: torch.Tensor,
        center: torch.Tensor,
        stdev: torch.Tensor,
    ) -> torch.Tensor:
        if self.affine:
            x = (
                x - self.bias.view(1, -1, 1)
            ) / (self.weight.view(1, -1, 1) + self.eps * self.eps)
        return x * stdev + center


class _PatchTSTAttention(nn.Module):
    """PatchTST multi-head attention with optional residual attention scores."""

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        attn_dropout: float,
        proj_dropout: float,
        residual_attention: bool,
    ):
        super().__init__()
        if model_dim % n_heads:
            raise ValueError(f"model_dim={model_dim} must be divisible by n_heads={n_heads}")
        self.n_heads = int(n_heads)
        self.head_dim = model_dim // self.n_heads
        self.residual_attention = bool(residual_attention)
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim), nn.Dropout(proj_dropout)
        )
        self.register_buffer(
            "scale", torch.tensor(self.head_dim ** -0.5), persistent=False
        )

    def forward(
        self,
        x: torch.Tensor,
        previous_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, model_dim = x.shape
        q = self.q_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.n_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.v_proj(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k) * self.scale.to(dtype=x.dtype)
        if self.residual_attention and previous_scores is not None:
            scores = scores + previous_scores
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).contiguous().view(batch, length, model_dim)
        return self.out_proj(output), scores


class _PatchTSTEncoderLayer(nn.Module):
    """Released PatchTST post-norm encoder layer (BatchNorm by default)."""

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
        attn_dropout: float,
        residual_attention: bool = True,
    ):
        super().__init__()
        self.attention = _PatchTSTAttention(
            model_dim,
            n_heads,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            residual_attention=residual_attention,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.BatchNorm1d(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.BatchNorm1d(model_dim)

    @staticmethod
    def _batch_norm(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        previous_scores: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(x, previous_scores)
        x = self._batch_norm(self.attn_norm, x + self.attn_dropout(attended))
        fed = self.feedforward(x)
        x = self._batch_norm(self.ffn_norm, x + self.ffn_dropout(fed))
        return x, scores


class _PatchTSTEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
        attn_dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            _PatchTSTEncoderLayer(
                model_dim=model_dim,
                n_heads=n_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                residual_attention=True,
            )
            for _ in range(max(1, int(num_layers)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = None
        for layer in self.layers:
            x, scores = layer(x, scores)
        return x


class PatchTSTBaseline(TemporalBaselineBase):
    """Faithful supervised PatchTST adapted only to project tensor I/O."""

    architecture_version = "patchtst_supervised_official_v1"

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        num_timesteps_in: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        dropout: float = 0.1,
        attn_dropout: float = 0.0,
        head_dropout: float = 0.0,
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        self.patch_len = max(1, min(int(patch_len), self.num_timesteps_in))
        self.patch_stride = max(1, int(patch_stride))
        self.model_dim = max(16, int(hidden_dim))
        self.n_heads = _choose_nhead(self.model_dim)
        self.patch_num = int(
            (self.num_timesteps_in - self.patch_len) / self.patch_stride + 1
        ) + 1

        self.revin = _RevIN(
            num_features=node_num,
            affine=False,
            subtract_last=False,
        )
        self.end_padding = nn.ReplicationPad1d((0, self.patch_stride))
        self.patch_projection = nn.Linear(self.patch_len, self.model_dim)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.patch_num, self.model_dim)
        )
        nn.init.uniform_(self.position_embedding, -0.02, 0.02)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder = _PatchTSTEncoder(
            num_layers=max(1, int(num_layers)),
            model_dim=self.model_dim,
            n_heads=self.n_heads,
            feedforward_dim=2 * self.model_dim,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        # Official shared Flatten Head (individual=False).
        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(self.model_dim * self.patch_num, self.output_dim)
        self.head_dropout = nn.Dropout(head_dropout)

    def forward_normalized(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        target_series = self._extract_target_series(x)
        batch, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"PatchTST expected T_in={self.num_timesteps_in}, got {t_in}")

        normalized, center, stdev = self.revin.normalize(target_series)
        padded = self.end_padding(normalized)
        patches = padded.unfold(
            dimension=-1, size=self.patch_len, step=self.patch_stride
        )
        if patches.shape[2] != self.patch_num:
            raise RuntimeError(
                f"PatchTST produced {patches.shape[2]} patches, expected {self.patch_num}."
            )
        tokens = self.patch_projection(
            patches.reshape(batch * n_nodes, self.patch_num, self.patch_len)
        )
        tokens = self.embedding_dropout(tokens + self.position_embedding)
        encoded = self.encoder(tokens)
        encoded = encoded.view(batch, n_nodes, self.patch_num, self.model_dim)
        encoded = encoded.permute(0, 1, 3, 2)
        pred_norm = self.head_dropout(self.head(self.flatten(encoded)))
        pred_norm = self.revin.denormalize(pred_norm, center, stdev)
        return _require_finite_transform(pred_norm, "normalized_prediction")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_norm = self.forward_normalized(x, edge_index, edge_weight)
        return self.prediction_from_normalized(pred_norm)

class _NHiTSIdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = int(backcast_size)
        self.forecast_size = int(forecast_size)

    def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :].unsqueeze(1)
        forecast = F.interpolate(
            knots, size=self.forecast_size, mode="linear", align_corners=False
        ).squeeze(1)
        return backcast, forecast


class _NHiTSBlock(nn.Module):
    """Official identity-basis N-HiTS block without exogenous regressors."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        hidden_dim: int,
        mlp_depth: int,
        pooling_size: int,
        frequency_downsample: int,
        dropout: float,
    ):
        super().__init__()
        self.input_len = int(input_len)
        self.output_len = int(output_len)
        self.pooling_size = int(pooling_size)
        self.frequency_downsample = int(frequency_downsample)
        pooled_len = int(math.ceil(self.input_len / self.pooling_size))
        forecast_knots = max(self.output_len // self.frequency_downsample, 1)
        self.pooling_layer = nn.MaxPool1d(
            kernel_size=self.pooling_size,
            stride=self.pooling_size,
            ceil_mode=True,
        )

        layers: list[nn.Module] = [nn.Linear(pooled_len, hidden_dim)]
        for _ in range(max(1, int(mlp_depth))):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, self.input_len + forecast_knots))
        self.layers = nn.Sequential(*layers)
        self.basis = _NHiTSIdentityBasis(self.input_len, self.output_len)

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.pooling_layer(residual.unsqueeze(1)).squeeze(1)
        theta = self.layers(pooled)
        return self.basis(theta)


class NHiTSBaseline(TemporalBaselineBase):
    """Original N-HiTS identity stacks adapted to node-wise project I/O."""

    architecture_version = "nhits_identity_stacks_official_v1"

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        num_timesteps_in: int,
        n_blocks: tuple[int, int, int] = (1, 1, 1),
        pooling_sizes: tuple[int, int, int] = (2, 2, 1),
        frequency_downsamples: tuple[int, int, int] = (4, 2, 1),
        dropout: float = 0.0,
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
        )
        if not (len(n_blocks) == len(pooling_sizes) == len(frequency_downsamples) == 3):
            raise ValueError("N-HiTS requires three aligned stack specifications.")
        self.num_timesteps_in = int(num_timesteps_in)
        block_hidden = max(16, int(hidden_dim))
        blocks = []
        for stack_idx in range(3):
            for _ in range(int(n_blocks[stack_idx])):
                blocks.append(
                    _NHiTSBlock(
                        input_len=self.num_timesteps_in,
                        output_len=self.output_dim,
                        hidden_dim=block_hidden,
                        mlp_depth=max(1, int(num_layers)),
                        pooling_size=int(pooling_sizes[stack_idx]),
                        frequency_downsample=int(frequency_downsamples[stack_idx]),
                        dropout=float(dropout),
                    )
                )
        self.blocks = nn.ModuleList(blocks)
        self.stack_n_blocks = tuple(int(v) for v in n_blocks)
        self.stack_pooling_sizes = tuple(int(v) for v in pooling_sizes)
        self.stack_frequency_downsamples = tuple(int(v) for v in frequency_downsamples)

    def forward_normalized(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        target_series = self._extract_target_series(x)
        batch, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"N-HiTS expected T_in={self.num_timesteps_in}, got {t_in}")

        flat = target_series.reshape(batch * n_nodes, t_in)
        residuals = flat.flip(dims=(-1,))
        forecast = flat[:, -1:].expand(-1, self.output_dim).clone()
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        pred_norm = forecast.view(batch, n_nodes, self.output_dim)
        return _require_finite_transform(pred_norm, "normalized_prediction")

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_norm = self.forward_normalized(x, edge_index, edge_weight)
        return self.prediction_from_normalized(pred_norm)


class _ITransformerEncoderLayer(nn.Module):
    """Encoder layer used by the released iTransformer forecasting model."""

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.conv1 = nn.Conv1d(model_dim, feedforward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(feedforward_dim, model_dim, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attention(x, x, x, need_weights=False)
        x = x + self.dropout(attended)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(1, 2))))
        y = self.dropout(self.conv2(y).transpose(1, 2))
        return self.norm2(x + y)


class ITransformerBaseline(TemporalBaselineBase):
    """Faithful iTransformer variate-token forecaster with instance normalization."""

    architecture_version = "itransformer_official_finalnorm_v2"

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        num_timesteps_in: int,
        dropout: float = 0.1,
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        self.model_dim = max(16, int(hidden_dim))
        self.n_heads = next(
            nhead for nhead in (8, 4, 2, 1) if self.model_dim % nhead == 0
        )

        # DataEmbedding_inverted without time covariates: each node's complete
        # input history is one variate token.
        self.token_projection = nn.Linear(self.num_timesteps_in, self.model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList(
            _ITransformerEncoderLayer(
                model_dim=self.model_dim,
                n_heads=self.n_heads,
                feedforward_dim=4 * self.model_dim,
                dropout=dropout,
            )
            for _ in range(max(1, int(num_layers)))
        )
        self.encoder_norm = nn.LayerNorm(self.model_dim)
        self.projector = nn.Linear(self.model_dim, self.output_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight
        target_series = self._extract_target_series(x)
        batch, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(
                f"iTransformer expected T_in={self.num_timesteps_in}, got {t_in}"
            )

        means = target_series.mean(dim=-1, keepdim=True).detach()
        centered = target_series - means
        stdev = torch.sqrt(
            centered.var(dim=-1, keepdim=True, unbiased=False) + 1e-5
        )
        normalized = centered / stdev

        encoded = self.embedding_dropout(self.token_projection(normalized))
        for layer in self.encoder_layers:
            encoded = layer(encoded)
        encoded = self.encoder_norm(encoded)
        pred_norm = self.projector(encoded).view(batch, n_nodes, self.output_dim)
        pred_norm = pred_norm * stdev + means
        return self.inverse_log(self.denormalize(pred_norm))
