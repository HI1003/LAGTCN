"""Implementation of the LAGTCN forecasting model.

The public class contains the transformations, graph-source construction,
level awareness, patch transformer, graph-temporal co-evolution, and residual
decoder used by LAGTCN.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .graphs import (
    FINAL_GRAPH_SOURCE_POLICY,
    INFORMATIVE_GRAPH_SOURCES,
    build_threshold_similarity_adj_torch,
    build_topk_similarity_adj_torch,
    graph_sources,
    normalize_graph_mode,
)


def _choose_nhead(model_dim: int) -> int:
    for nhead in (8, 4, 2, 1):
        if model_dim % nhead == 0:
            return nhead
    return 1


def _row_normalize(adjacency: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    return adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(epsilon)


def _require_finite(value: torch.Tensor, name: str) -> torch.Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} contains NaN or Inf values")
    return value


class _ForecastTransformBase(nn.Module):
    """Normalization, hierarchy, and graph helpers required by LAGTCN."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
    ) -> None:
        super().__init__()
        self.node_num = int(node_num)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)

        self.global_min = float(global_min)
        self.global_max = float(global_max)
        self.norm_method = "minmax"
        self.norm_min = self.global_min
        self.norm_max = self.global_max
        self.norm_mean: float | None = None
        self.norm_std: float | None = None
        self.use_log = True
        self.log_offset = 1.0

        self.register_buffer(
            "hier_level_ids",
            torch.zeros(self.node_num, dtype=torch.long),
            persistent=False,
        )
        self.graph_config: dict | None = None
        self.adaptive_emb: nn.Parameter | None = None
        self.use_adaptive = False
        self.use_dynamic = False

    def set_norm_params(self, params: dict | None) -> None:
        """Attach the normalization metadata created during preprocessing."""
        if not params:
            return
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
            if "mean" not in params or "std" not in params:
                raise KeyError("z-score normalization requires mean and std")
            self.norm_mean = float(params["mean"])
            self.norm_std = float(params["std"]) or 1.0
            self.norm_min = None
            self.norm_max = None
        else:
            minimum = params.get("min", params.get("global_min"))
            maximum = params.get("max", params.get("global_max"))
            if minimum is None or maximum is None:
                raise KeyError("min-max normalization requires min/max values")
            self.norm_min = float(minimum)
            self.norm_max = float(maximum)
            self.norm_mean = None
            self.norm_std = None

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        _require_finite(value, "normalized value")
        if self.norm_method == "zscore":
            result = value * float(self.norm_std) + float(self.norm_mean)
        else:
            result = value * (float(self.norm_max) - float(self.norm_min)) + float(
                self.norm_min
            )
        return _require_finite(result, "denormalized value")

    def inverse_log(self, value: torch.Tensor) -> torch.Tensor:
        if not self.use_log:
            return value
        return _require_finite(torch.exp(value) - self.log_offset, "original-scale value")

    def transform_target(self, target: torch.Tensor) -> torch.Tensor:
        """Map normalized training targets to the original load scale."""
        return self.inverse_log(self.denormalize(target))

    def prediction_from_normalized(self, prediction: torch.Tensor) -> torch.Tensor:
        return self.transform_target(prediction)

    def set_hierarchy_metadata(
        self,
        sum_matrix,
        middle_levels: list[list[int]] | None = None,
        bottom_start_idx: int | None = None,
    ) -> None:
        """Create a hierarchy-depth id for every node."""
        matrix = torch.as_tensor(sum_matrix, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != self.node_num:
            raise ValueError(
                f"sum_matrix must have {self.node_num} rows, got {tuple(matrix.shape)}"
            )

        if middle_levels is not None and bottom_start_idx is not None:
            bottom_start = int(bottom_start_idx)
            groups = [[int(index) for index in group] for group in middle_levels]
            flattened = [index for group in groups for index in group]
            expected = list(range(1, bottom_start))
            if sorted(flattened) != expected or len(flattened) != len(set(flattened)):
                raise ValueError("middle_levels must partition all middle-level nodes")
            level_ids = torch.full((self.node_num,), len(groups) + 1, dtype=torch.long)
            level_ids[0] = 0
            for level, group in enumerate(groups, start=1):
                if group:
                    level_ids[torch.as_tensor(group, dtype=torch.long)] = level
            self.hierarchy_level_encoding_version = "explicit_middle_levels_v1"
        else:
            coverage = torch.round(matrix.abs().sum(dim=1)).long().clamp_min(1)
            unique_counts = torch.sort(torch.unique(coverage), descending=True).values
            level_ids = torch.zeros(self.node_num, dtype=torch.long)
            for level, count in enumerate(unique_counts):
                level_ids[coverage == count] = min(level, 15)
            self.hierarchy_level_encoding_version = "row_coverage_v1"
        self.hier_level_ids = level_ids.to(next(self.parameters()).device)

    def set_graph_config(self, graph_config: dict) -> None:
        mode = normalize_graph_mode(graph_config.get("graph_mode", "H"))
        policy = graph_config.get("graph_sparsity_policy", FINAL_GRAPH_SOURCE_POLICY)
        if policy != FINAL_GRAPH_SOURCE_POLICY:
            raise ValueError(
                f"graph_sparsity_policy must be {FINAL_GRAPH_SOURCE_POLICY!r}"
            )
        components = set(mode.split("+"))
        self.graph_config = {
            "graph_mode": mode,
            "sim_type": str(graph_config.get("sim_type", "cosine")).lower(),
            "adaptive_sim_type": str(
                graph_config.get("adaptive_sim_type", graph_config.get("sim_type", "cosine"))
            ).lower(),
            "dynamic_sim_type": str(
                graph_config.get("dynamic_sim_type", graph_config.get("sim_type", "cosine"))
            ).lower(),
            "adaptive_top_k": graph_config.get("adaptive_top_k"),
            "dynamic_threshold": graph_config.get("dynamic_threshold"),
            "static_threshold": graph_config.get("static_threshold"),
            "include_self_loops": bool(graph_config.get("include_self_loops", True)),
        }
        self.use_adaptive = "A" in components
        self.use_dynamic = "D" in components
        if self.use_adaptive and self.adaptive_emb is None:
            embedding_dim = int(graph_config.get("adaptive_emb_dim", 16))
            device = next(self.parameters()).device
            self.adaptive_emb = nn.Parameter(
                torch.randn(self.node_num, embedding_dim, device=device) * 0.01
            )

    @staticmethod
    def _similarity(node_repr: torch.Tensor, similarity_type: str, use_abs: bool) -> torch.Tensor:
        if node_repr.dim() not in (2, 3):
            raise ValueError("node representations must have shape [N,F] or [B,N,F]")
        if similarity_type == "pearson":
            centered = node_repr - node_repr.mean(dim=-1, keepdim=True)
            covariance = centered @ centered.transpose(-1, -2)
            variance = (centered**2).sum(dim=-1, keepdim=True)
            denominator = torch.sqrt(variance) @ torch.sqrt(variance).transpose(-1, -2)
            similarity = covariance / (denominator + 1e-8)
        elif similarity_type == "cosine":
            normalized = node_repr / torch.norm(
                node_repr, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            similarity = normalized @ normalized.transpose(-1, -2)
        else:
            raise ValueError("similarity type must be 'cosine' or 'pearson'")
        if use_abs:
            similarity = similarity.abs()
        similarity = torch.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
        similarity.diagonal(dim1=-2, dim2=-1).fill_(1.0)
        return similarity

    def _compute_adaptive_adj(self) -> torch.Tensor:
        if self.graph_config is None or self.adaptive_emb is None:
            raise RuntimeError("adaptive graph source is not configured")
        top_k = self.graph_config.get("adaptive_top_k")
        if top_k is None:
            raise ValueError("graph mode A requires adaptive_top_k")
        similarity = self._similarity(
            self.adaptive_emb,
            self.graph_config["adaptive_sim_type"],
            use_abs=False,
        )
        return build_topk_similarity_adj_torch(
            similarity,
            top_k=int(top_k),
            include_self_loops=self.graph_config["include_self_loops"],
        )

    @staticmethod
    def _extract_target_series(features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4:
            raise ValueError(
                f"features must have shape [batch,nodes,features,time], got {tuple(features.shape)}"
            )
        return features[:, :, 0, :]


class _LAGTCNBlock(nn.Module):
    """One level-aware graph-temporal co-evolution block."""

    def __init__(
        self,
        model_dim: int,
        nhead: int,
        hop_order: int,
        dropout: float,
        uniform_source_fusion: bool,
        sequential_no_coevolution: bool,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.hop_order = max(1, int(hop_order))
        self.uniform_source_fusion = bool(uniform_source_fusion)
        self.sequential_no_coevolution = bool(sequential_no_coevolution)
        self.temporal_encoder = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.source_logits = nn.Parameter(torch.zeros(len(INFORMATIVE_GRAPH_SOURCES)))
        self.graph_mix = nn.Linear(
            self.hop_order * self.model_dim,
            self.model_dim,
            bias=False,
        )
        self.source_norm = nn.LayerNorm(self.model_dim)
        self.source_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_dropout = nn.Dropout(dropout)
        if self.sequential_no_coevolution:
            self.fusion_gate = None
            self.fusion = None
        else:
            self.fusion_gate = nn.Linear(self.model_dim * 4, self.model_dim)
            self.fusion = nn.Sequential(
                nn.Linear(self.model_dim * 3, self.model_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.model_dim * 4, self.model_dim),
            )

    def temporal_update(self, patch_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, patch_count, dimension = patch_tokens.shape
        tokens = patch_tokens + state.unsqueeze(2)
        encoded = self.temporal_encoder(
            tokens.reshape(batch_size * node_count, patch_count, dimension)
        )
        return encoded.mean(dim=1).reshape(batch_size, node_count, dimension)

    def graph_update(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        relation_hops = []
        propagated = features
        for _ in range(self.hop_order):
            if adjacency.dim() == 2:
                propagated = torch.einsum("nm,bmd->bnd", adjacency, propagated)
            elif adjacency.dim() == 3:
                propagated = torch.einsum("bnm,bmd->bnd", adjacency, propagated)
            else:
                raise ValueError("adjacency must be two- or three-dimensional")
            relation_hops.append(propagated - features)
        return self.graph_mix(torch.cat(relation_hops, dim=-1))

    def source_gates(self, source_names: tuple[str, ...]) -> torch.Tensor:
        if not source_names:
            return self.source_logits.new_zeros(0)
        if len(source_names) == 1 or self.uniform_source_fusion:
            return self.source_logits.new_ones(len(source_names))
        indices = [INFORMATIVE_GRAPH_SOURCES.index(name) for name in source_names]
        gates = torch.sigmoid(self.source_logits[indices])
        if "hierarchy" in source_names:
            hierarchy_index = source_names.index("hierarchy")
            anchor = torch.zeros_like(gates, dtype=torch.bool)
            anchor[hierarchy_index] = True
            gates = torch.where(anchor, torch.ones_like(gates), gates)
        return gates

    def forward(
        self,
        patch_tokens: torch.Tensor,
        state: torch.Tensor,
        graph_parts: list[tuple[str, torch.Tensor]],
        node_embedding: torch.Tensor,
        level_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal_update(patch_tokens, state)
        graph_outputs = [self.graph_update(temporal, adjacency) for _, adjacency in graph_parts]
        source_names = tuple(name for name, _ in graph_parts)
        if graph_outputs:
            gates = self.source_gates(source_names)
            graph = sum(gate * output for gate, output in zip(gates, graph_outputs))
            graph = self.source_dropout(self.source_norm(graph))
        else:
            gates = self.source_logits.new_zeros(0)
            graph = torch.zeros_like(temporal)

        metadata = node_embedding + level_embedding
        if self.sequential_no_coevolution:
            next_state = self.output_norm(
                state + metadata + temporal + self.output_dropout(graph)
            )
        else:
            gate = torch.sigmoid(
                self.fusion_gate(
                    torch.cat([temporal, graph, node_embedding, level_embedding], dim=-1)
                )
            )
            fused = self.fusion(
                torch.cat([temporal, gate * graph, temporal * graph], dim=-1)
            )
            next_state = self.output_norm(
                state + metadata + self.output_dropout(fused)
            )
        return next_state, gates


class LAGTCN(_ForecastTransformBase):
    """Level-Aware Graph-Temporal Co-Evolution Network.

    Input shape: ``[batch, nodes, features, input_steps]``.
    Output shape: ``[batch, nodes, forecast_steps]`` in original load units.
    """

    architecture_version = "lagtcn_decoder_modes_hanchor_v3"

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
        patch_len: int = 8,
        patch_stride: int = 4,
        dropout: float = 0.1,
        hop_order: int = 2,
        use_level_awareness: bool = True,
        use_coevolution: bool = True,
        learn_source_fusion: bool = True,
        decoder_mode: str = "persistence_residual",
        residual_scale_mode: str = "unit",
        residual_scale_init: float = 1.0,
        seasonal_lag: int = 24,
    ) -> None:
        super().__init__(
            node_num,
            input_dim,
            hidden_dim,
            output_dim,
            num_layers,
            global_min,
            global_max,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        self.model_dim = max(64, int(hidden_dim))
        self.patch_len = max(1, min(int(patch_len), self.num_timesteps_in))
        self.patch_stride = max(1, int(patch_stride))
        self.use_level_awareness = bool(use_level_awareness)
        self.use_coevolution = bool(use_coevolution)
        self.learn_source_fusion = bool(learn_source_fusion)

        self.decoder_mode = str(decoder_mode).lower()
        if self.decoder_mode not in {"persistence_residual", "seasonal_residual", "direct"}:
            raise ValueError(f"unsupported decoder_mode={decoder_mode!r}")
        self.residual_scale_mode = str(residual_scale_mode).lower()
        if self.residual_scale_mode not in {"fixed", "unit", "learnable"}:
            raise ValueError(f"unsupported residual_scale_mode={residual_scale_mode!r}")
        requested_scale = float(residual_scale_init)
        self.residual_scale_init = 1.0 if self.residual_scale_mode == "unit" else requested_scale
        if not math.isfinite(self.residual_scale_init) or self.residual_scale_init < 0:
            raise ValueError("residual_scale_init must be finite and nonnegative")
        if self.residual_scale_mode == "learnable" and not 0 < self.residual_scale_init < 1:
            raise ValueError("learnable residual scale requires 0 < initial value < 1")
        self.seasonal_lag = int(seasonal_lag)
        if self.seasonal_lag <= 0:
            raise ValueError("seasonal_lag must be positive")
        if self.decoder_mode == "seasonal_residual" and self.num_timesteps_in < self.seasonal_lag:
            raise ValueError("seasonal_residual requires input_steps >= seasonal_lag")

        patch_count = 1 + max(
            0,
            (self.num_timesteps_in - self.patch_len) // self.patch_stride,
        )
        self.patch_proj = nn.Linear(self.patch_len, self.model_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, 1, patch_count, self.model_dim)
        )
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.initial_state = nn.Linear(self.num_timesteps_in, self.model_dim)
        self.node_embedding = nn.Embedding(self.node_num, self.model_dim)
        self.level_embedding = (
            nn.Embedding(16, self.model_dim) if self.use_level_awareness else None
        )
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            _LAGTCNBlock(
                model_dim=self.model_dim,
                nhead=_choose_nhead(self.model_dim),
                hop_order=hop_order,
                dropout=dropout,
                uniform_source_fusion=not self.learn_source_fusion,
                sequential_no_coevolution=not self.use_coevolution,
            )
            for _ in range(max(1, self.num_layers))
        )
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.output_dim),
        )
        nn.init.normal_(self.forecast_head[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.forecast_head[-1].bias)
        if self.decoder_mode != "direct" and self.residual_scale_mode == "learnable":
            logit = math.log(self.residual_scale_init / (1 - self.residual_scale_init))
            self.residual_scale_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        else:
            self.register_parameter("residual_scale_logit", None)

        self.hierarchy_source_adj: torch.Tensor | None = None
        self.similarity_source_adj: torch.Tensor | None = None
        self._configured_sources: tuple[str, ...] | None = None

    def set_graph_config(self, graph_config: dict) -> None:
        super().set_graph_config(graph_config)
        active_sources = graph_sources(self.graph_config["graph_mode"])
        if self._configured_sources is not None and self._configured_sources != active_sources:
            raise RuntimeError("construct a new model to change active graph sources")
        for block in self.blocks:
            if block.uniform_source_fusion or len(active_sources) <= 1:
                block.source_logits.requires_grad_(False)
        self._configured_sources = active_sources

    def set_static_graph_sources(self, hierarchy_adj=None, similarity_adj=None) -> None:
        self.hierarchy_source_adj = self._validate_static_source(
            hierarchy_adj, "hierarchy"
        )
        self.similarity_source_adj = self._validate_static_source(
            similarity_adj, "similarity"
        )

    def _validate_static_source(self, adjacency, name: str) -> torch.Tensor | None:
        if adjacency is None:
            return None
        tensor = torch.as_tensor(adjacency, dtype=torch.float32).detach().clone()
        if tuple(tensor.shape) != (self.node_num, self.node_num):
            raise ValueError(f"{name} adjacency has shape {tuple(tensor.shape)}")
        _require_finite(tensor, f"{name} adjacency")
        return tensor

    def _patch_tokens(self, target_series: torch.Tensor) -> torch.Tensor:
        batch_size, node_count, _ = target_series.shape
        patches = target_series.unfold(-1, self.patch_len, self.patch_stride)
        patch_count = patches.shape[2]
        tokens = self.patch_proj(
            patches.reshape(batch_size * node_count, patch_count, self.patch_len)
        ).reshape(batch_size, node_count, patch_count, self.model_dim)
        tokens = tokens + self.pos_embedding[:, :, :patch_count, :]
        return self.input_dropout(tokens)

    def _prepare_static_source(
        self,
        adjacency: torch.Tensor | None,
        features: torch.Tensor,
        name: str,
    ) -> torch.Tensor:
        if adjacency is None:
            raise RuntimeError(f"graph mode requires the {name} adjacency")
        prepared = torch.clamp(
            adjacency.to(device=features.device, dtype=features.dtype), min=0.0
        ).clone()
        if self.graph_config["include_self_loops"]:
            prepared.fill_diagonal_(1.0)
        return _row_normalize(prepared)

    def _compute_dynamic_adj(self, features: torch.Tensor) -> torch.Tensor:
        representation = features.reshape(features.shape[0], self.node_num, -1)
        similarity = self._similarity(
            representation,
            self.graph_config["dynamic_sim_type"],
            use_abs=True,
        )
        threshold = self.graph_config.get("dynamic_threshold")
        if threshold is None:
            raise ValueError("graph mode D requires dynamic_threshold")
        return build_threshold_similarity_adj_torch(
            similarity,
            threshold=float(threshold),
            include_self_loops=self.graph_config["include_self_loops"],
        )

    def _graph_parts(self, features: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        mode = "H" if self.graph_config is None else self.graph_config["graph_mode"]
        parts: list[tuple[str, torch.Tensor]] = []
        for source in graph_sources(mode):
            if source == "identity":
                adjacency = torch.eye(
                    self.node_num, device=features.device, dtype=features.dtype
                )
            elif source == "hierarchy":
                adjacency = self._prepare_static_source(
                    self.hierarchy_source_adj, features, "hierarchy"
                )
            elif source == "similarity":
                adjacency = self._prepare_static_source(
                    self.similarity_source_adj, features, "similarity"
                )
            elif source == "adaptive":
                adjacency = _row_normalize(
                    torch.clamp(self._compute_adaptive_adj(), min=0.0)
                )
            elif source == "dynamic":
                adjacency = _row_normalize(
                    torch.clamp(self._compute_dynamic_adj(features), min=0.0)
                )
            else:  # guarded by graph_sources
                raise RuntimeError(f"unknown graph source {source}")
            parts.append((source, adjacency.to(features.device, features.dtype)))
        return parts

    def _level_embedding(self, device: torch.device) -> torch.Tensor:
        level_ids = self.hier_level_ids.to(device=device).clamp(
            min=0, max=self.level_embedding.num_embeddings - 1
        )
        return self.level_embedding(level_ids).unsqueeze(0)

    def residual_scale(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        if self.decoder_mode == "direct" or self.residual_scale_mode == "unit":
            value = 1.0
        elif self.residual_scale_logit is not None:
            return torch.sigmoid(self.residual_scale_logit)
        else:
            value = self.residual_scale_init
        parameter = self.forecast_head[-1].weight
        device = reference.device if reference is not None else parameter.device
        dtype = reference.dtype if reference is not None else parameter.dtype
        return torch.tensor(value, device=device, dtype=dtype)

    def _decoder_reference(self, target_series: torch.Tensor) -> torch.Tensor | None:
        if self.decoder_mode == "direct":
            return None
        if self.decoder_mode == "persistence_residual":
            return target_series[:, :, -1:].expand(-1, -1, self.output_dim)
        input_length = target_series.shape[-1]
        indices = input_length - self.seasonal_lag + (
            torch.arange(self.output_dim, device=target_series.device) % self.seasonal_lag
        )
        return target_series.index_select(dim=-1, index=indices)

    def get_graph_source_gates(self) -> list[dict[str, object]]:
        mode = "H" if self.graph_config is None else self.graph_config["graph_mode"]
        source_names = graph_sources(mode)
        return [
            {
                "block": index,
                "gates": {
                    name: float(value)
                    for name, value in zip(
                        source_names,
                        block.source_gates(source_names).detach().cpu().tolist(),
                    )
                },
            }
            for index, block in enumerate(self.blocks, start=1)
        ]

    def forward_normalized(self, features: torch.Tensor) -> torch.Tensor:
        target_series = self._extract_target_series(features)
        batch_size, node_count, input_length = target_series.shape
        if node_count != self.node_num:
            raise ValueError(f"expected {self.node_num} nodes, got {node_count}")
        if input_length != self.num_timesteps_in:
            raise ValueError(
                f"expected input length {self.num_timesteps_in}, got {input_length}"
            )

        patch_tokens = self._patch_tokens(target_series)
        state = self.initial_state(target_series)
        node_ids = torch.arange(node_count, device=features.device)
        node_embedding = self.node_embedding(node_ids).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if self.use_level_awareness:
            level_embedding = self._level_embedding(features.device).expand(
                batch_size, -1, -1
            )
        else:
            level_embedding = torch.zeros_like(node_embedding)
        state = state + node_embedding + level_embedding
        graph_parts = self._graph_parts(features)
        for block in self.blocks:
            state, _ = block(
                patch_tokens,
                state,
                graph_parts,
                node_embedding,
                level_embedding,
            )

        decoder_output = self.forecast_head(state)
        reference = self._decoder_reference(target_series)
        prediction = (
            decoder_output
            if reference is None
            else reference + self.residual_scale(decoder_output) * decoder_output
        )
        return _require_finite(
            prediction.reshape(batch_size, node_count, self.output_dim),
            "normalized prediction",
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.prediction_from_normalized(self.forward_normalized(features))
