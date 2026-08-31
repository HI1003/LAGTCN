import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, TransformerConv

from lagtcn.core.graphs import FINAL_GRAPH_SOURCE_POLICY, build_threshold_similarity_adj_torch
from lagtcn.models.backbones import BaseGCNGRUModel
from lagtcn.core.naming import LAGTCN_INFORMATIVE_GRAPH_SOURCES, lagtcn_graph_sources


def _choose_nhead(hidden_dim: int) -> int:
    for nhead in (8, 4, 2, 1):
        if hidden_dim % nhead == 0:
            return nhead
    return 1


def _row_normalize(adj: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    row_sum = adj.sum(dim=-1, keepdim=True).clamp_min(eps)
    return adj / row_sum


class _GraphBaselineBase(BaseGCNGRUModel):
    """Shared graph-resolution utilities for graph baselines."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        stgnn_graph_source: str = "hybrid",
    ):
        # Backbone defined in subclasses; we only reuse normalization + graph config utils.
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
            gnn_type="none",
            temporal_type="gru",
        )
        # Subclasses implement their own forecasting backbones. The parent
        # constructor creates a GRU/linear placeholder only to provide shared
        # transform and graph utilities; unregister it so it is not optimized,
        # checkpointed, or counted as part of a dedicated baseline.
        self.layer_norm = None
        self.projection = None
        self.node_encoder = None
        self.temporal_encoder = None
        self.temporal_pos_dropout = None
        self.gnn_layer = None
        self.gcn = None
        self.gru = None
        self.stgnn_graph_source = self._normalize_stgnn_graph_source(stgnn_graph_source)

    @staticmethod
    def _normalize_stgnn_graph_source(value: str) -> str:
        source = str(value).strip().lower()
        aliases = {
            "project": "project",
            "hierarchy": "project",
            "hierarchy_graph": "project",
            "project_graph": "project",
            "native": "native",
            "native_graph": "native",
            "adaptive": "native",
            "model": "native",
            "hybrid": "hybrid",
            "hybrid_graph": "hybrid",
        }
        if source not in aliases:
            raise ValueError(
                f"Unsupported stgnn_graph_source={value!r}. "
                "Choose from project/native/hybrid."
            )
        return aliases[source]

    def set_graph_config(self, graph_config: dict, base_adj=None, base_edge_index=None, base_edge_weight=None) -> None:
        super().set_graph_config(
            graph_config,
            base_adj=base_adj,
            base_edge_index=base_edge_index,
            base_edge_weight=base_edge_weight,
        )
        self.stgnn_graph_source = self._normalize_stgnn_graph_source(
            graph_config.get("stgnn_graph_source", self.stgnn_graph_source)
        )
        if hasattr(self, "top_k"):
            native_top_k = graph_config.get("native_top_k")
            if native_top_k is not None:
                self.top_k = max(1, min(int(native_top_k), max(1, self.node_num - 1)))

    @staticmethod
    def _extract_target_series(x: torch.Tensor) -> torch.Tensor:
        # x: [S, N, F, T], use target channel only (F index 0).
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [S,N,F,T], got {tuple(x.shape)}")
        return x[:, :, 0, :]

    def _resolve_dense_adj(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        n_nodes = x.shape[1]
        edge_index_use, edge_weight_use = self._resolve_graph_edges(x, edge_index, edge_weight)
        if edge_weight_use is None:
            edge_weight_use = torch.ones(
                edge_index_use.shape[1],
                device=x.device,
                dtype=x.dtype,
            )
        adj = x.new_zeros((n_nodes, n_nodes))
        adj[edge_index_use[0], edge_index_use[1]] = edge_weight_use.to(dtype=x.dtype)
        adj = torch.clamp(adj, min=0.0)
        if self.graph_config is None or self.graph_config.get("include_self_loops", True):
            adj.fill_diagonal_(1.0)
        return _row_normalize(adj)

    def _combine_project_and_native_adj(
        self,
        project_adj: torch.Tensor,
        native_adj: torch.Tensor,
    ) -> torch.Tensor:
        source = self.stgnn_graph_source
        if source == "native":
            adj = native_adj
        elif source == "project":
            adj = project_adj
        else:
            adj = 0.5 * project_adj + 0.5 * native_adj
        if self.graph_config is None or self.graph_config.get("include_self_loops", True):
            eye = torch.eye(adj.shape[0], device=adj.device, dtype=torch.bool)
            adj = adj.masked_fill(eye, 1.0)
        return _row_normalize(adj)


class _DCRNNDiffusionGraphConv(nn.Module):
    """Chebyshev diffusion graph convolution used inside a DCGRU cell."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        max_diffusion_step: int,
        num_supports: int = 2,
        bias_start: float = 0.0,
    ):
        super().__init__()
        self.max_diffusion_step = max(0, int(max_diffusion_step))
        self.num_supports = int(num_supports)
        num_matrices = self.num_supports * self.max_diffusion_step + 1
        self.linear = nn.Linear(num_matrices * int(input_dim), int(output_dim))
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, float(bias_start))

    def forward(
        self,
        features: torch.Tensor,
        supports: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        if len(supports) != self.num_supports:
            raise ValueError(
                f"Expected {self.num_supports} diffusion supports, got {len(supports)}."
            )
        outputs = [features]
        for support in supports:
            x0 = features
            if self.max_diffusion_step >= 1:
                x1 = torch.einsum("nm,bmc->bnc", support, x0)
                outputs.append(x1)
                for _ in range(2, self.max_diffusion_step + 1):
                    x2 = 2.0 * torch.einsum("nm,bmc->bnc", support, x1) - x0
                    outputs.append(x2)
                    x0, x1 = x1, x2
        return self.linear(torch.cat(outputs, dim=-1))


class _DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell from the DCRNN encoder/decoder."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        max_diffusion_step: int,
        projection_dim: int | None = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        combined_dim = self.input_dim + self.hidden_dim
        self.gate_conv = _DCRNNDiffusionGraphConv(
            combined_dim,
            2 * self.hidden_dim,
            max_diffusion_step=max_diffusion_step,
            num_supports=2,
            bias_start=1.0,
        )
        self.candidate_conv = _DCRNNDiffusionGraphConv(
            combined_dim,
            self.hidden_dim,
            max_diffusion_step=max_diffusion_step,
            num_supports=2,
            bias_start=0.0,
        )
        self.projection = (
            nn.Linear(self.hidden_dim, int(projection_dim), bias=False)
            if projection_dim is not None
            else None
        )
        if self.projection is not None:
            nn.init.xavier_uniform_(self.projection.weight)

    def forward(
        self,
        inputs: torch.Tensor,
        state: torch.Tensor,
        supports: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_input = torch.cat([inputs, state], dim=-1)
        reset_gate, update_gate = torch.sigmoid(
            self.gate_conv(gate_input, supports)
        ).chunk(2, dim=-1)
        candidate_input = torch.cat([inputs, reset_gate * state], dim=-1)
        candidate = torch.tanh(self.candidate_conv(candidate_input, supports))
        new_state = update_gate * state + (1.0 - update_gate) * candidate
        output = self.projection(new_state) if self.projection is not None else new_state
        return output, new_state


class DCRNNBaseline(_GraphBaselineBase):
    """DCRNN encoder--decoder with dual random-walk diffusion and curriculum."""

    architecture_version = "dcrnn_seq2seq_dual_random_walk_official_v1"
    requires_teacher_forcing_targets = True

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        diffusion_steps: int = 2,
        cl_decay_steps: int = 1000,
        use_curriculum_learning: bool = True,
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
            stgnn_graph_source="project",
        )
        self.state_dim = max(1, int(hidden_dim))
        self.diffusion_steps = max(0, int(diffusion_steps))
        self.cl_decay_steps = max(1, int(cl_decay_steps))
        self.use_curriculum_learning = bool(use_curriculum_learning)
        self.num_rnn_layers = max(1, int(num_layers))

        encoder_cells = []
        for layer_idx in range(self.num_rnn_layers):
            encoder_cells.append(
                _DCGRUCell(
                    input_dim=1 if layer_idx == 0 else self.state_dim,
                    hidden_dim=self.state_dim,
                    max_diffusion_step=self.diffusion_steps,
                )
            )
        self.encoder_cells = nn.ModuleList(encoder_cells)

        decoder_cells = []
        for layer_idx in range(self.num_rnn_layers):
            decoder_cells.append(
                _DCGRUCell(
                    input_dim=1 if layer_idx == 0 else self.state_dim,
                    hidden_dim=self.state_dim,
                    max_diffusion_step=self.diffusion_steps,
                    projection_dim=1 if layer_idx == self.num_rnn_layers - 1 else None,
                )
            )
        self.decoder_cells = nn.ModuleList(decoder_cells)
        self._last_sampling_threshold: float | None = None

    @staticmethod
    def _random_walk(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=1, keepdim=True)
        return adjacency / degree.clamp_min(1e-8)

    def _dual_random_walk_supports(
        self, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward = self._random_walk(adjacency).transpose(0, 1)
        backward = self._random_walk(adjacency.transpose(0, 1)).transpose(0, 1)
        return forward, backward

    def _project_adjacency(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        edge_index_use, edge_weight_use = self._resolve_graph_edges(
            x, edge_index, edge_weight
        )
        if edge_weight_use is None:
            edge_weight_use = torch.ones(
                edge_index_use.shape[1], device=x.device, dtype=x.dtype
            )
        adjacency = x.new_zeros((self.node_num, self.node_num))
        adjacency[edge_index_use[0], edge_index_use[1]] = edge_weight_use.to(
            device=x.device, dtype=x.dtype
        )
        adjacency = torch.clamp(adjacency, min=0.0)
        adjacency.fill_diagonal_(0.0)
        return adjacency

    @staticmethod
    def sampling_threshold(batches_seen: int, decay_steps: int) -> float:
        ratio = float(batches_seen) / float(decay_steps)
        if ratio > 60.0:
            return 0.0
        return float(decay_steps / (decay_steps + math.exp(ratio)))

    def _encode(
        self,
        target_series: torch.Tensor,
        supports: tuple[torch.Tensor, torch.Tensor],
    ) -> list[torch.Tensor]:
        batch, n_nodes, _ = target_series.shape
        states = [
            target_series.new_zeros(batch, n_nodes, self.state_dim)
            for _ in range(self.num_rnn_layers)
        ]
        for time_idx in range(target_series.shape[-1]):
            layer_input = target_series[:, :, time_idx : time_idx + 1]
            for layer_idx, cell in enumerate(self.encoder_cells):
                layer_input, states[layer_idx] = cell(
                    layer_input, states[layer_idx], supports
                )
        return states

    def _decode(
        self,
        states: list[torch.Tensor],
        supports: tuple[torch.Tensor, torch.Tensor],
        teacher_targets: torch.Tensor | None,
        batches_seen: int,
    ) -> torch.Tensor:
        batch = states[0].shape[0]
        decoder_input = states[0].new_zeros(batch, self.node_num, 1)
        predictions = []
        threshold = self.sampling_threshold(batches_seen, self.cl_decay_steps)
        self._last_sampling_threshold = threshold
        for horizon_idx in range(self.output_dim):
            layer_input = decoder_input
            for layer_idx, cell in enumerate(self.decoder_cells):
                layer_input, states[layer_idx] = cell(
                    layer_input, states[layer_idx], supports
                )
            prediction = layer_input
            predictions.append(prediction)
            if teacher_targets is not None:
                if self.use_curriculum_learning:
                    use_teacher = bool(
                        torch.rand((), device=prediction.device).item() < threshold
                    )
                else:
                    use_teacher = True
                decoder_input = (
                    teacher_targets[:, :, horizon_idx : horizon_idx + 1]
                    if use_teacher
                    else prediction
                )
            else:
                decoder_input = prediction
        return torch.cat(predictions, dim=-1)

    def _forward_normalized(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
        teacher_targets: torch.Tensor | None,
        batches_seen: int,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        if teacher_targets is not None:
            if teacher_targets.dim() == 2:
                teacher_targets = teacher_targets.unsqueeze(-1)
            if tuple(teacher_targets.shape) != (
                target_series.shape[0], self.node_num, self.output_dim
            ):
                raise ValueError(
                    "DCRNN teacher target shape mismatch: "
                    f"{tuple(teacher_targets.shape)}."
                )
        adjacency = self._project_adjacency(x, edge_index, edge_weight)
        supports = self._dual_random_walk_supports(adjacency)
        states = self._encode(target_series, supports)
        return self._decode(
            states,
            supports,
            teacher_targets=teacher_targets,
            batches_seen=int(batches_seen),
        )

    def forward_train(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        teacher_targets: torch.Tensor,
        batches_seen: int,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_norm = self._forward_normalized(
            x,
            edge_index,
            edge_weight,
            teacher_targets=teacher_targets,
            batches_seen=batches_seen,
        )
        return self.inverse_log(self.denormalize(pred_norm))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_norm = self._forward_normalized(
            x,
            edge_index,
            edge_weight,
            teacher_targets=None,
            batches_seen=0,
        )
        return self.inverse_log(self.denormalize(pred_norm))
class _MTGNNMixProp(nn.Module):
    """Official MTGNN mix-hop propagation layer."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        graph_depth: int,
        alpha: float,
    ):
        super().__init__()
        self.graph_depth = max(1, int(graph_depth))
        self.alpha = float(alpha)
        self.mlp = nn.Conv2d(
            (self.graph_depth + 1) * input_channels,
            output_channels,
            kernel_size=(1, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        adjacency = adjacency + torch.eye(
            adjacency.shape[0], device=x.device, dtype=x.dtype
        )
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-8)
        hidden = x
        outputs = [hidden]
        for _ in range(self.graph_depth):
            propagated = torch.einsum("bcwt,vw->bcvt", hidden, adjacency)
            hidden = self.alpha * x + (1.0 - self.alpha) * propagated
            outputs.append(hidden)
        return self.mlp(torch.cat(outputs, dim=1))


class _MTGNNDilatedInception(nn.Module):
    """Parallel temporal convolutions with the released [2,3,6,7] kernels."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilation_factor: int,
    ):
        super().__init__()
        if output_channels % 4:
            raise ValueError("MTGNN convolution channels must be divisible by four.")
        per_kernel = output_channels // 4
        self.kernel_set = (2, 3, 6, 7)
        self.convolutions = nn.ModuleList(
            nn.Conv2d(
                input_channels,
                per_kernel,
                kernel_size=(1, kernel),
                dilation=(1, int(dilation_factor)),
            )
            for kernel in self.kernel_set
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [conv(x) for conv in self.convolutions]
        target_length = outputs[-1].shape[-1]
        return torch.cat([value[..., -target_length:] for value in outputs], dim=1)


class _MTGNNGraphConstructor(nn.Module):
    """Asymmetric sparse directed graph constructor from MTGNN."""

    def __init__(
        self,
        num_nodes: int,
        top_k: int,
        embedding_dim: int,
        tanh_alpha: float,
    ):
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.top_k = max(1, min(int(top_k), self.num_nodes))
        self.tanh_alpha = float(tanh_alpha)
        self.embedding1 = nn.Embedding(self.num_nodes, int(embedding_dim))
        self.embedding2 = nn.Embedding(self.num_nodes, int(embedding_dim))
        self.linear1 = nn.Linear(int(embedding_dim), int(embedding_dim))
        self.linear2 = nn.Linear(int(embedding_dim), int(embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        nodevec1 = torch.tanh(
            self.tanh_alpha * self.linear1(self.embedding1(indices))
        )
        nodevec2 = torch.tanh(
            self.tanh_alpha * self.linear2(self.embedding2(indices))
        )
        antisymmetric = (
            nodevec1 @ nodevec2.transpose(0, 1)
            - nodevec2 @ nodevec1.transpose(0, 1)
        )
        adjacency = torch.relu(torch.tanh(self.tanh_alpha * antisymmetric))
        if self.top_k < adjacency.shape[1]:
            # Retain the released implementation's stochastic tie breaking while
            # training, but make validation/test inference deterministic.
            ranking_scores = adjacency
            if self.training:
                ranking_scores = ranking_scores + torch.rand_like(adjacency) * 0.01
            _, top_indices = torch.topk(
                ranking_scores,
                self.top_k,
                dim=1,
            )
            mask = torch.zeros_like(adjacency)
            mask.scatter_(1, top_indices, 1.0)
            adjacency = adjacency * mask
        return adjacency


class _MTGNNLayerNorm(nn.Module):
    """Node-index-aware LayerNorm used by the released MTGNN network."""

    def __init__(self, channels: int, num_nodes: int, time_steps: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, num_nodes, time_steps))
        self.bias = nn.Parameter(torch.zeros(channels, num_nodes, time_steps))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        weight = self.weight[:, indices, :]
        bias = self.bias[:, indices, :]
        return F.layer_norm(x, tuple(x.shape[1:]), weight, bias, self.eps)


class MTGNNBaseline(_GraphBaselineBase):
    """Faithful MTGNN architecture with its native learned directed graph."""

    architecture_version = "mtgnn_official_direct_multihorizon_v3"

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        num_timesteps_in: int = 168,
        adaptive_emb_dim: int = 40,
        top_k: int = 20,
        gdep: int = 2,
        prop_alpha: float = 0.05,
        tanh_alpha: float = 3.0,
        dropout: float = 0.3,
        dilation_exponential: int = 1,
        stgnn_graph_source: str = "native",
    ):
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
            stgnn_graph_source=stgnn_graph_source,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        channels = max(8, int(hidden_dim))
        channels = 4 * math.ceil(channels / 4)
        self.residual_channels = channels
        self.conv_channels = channels
        self.skip_channels = 2 * channels
        self.end_channels = 4 * channels
        self.layers = max(1, int(num_layers))
        self.dropout_probability = float(dropout)
        self.top_k = max(1, min(int(top_k), self.node_num))
        self.dilation_exponential = max(1, int(dilation_exponential))

        self.start_conv = nn.Conv2d(1, channels, kernel_size=(1, 1))
        self.graph_constructor = _MTGNNGraphConstructor(
            num_nodes=self.node_num,
            top_k=self.top_k,
            embedding_dim=int(adaptive_emb_dim),
            tanh_alpha=float(tanh_alpha),
        )
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.mixprop_forward = nn.ModuleList()
        self.mixprop_backward = nn.ModuleList()
        self.norms = nn.ModuleList()

        kernel_size = 7
        if self.dilation_exponential > 1:
            self.receptive_field = int(
                1
                + (kernel_size - 1)
                * (self.dilation_exponential ** self.layers - 1)
                / (self.dilation_exponential - 1)
            )
        else:
            self.receptive_field = self.layers * (kernel_size - 1) + 1

        dilation = 1
        for layer_idx in range(1, self.layers + 1):
            if self.dilation_exponential > 1:
                rf_size = int(
                    1
                    + (kernel_size - 1)
                    * (self.dilation_exponential ** layer_idx - 1)
                    / (self.dilation_exponential - 1)
                )
            else:
                rf_size = 1 + layer_idx * (kernel_size - 1)
            self.filter_convs.append(
                _MTGNNDilatedInception(channels, channels, dilation)
            )
            self.gate_convs.append(
                _MTGNNDilatedInception(channels, channels, dilation)
            )
            self.residual_convs.append(
                nn.Conv2d(channels, channels, kernel_size=(1, 1))
            )
            if self.num_timesteps_in > self.receptive_field:
                skip_kernel = self.num_timesteps_in - rf_size + 1
                norm_time = self.num_timesteps_in - rf_size + 1
            else:
                skip_kernel = self.receptive_field - rf_size + 1
                norm_time = self.receptive_field - rf_size + 1
            self.skip_convs.append(
                nn.Conv2d(channels, self.skip_channels, kernel_size=(1, skip_kernel))
            )
            self.mixprop_forward.append(
                _MTGNNMixProp(channels, channels, gdep, prop_alpha)
            )
            self.mixprop_backward.append(
                _MTGNNMixProp(channels, channels, gdep, prop_alpha)
            )
            self.norms.append(_MTGNNLayerNorm(channels, self.node_num, norm_time))
            dilation *= self.dilation_exponential

        if self.num_timesteps_in > self.receptive_field:
            skip0_kernel = self.num_timesteps_in
            skip_end_kernel = self.num_timesteps_in - self.receptive_field + 1
        else:
            skip0_kernel = self.receptive_field
            skip_end_kernel = 1
        self.skip0 = nn.Conv2d(1, self.skip_channels, kernel_size=(1, skip0_kernel))
        self.skip_end = nn.Conv2d(
            channels, self.skip_channels, kernel_size=(1, skip_end_kernel)
        )
        self.end_conv1 = nn.Conv2d(
            self.skip_channels, self.end_channels, kernel_size=(1, 1)
        )
        self.end_conv2 = nn.Conv2d(
            self.end_channels, self.output_dim, kernel_size=(1, 1)
        )
        self.register_buffer("node_indices", torch.arange(self.node_num), persistent=True)
        self._last_adaptive_adjacency: torch.Tensor | None = None

    def set_graph_config(
        self,
        graph_config: dict,
        base_adj=None,
        base_edge_index=None,
        base_edge_weight=None,
    ) -> None:
        super().set_graph_config(
            graph_config,
            base_adj=base_adj,
            base_edge_index=base_edge_index,
            base_edge_weight=base_edge_weight,
        )
        self.graph_constructor.top_k = max(
            1, min(int(self.top_k), self.node_num)
        )

    def _project_adjacency(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        edge_index_use, edge_weight_use = self._resolve_graph_edges(
            x, edge_index, edge_weight
        )
        if edge_weight_use is None:
            edge_weight_use = torch.ones(
                edge_index_use.shape[1], device=x.device, dtype=x.dtype
            )
        adjacency = x.new_zeros((self.node_num, self.node_num))
        adjacency[edge_index_use[0], edge_index_use[1]] = edge_weight_use.to(
            device=x.device, dtype=x.dtype
        )
        adjacency = torch.clamp(adjacency, min=0.0)
        adjacency.fill_diagonal_(0.0)
        return adjacency

    def get_adaptive_adjacency(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        indices = self.node_indices
        if device is not None:
            indices = indices.to(device)
        adjacency = self.graph_constructor(indices)
        if dtype is not None:
            adjacency = adjacency.to(dtype=dtype)
        return adjacency

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        if target_series.shape[-1] != self.num_timesteps_in:
            raise ValueError(
                f"MTGNN expected T_in={self.num_timesteps_in}, got {target_series.shape[-1]}"
            )
        model_input = target_series.unsqueeze(1)
        if self.num_timesteps_in < self.receptive_field:
            model_input = F.pad(
                model_input,
                (self.receptive_field - self.num_timesteps_in, 0, 0, 0),
            )

        source = self.stgnn_graph_source
        if source == "native":
            adjacency = self.graph_constructor(self.node_indices)
        elif source == "project":
            adjacency = self._project_adjacency(x, edge_index, edge_weight)
        else:
            raise ValueError(
                "Faithful MTGNN supports either its native learned graph or the "
                "original predefined-adjacency path; hybrid graph mixing is not part "
                "of the published architecture."
            )
        adjacency = adjacency.to(device=x.device, dtype=x.dtype)
        self._last_adaptive_adjacency = adjacency.detach()

        hidden = self.start_conv(model_input)
        skip = self.skip0(
            F.dropout(
                model_input,
                self.dropout_probability,
                training=self.training,
            )
        )
        for layer_idx in range(self.layers):
            residual = hidden
            filtered = torch.tanh(self.filter_convs[layer_idx](hidden))
            gated = torch.sigmoid(self.gate_convs[layer_idx](hidden))
            hidden = filtered * gated
            hidden = F.dropout(
                hidden,
                self.dropout_probability,
                training=self.training,
            )
            skip = self.skip_convs[layer_idx](hidden) + skip
            hidden = (
                self.mixprop_forward[layer_idx](hidden, adjacency)
                + self.mixprop_backward[layer_idx](hidden, adjacency.transpose(0, 1))
            )
            hidden = hidden + residual[..., -hidden.shape[-1] :]
            hidden = self.norms[layer_idx](hidden, self.node_indices)

        skip = self.skip_end(hidden) + skip
        output = self.end_conv2(F.relu(self.end_conv1(F.relu(skip))))
        if output.shape[-1] != 1:
            raise RuntimeError(f"MTGNN expected singleton output time axis, got {output.shape}")
        pred_norm = output.squeeze(-1).transpose(1, 2).contiguous()
        return self.inverse_log(self.denormalize(pred_norm))


class _LAGTCNBlock(nn.Module):
    """One level-aware graph-temporal co-evolution block."""

    def __init__(
        self,
        model_dim: int,
        nhead: int,
        hop_order: int = 2,
        dropout: float = 0.1,
        uniform_source_fusion: bool = False,
        sequential_no_coevolution: bool = False,
    ):
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
        self.source_logits = nn.Parameter(torch.zeros(len(LAGTCN_INFORMATIVE_GRAPH_SOURCES)))
        # One shared, bias-free relation projection keeps the graph-transformation
        # parameter budget identical between single- and multi-source variants.
        self.graph_mix = nn.Linear(
            self.hop_order * self.model_dim,
            self.model_dim,
            bias=False,
        )
        self.source_norm = nn.LayerNorm(self.model_dim)
        self.source_dropout = nn.Dropout(dropout)
        self.fusion_gate = nn.Linear(self.model_dim * 4, self.model_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.model_dim * 3, self.model_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim * 4, self.model_dim),
        )
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.output_dropout = nn.Dropout(dropout)
        if self.sequential_no_coevolution:
            # The sequential ablation never evaluates the co-evolution gate/MLP.
            self.fusion_gate = None
            self.fusion = None

    def temporal_update(self, patch_tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, n_patches, dim = patch_tokens.shape
        tokens = patch_tokens + state.unsqueeze(2)
        encoded = self.temporal_encoder(tokens.reshape(bsz * n_nodes, n_patches, dim))
        return encoded.mean(dim=1).reshape(bsz, n_nodes, dim)

    def graph_update(self, features: torch.Tensor, adj: torch.Tensor, part_name: str) -> torch.Tensor:
        del part_name  # All sources use the same transformation.
        relation_hops = []
        h = features
        for _ in range(self.hop_order):
            if adj.dim() == 2:
                h = torch.einsum("nm,bmd->bnd", adj, h)
            elif adj.dim() == 3:
                if adj.shape[0] != features.shape[0]:
                    raise ValueError(
                        f"Batched adjacency has batch size {adj.shape[0]}, "
                        f"but node features have batch size {features.shape[0]}."
                    )
                h = torch.einsum("bnm,bmd->bnd", adj, h)
            else:
                raise ValueError(f"Expected adjacency with 2 or 3 dimensions, got {tuple(adj.shape)}")
            # Remove the identity path: this branch represents only information
            # imported through cross-node relations. Identity adjacency is zero.
            relation_hops.append(h - features)
        return self.graph_mix(torch.cat(relation_hops, dim=-1))

    def source_gates(self, part_names: tuple[str, ...]) -> torch.Tensor:
        """Return independent block gates, with hierarchy fixed as unit anchor."""
        if not part_names:
            return self.source_logits.new_zeros(0)
        if len(part_names) == 1 or self.uniform_source_fusion:
            return self.source_logits.new_ones(len(part_names))
        if "identity" in part_names:
            raise ValueError("Identity cannot be fused with informative LAGTCN graph sources.")
        indices = [
            LAGTCN_INFORMATIVE_GRAPH_SOURCES.index(part_name)
            for part_name in part_names
        ]
        gates = torch.sigmoid(self.source_logits[indices])
        if "hierarchy" in part_names:
            hierarchy_index = part_names.index("hierarchy")
            anchor_mask = torch.zeros_like(gates, dtype=torch.bool)
            anchor_mask[hierarchy_index] = True
            gates = torch.where(anchor_mask, torch.ones_like(gates), gates)
        return gates

    def forward(
        self,
        patch_tokens: torch.Tensor,
        state: torch.Tensor,
        graph_parts: list[tuple[str, torch.Tensor]],
        node_emb: torch.Tensor,
        level_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_temporal = self.temporal_update(patch_tokens, state)
        outputs = []
        for part_name, adj in graph_parts:
            outputs.append(self.graph_update(z_temporal, adj, part_name))

        part_names = tuple(part_name for part_name, _ in graph_parts)
        if outputs:
            source_gates = self.source_gates(part_names)
            z_graph = sum(w * output for w, output in zip(source_gates, outputs))
            z_graph = self.source_dropout(self.source_norm(z_graph))
        else:
            source_gates = self.source_logits.new_zeros(0)
            z_graph = torch.zeros_like(z_temporal)
        meta = node_emb + level_emb
        if self.sequential_no_coevolution:
            state_next = self.output_norm(
                state + meta + z_temporal + self.output_dropout(z_graph)
            )
        else:
            gate = torch.sigmoid(
                self.fusion_gate(
                    torch.cat([z_temporal, z_graph, node_emb, level_emb], dim=-1)
                )
            )
            fused = self.fusion(
                torch.cat([z_temporal, gate * z_graph, z_temporal * z_graph], dim=-1)
            )
            state_next = self.output_norm(state + meta + self.output_dropout(fused))
        return state_next, source_gates


class LAGTCNBaseline(_GraphBaselineBase):
    """Level-aware Adaptive Graph-Temporal Co-evolution Network.

    LAGTCN jointly updates temporal and graph-aware hidden representations.
    It does not first produce a temporal forecast and then correct that forecast;
    graph information participates in intermediate representation updates.

    The temporal path is a project-specific, PatchTST-inspired patch transformer.
    It is intentionally not the released PatchTST backbone.
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
        temporal_backbone: str = "patch_transformer",
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
        self.supports_data_driven_graph_sources = True
        self.model_dim = max(64, int(hidden_dim))
        self.temporal_backbone = str(temporal_backbone).lower()
        if self.temporal_backbone != "patch_transformer":
            raise ValueError(
                "LAGTCN requires its project-specific "
                "temporal_backbone='patch_transformer'."
            )
        self.patch_len = max(1, min(int(patch_len), self.num_timesteps_in))
        self.patch_stride = max(1, int(patch_stride))
        self.use_level_awareness = bool(use_level_awareness)
        self.use_coevolution = bool(use_coevolution)
        self.learn_source_fusion = bool(learn_source_fusion)
        self.max_num_patches = 1 + max(0, (self.num_timesteps_in - self.patch_len) // self.patch_stride)
        self.patch_proj = nn.Linear(self.patch_len, self.model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, self.max_num_patches, self.model_dim))
        self.decoder_mode = str(decoder_mode).strip().lower()
        if self.decoder_mode not in {
            "persistence_residual", "seasonal_residual", "direct"
        }:
            raise ValueError(f"Unsupported LAGTCN decoder_mode={decoder_mode!r}.")
        self.residual_scale_mode = str(residual_scale_mode).strip().lower()
        if self.residual_scale_mode not in {"fixed", "unit", "learnable"}:
            raise ValueError(
                f"Unsupported LAGTCN residual_scale_mode={residual_scale_mode!r}."
            )
        requested_residual_scale_init = float(residual_scale_init)
        self.residual_scale_init = (
            1.0 if self.residual_scale_mode == "unit" else requested_residual_scale_init
        )
        if not math.isfinite(self.residual_scale_init) or self.residual_scale_init < 0.0:
            raise ValueError("LAGTCN residual_scale_init must be finite and nonnegative.")
        if self.residual_scale_mode == "learnable" and not 0.0 < self.residual_scale_init < 1.0:
            raise ValueError(
                "A learnable LAGTCN residual scale uses a sigmoid gate and therefore "
                "requires 0 < residual_scale_init < 1."
            )
        self.seasonal_lag = int(seasonal_lag)
        if self.seasonal_lag <= 0:
            raise ValueError("LAGTCN seasonal_lag must be positive.")
        if self.decoder_mode == "seasonal_residual" and self.num_timesteps_in < self.seasonal_lag:
            raise ValueError(
                "seasonal_residual requires num_timesteps_in >= seasonal_lag, got "
                f"{self.num_timesteps_in} < {self.seasonal_lag}."
            )
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.initial_state = nn.Linear(self.num_timesteps_in, self.model_dim)
        self.node_embedding = nn.Embedding(node_num, self.model_dim)
        self.level_embedding = (
            nn.Embedding(16, self.model_dim) if self.use_level_awareness else None
        )
        self.input_dropout = nn.Dropout(dropout)
        nhead = _choose_nhead(self.model_dim)
        self.blocks = nn.ModuleList(
            _LAGTCNBlock(
                model_dim=self.model_dim,
                nhead=nhead,
                hop_order=hop_order,
                dropout=dropout,
                uniform_source_fusion=not self.learn_source_fusion,
                sequential_no_coevolution=not self.use_coevolution,
            )
            for _ in range(max(1, int(num_layers)))
        )
        self.forecast_head = nn.Sequential(
            nn.LayerNorm(self.model_dim),
            nn.Linear(self.model_dim, self.output_dim),
        )
        nn.init.normal_(self.forecast_head[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.forecast_head[-1].bias)
        if self.decoder_mode != "direct" and self.residual_scale_mode == "learnable":
            residual_scale_logit = math.log(
                self.residual_scale_init / (1.0 - self.residual_scale_init)
            )
            self.residual_scale_logit = nn.Parameter(
                torch.tensor(residual_scale_logit, dtype=torch.float32)
            )
        else:
            self.register_parameter("residual_scale_logit", None)
        self.hierarchy_source_adj: torch.Tensor | None = None
        self.similarity_source_adj: torch.Tensor | None = None
        self._last_source_names: tuple[str, ...] = ()
        self._last_source_gates: torch.Tensor | None = None
        self._configured_graph_sources: tuple[str, ...] | None = None

    def set_graph_config(
        self,
        graph_config: dict,
        base_adj=None,
        base_edge_index=None,
        base_edge_weight=None,
    ) -> None:
        lagtcn_config = dict(graph_config)
        super().set_graph_config(
            lagtcn_config,
            base_adj=base_adj,
            base_edge_index=base_edge_index,
            base_edge_weight=base_edge_weight,
        )
        active_sources = lagtcn_graph_sources(lagtcn_config.get("graph_mode", "H"))
        if self._configured_graph_sources is not None and self._configured_graph_sources != active_sources:
            raise RuntimeError(
                "A configured LAGTCN instance cannot change graph sources. "
                "Construct a new model instead."
            )
        for block in self.blocks:
            if block.uniform_source_fusion or len(active_sources) <= 1:
                block.source_logits.requires_grad_(False)
        self._configured_graph_sources = active_sources

    def set_static_graph_sources(self, hierarchy_adj=None, similarity_adj=None) -> None:
        """Attach the fixed hierarchy and similarity sources used by LAGTCN."""
        self.hierarchy_source_adj = self._validate_static_source(hierarchy_adj, "hierarchy")
        self.similarity_source_adj = self._validate_static_source(similarity_adj, "similarity")

    def _validate_static_source(self, adj, source_name: str) -> torch.Tensor | None:
        if adj is None:
            return None
        tensor = torch.as_tensor(adj, dtype=torch.float32).detach().clone()
        expected = (self.node_num, self.node_num)
        if tuple(tensor.shape) != expected:
            raise ValueError(
                f"LAGTCN {source_name} adjacency shape {tuple(tensor.shape)} != {expected}."
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"LAGTCN {source_name} adjacency contains NaN or Inf values.")
        return tensor

    def _patch_tokens(self, target_series: torch.Tensor) -> torch.Tensor:
        bsz, n_nodes, t_in = target_series.shape
        patches = target_series.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        _, _, n_patches, _ = patches.shape
        tokens = self.patch_proj(patches.reshape(bsz * n_nodes, n_patches, self.patch_len))
        tokens = tokens.reshape(bsz, n_nodes, n_patches, self.model_dim)
        tokens = tokens + self.pos_embedding[:, :, :n_patches, :].to(device=tokens.device, dtype=tokens.dtype)
        return self.input_dropout(tokens)

    def _prepare_static_source(
        self,
        adj: torch.Tensor | None,
        x: torch.Tensor,
        source_name: str,
    ) -> torch.Tensor:
        if adj is None:
            raise RuntimeError(
                f"LAGTCN graph mode requires the {source_name} source, but no adjacency was attached."
            )
        prepared = torch.clamp(adj.to(device=x.device, dtype=x.dtype), min=0.0).clone()
        if self.graph_config is None or self.graph_config.get("include_self_loops", True):
            prepared.fill_diagonal_(1.0)
        return _row_normalize(prepared)

    def _compute_samplewise_dynamic_adj(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x shape [B,N,F,T], got {tuple(x.shape)}")
        config = self.graph_config
        node_repr = x.reshape(x.shape[0], self.node_num, -1)
        sim = self._compute_similarity_matrix(
            node_repr,
            config["dynamic_sim_type"],
            use_abs=True,
        )
        threshold = config.get("dynamic_threshold")
        if threshold is None:
            raise ValueError(
                f"{FINAL_GRAPH_SOURCE_POLICY} requires dynamic_threshold for graph modes using D."
            )
        return build_threshold_similarity_adj_torch(
            sim,
            threshold=float(threshold),
            include_self_loops=config["include_self_loops"],
            use_weights=True,
        )

    def _graph_parts(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> list[tuple[str, torch.Tensor]]:
        del edge_index, edge_weight  # LAGTCN consumes the source matrices directly.
        graph_mode = "H" if self.graph_config is None else self.graph_config.get("graph_mode", "H")
        source_names = lagtcn_graph_sources(graph_mode)
        parts: list[tuple[str, torch.Tensor]] = []
        for source_name in source_names:
            if source_name == "identity":
                parts.append(("identity", torch.eye(self.node_num, device=x.device, dtype=x.dtype)))
            elif source_name == "hierarchy":
                parts.append((
                    "hierarchy",
                    self._prepare_static_source(self.hierarchy_source_adj, x, "hierarchy"),
                ))
            elif source_name == "similarity":
                parts.append((
                    "similarity",
                    self._prepare_static_source(self.similarity_source_adj, x, "similarity"),
                ))
            elif source_name == "adaptive":
                if not getattr(self, "use_adaptive", False):
                    raise RuntimeError("Adaptive source requested but adaptive graph construction is disabled.")
                parts.append((
                    "adaptive",
                    _row_normalize(torch.clamp(self._compute_adaptive_adj(), min=0.0)),
                ))
            elif source_name == "dynamic":
                if not getattr(self, "use_dynamic", False):
                    raise RuntimeError("Dynamic source requested but dynamic graph construction is disabled.")
                parts.append((
                    "dynamic",
                    _row_normalize(torch.clamp(self._compute_samplewise_dynamic_adj(x), min=0.0)),
                ))
            else:  # pragma: no cover - guarded by lagtcn_graph_sources()
                raise RuntimeError(f"Unsupported LAGTCN graph source {source_name!r}.")
        return [(name, adj.to(device=x.device, dtype=x.dtype)) for name, adj in parts]

    def get_graph_source_gates(self) -> list[dict[str, object]]:
        """Return block-specific independent gates for active graph sources."""
        graph_mode = "H" if self.graph_config is None else self.graph_config.get("graph_mode", "H")
        source_names = lagtcn_graph_sources(graph_mode)
        weights_by_block: list[dict[str, object]] = []
        for block_idx, block in enumerate(self.blocks, start=1):
            values = block.source_gates(source_names).detach().cpu().tolist()
            gate_map = {name: float(value) for name, value in zip(source_names, values)}
            weights_by_block.append({
                "block": block_idx,
                "gates": gate_map,
            })
        return weights_by_block

    def _level_embedding(self, n_nodes: int, device: torch.device) -> torch.Tensor:
        if self.hier_level_ids.numel() == n_nodes:
            level_ids = self.hier_level_ids.to(device=device)
        else:
            level_ids = torch.zeros(n_nodes, dtype=torch.long, device=device)
        level_ids = level_ids.clamp(min=0, max=self.level_embedding.num_embeddings - 1)
        return self.level_embedding(level_ids).unsqueeze(0)

    def residual_scale(self, reference: torch.Tensor | None = None) -> torch.Tensor:
        """Return the scalar residual gate used by the configured decoder."""
        if self.decoder_mode == "direct":
            value = 1.0
        elif self.residual_scale_mode == "unit":
            value = 1.0
        elif self.residual_scale_logit is not None:
            return torch.sigmoid(self.residual_scale_logit)
        else:
            value = self.residual_scale_init
        device = reference.device if reference is not None else self.forecast_head[-1].weight.device
        dtype = reference.dtype if reference is not None else self.forecast_head[-1].weight.dtype
        return torch.tensor(value, device=device, dtype=dtype)

    def _decoder_reference(self, target_series: torch.Tensor) -> torch.Tensor | None:
        """Construct the normalized-scale forecast anchor for residual decoding."""
        if self.decoder_mode == "direct":
            return None
        if self.decoder_mode == "persistence_residual":
            return target_series[:, :, -1:].expand(-1, -1, self.output_dim)
        t_in = target_series.shape[-1]
        if t_in < self.seasonal_lag:
            raise ValueError(
                "seasonal_residual received a history shorter than seasonal_lag: "
                f"{t_in} < {self.seasonal_lag}."
            )
        indices = t_in - self.seasonal_lag + (
            torch.arange(self.output_dim, device=target_series.device) % self.seasonal_lag
        )
        return target_series.index_select(dim=-1, index=indices)

    def get_decoder_metadata(self) -> dict[str, object]:
        """Return JSON-safe decoder settings and the current learned gate value."""
        return {
            "decoder_mode": self.decoder_mode,
            "residual_scale_mode": (
                "not_applicable" if self.decoder_mode == "direct" else self.residual_scale_mode
            ),
            "residual_scale_init": (
                None if self.decoder_mode == "direct" else self.residual_scale_init
            ),
            "residual_scale_effective": (
                None
                if self.decoder_mode == "direct"
                else float(self.residual_scale().detach().cpu().item())
            ),
            "seasonal_lag": self.seasonal_lag if self.decoder_mode == "seasonal_residual" else None,
        }

    def forward_normalized(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        bsz, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"LAGTCN expected T_in={self.num_timesteps_in}, got {t_in}")
        patch_tokens = self._patch_tokens(target_series)
        state = self.initial_state(target_series)
        node_ids = torch.arange(n_nodes, device=x.device)
        node_emb = self.node_embedding(node_ids).unsqueeze(0).expand(bsz, -1, -1)
        if self.use_level_awareness:
            level_emb = self._level_embedding(n_nodes, x.device).expand(bsz, -1, -1)
        else:
            level_emb = torch.zeros_like(node_emb)
        state = state + node_emb + level_emb
        graph_parts = self._graph_parts(x, edge_index, edge_weight)
        source_gates_by_block = []
        for block in self.blocks:
            state, source_gates = block(patch_tokens, state, graph_parts, node_emb, level_emb)
            source_gates_by_block.append(source_gates)
        self._last_source_names = tuple(name for name, _ in graph_parts)
        self._last_source_gates = (
            torch.stack(source_gates_by_block).detach().cpu()
            if source_gates_by_block
            else None
        )
        decoder_output = self.forecast_head(state)
        reference = self._decoder_reference(target_series)
        if reference is None:
            pred_norm = decoder_output
        else:
            pred_norm = reference + self.residual_scale(decoder_output) * decoder_output
        pred_norm = pred_norm.view(bsz, n_nodes, self.output_dim)
        if not bool(torch.isfinite(pred_norm).all()):
            raise FloatingPointError("LAGTCN produced a non-finite normalized prediction.")
        return pred_norm

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_norm = self.forward_normalized(x, edge_index, edge_weight)
        return self.prediction_from_normalized(pred_norm)
