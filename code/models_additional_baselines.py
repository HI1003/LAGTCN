import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, TransformerConv

from graph_sparsity import FINAL_GRAPH_SOURCE_POLICY, build_threshold_similarity_adj_torch
from models_base import BaseGCNGRUModel
from models_baselines_temporal import TemporalBaselineBase
from output_naming import LAGTCN_INFORMATIVE_GRAPH_SOURCES, lagtcn_graph_sources


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


class GraphWaveNetBaseline(_GraphBaselineBase):
    """Compact Graph WaveNet-style baseline with temporal dilated conv + adaptive graph."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        adaptive_emb_dim: int = 16,
        dropout: float = 0.2,
        stgnn_graph_source: str = "hybrid",
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
        self.channels = max(32, int(hidden_dim))
        self.num_blocks = max(1, int(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.start_conv = nn.Conv2d(1, self.channels, kernel_size=(1, 1))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.dilations = []

        dilation = 1
        for _ in range(self.num_blocks):
            self.filter_convs.append(
                nn.Conv2d(self.channels, self.channels, kernel_size=(1, 2), dilation=(1, dilation))
            )
            self.gate_convs.append(
                nn.Conv2d(self.channels, self.channels, kernel_size=(1, 2), dilation=(1, dilation))
            )
            self.residual_convs.append(nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1)))
            self.skip_convs.append(nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1)))
            self.dilations.append(dilation)
            dilation *= 2

        emb_dim = max(4, int(adaptive_emb_dim))
        self.node_emb1 = nn.Parameter(torch.randn(node_num, emb_dim) * 0.01)
        self.node_emb2 = nn.Parameter(torch.randn(node_num, emb_dim) * 0.01)
        self.fuse_alpha = nn.Parameter(torch.tensor(0.5))
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.output_dim),
        )

    def _adaptive_adj(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scores = torch.relu(self.node_emb1 @ self.node_emb2.T)
        adj = torch.softmax(scores, dim=1).to(device=device, dtype=dtype)
        adj = adj + torch.eye(adj.shape[0], device=device, dtype=dtype) * 1e-3
        return _row_normalize(adj)

    def get_adaptive_adjacency(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        param = self.node_emb1
        return self._adaptive_adj(
            device=device or param.device,
            dtype=dtype or param.dtype,
        )

    def _graph_mix(self, x: torch.Tensor, adj_static: torch.Tensor, adj_adaptive: torch.Tensor) -> torch.Tensor:
        # x: [S, C, N, T]
        source = self.stgnn_graph_source
        if source == "native":
            adj = adj_adaptive
        elif source == "project":
            adj = adj_static
        else:
            adj = 0.5 * adj_static + 0.5 * adj_adaptive
        x_graph = torch.einsum("nm,scmt->scnt", _row_normalize(adj), x)
        alpha = torch.sigmoid(self.fuse_alpha)
        return alpha * x + (1.0 - alpha) * x_graph

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = x[:, :, 0, :]  # [S, N, T]
        h = self.start_conv(target_series.unsqueeze(1))  # [S, C, N, T]

        adj_static = self._resolve_dense_adj(x, edge_index, edge_weight)
        adj_adaptive = self._adaptive_adj(device=h.device, dtype=h.dtype)

        skip_total = None
        for i in range(self.num_blocks):
            residual = h
            dilation = self.dilations[i]
            h_pad = F.pad(h, (dilation, 0, 0, 0))
            filt = torch.tanh(self.filter_convs[i](h_pad))
            gate = torch.sigmoid(self.gate_convs[i](h_pad))
            h = filt * gate
            h = self._graph_mix(h, adj_static, adj_adaptive)
            h = self.dropout(h)
            h = self.residual_convs[i](h) + residual

            skip_term = self.skip_convs[i](h)
            skip_total = skip_term if skip_total is None else (skip_total + skip_term)

        h = torch.relu(skip_total)
        last_state = h[:, :, :, -1].transpose(1, 2)  # [S, N, C]
        pred_norm = self.out_proj(last_state)
        pred_denorm = self.denormalize(pred_norm)
        pred_delog = self.inverse_log(pred_denorm)
        return pred_delog


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


class _ComplexTemporalModeFilter(nn.Module):
    """Individual reduced-order complex spectral filters from the TGC paper."""

    def __init__(
        self,
        node_num: int,
        feature_dim: int,
        num_modes: int,
        frequency_indices: torch.Tensor,
    ):
        super().__init__()
        self.node_num = int(node_num)
        self.feature_dim = int(feature_dim)
        self.num_modes = int(num_modes)
        self.register_buffer(
            "frequency_indices",
            frequency_indices.to(dtype=torch.long),
            persistent=True,
        )
        shape = (self.node_num, self.feature_dim, self.num_modes, self.num_modes)
        self.weight_real = nn.Parameter(torch.zeros(shape))
        self.weight_imag = nn.Parameter(torch.zeros(shape))
        identity = torch.eye(self.num_modes).view(1, 1, self.num_modes, self.num_modes)
        with torch.no_grad():
            self.weight_real.copy_(0.1 * identity.expand(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, nodes, time, features]
        spectrum = torch.fft.rfft(x, dim=2, norm="ortho")
        selected = spectrum.index_select(2, self.frequency_indices)
        weight = torch.complex(self.weight_real, self.weight_imag).to(spectrum.dtype)
        filtered = torch.einsum("bnsc,ncsr->bnrc", selected, weight)
        padded = torch.zeros_like(spectrum).index_copy(
            2, self.frequency_indices, filtered
        )
        return torch.fft.irfft(padded, n=x.shape[2], dim=2, norm="ortho")


class _TemporalGraphGegenConvBlock(nn.Module):
    """Canonical linear TGC block: GegenConv, two temporal FDMs, residual."""

    def __init__(
        self,
        node_num: int,
        feature_dim: int,
        num_timesteps_in: int,
        gegenbauer_alpha: float = 1.2,
        polynomial_degree: int = 4,
        num_modes: int = 5,
        trend_window: int = 24,
    ):
        super().__init__()
        if gegenbauer_alpha <= -0.5:
            raise ValueError("SpecTGNN Gegenbauer alpha must be greater than -0.5.")
        if polynomial_degree < 1:
            raise ValueError("SpecTGNN polynomial degree must be positive.")
        frequency_bins = int(num_timesteps_in) // 2 + 1
        if not 1 <= int(num_modes) <= frequency_bins:
            raise ValueError(
                f"SpecTGNN num_modes must be in [1, {frequency_bins}], got {num_modes}."
            )
        if not 1 <= int(trend_window) <= int(num_timesteps_in):
            raise ValueError(
                "SpecTGNN trend_window must be between 1 and num_timesteps_in."
            )

        self.node_num = int(node_num)
        self.feature_dim = int(feature_dim)
        self.num_timesteps_in = int(num_timesteps_in)
        self.gegenbauer_alpha = float(gegenbauer_alpha)
        self.polynomial_degree = int(polynomial_degree)
        self.num_modes = int(num_modes)
        self.trend_window = int(trend_window)

        self.graph_filter_coefficients = nn.Parameter(
            torch.zeros(self.polynomial_degree + 1, self.feature_dim)
        )
        with torch.no_grad():
            self.graph_filter_coefficients[0].fill_(1.0)

        coarse_indices = torch.randperm(frequency_bins)[: self.num_modes].sort().values
        fine_indices = torch.randperm(frequency_bins)[: self.num_modes].sort().values
        self.coarse_filter = _ComplexTemporalModeFilter(
            self.node_num, self.feature_dim, self.num_modes, coarse_indices
        )
        self.fine_filter = _ComplexTemporalModeFilter(
            self.node_num, self.feature_dim, self.num_modes, fine_indices
        )

    @staticmethod
    def _propagate(adjacency: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("nm,bmtc->bntc", adjacency, x)

    def _gegenbauer_graph_filter(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        alpha = self.gegenbauer_alpha
        polynomials = [x]
        p_prev = x
        p_curr = 2.0 * alpha * self._propagate(adjacency, x)
        polynomials.append(p_curr)
        for degree in range(2, self.polynomial_degree + 1):
            propagated = self._propagate(adjacency, p_curr)
            p_next = (
                2.0 * (degree + alpha - 1.0) * propagated
                - (degree + 2.0 * alpha - 2.0) * p_prev
            ) / float(degree)
            polynomials.append(p_next)
            p_prev, p_curr = p_curr, p_next

        output = torch.zeros_like(x)
        for degree, polynomial in enumerate(polynomials):
            coefficient = self.graph_filter_coefficients[degree].view(1, 1, 1, -1)
            output = output + coefficient * polynomial
        return output

    def _causal_trend(self, x: torch.Tensor) -> torch.Tensor:
        if self.trend_window == 1:
            return x
        trend = torch.zeros_like(x)
        if x.shape[2] < self.trend_window:
            return trend
        cumulative = torch.cumsum(x, dim=2)
        prefix = F.pad(
            cumulative[:, :, :-self.trend_window, :],
            (0, 0, 1, 0),
        )
        rolling = cumulative[:, :, self.trend_window - 1 :, :] - prefix
        trend[:, :, self.trend_window - 1 :, :] = rolling / float(self.trend_window)
        return trend

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph_filtered = self._gegenbauer_graph_filter(x, adjacency)
        coarse = self.coarse_filter(graph_filtered)
        trend = self._causal_trend(coarse)
        detail = coarse - trend
        fine = self.fine_filter(detail)
        return x + trend + fine


class DeepHGNNSpecTGNNBaseline(_GraphBaselineBase):
    """DeepHGNN with the strongest backbone reported in the source paper.

    The multivariate graph model follows the canonical linear Temporal Graph
    Gegenbauer Convolution (TGC): Gegenbauer graph filtering, coarse- and
    fine-grained DFT-domain filtering, residual blocks, and a direct FC
    multi-horizon readout. DeepHGNN then retains only bottom forecasts and
    reconstructs all hierarchy levels with the fixed summing matrix.

    Neither the DeepHGNN article nor the referenced SpecTGNN article exposes a
    verifiable official implementation. Metadata therefore identifies this as
    a paper-based reimplementation.
    """

    paper_doi = "10.1016/j.eswa.2025.127658"
    backbone_paper_arxiv = "2305.06587"
    backbone_variant = "SpecTGNN-TGC"
    implementation_status = "paper_reimplementation_no_official_code"
    prediction_role = "end_to_end_coherent"

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        sum_matrix,
        bottom_start_idx: int,
        hierarchical_loss_weight: float = 1.0,
        num_timesteps_in: int = 168,
        gegenbauer_alpha: float = 1.2,
        polynomial_degree: int = 4,
        num_modes: int = 5,
        trend_window: int = 24,
        stgnn_graph_source: str = "project",
    ):
        if str(stgnn_graph_source).strip().lower() not in {
            "project", "hierarchy", "hierarchy_graph", "project_graph"
        }:
            raise ValueError(
                "DeepHGNN-SpecTGNN uses the supplied hierarchy graph; "
                "stgnn_graph_source must be project."
            )
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
        self.num_timesteps_in = int(num_timesteps_in)
        self.gegenbauer_alpha = float(gegenbauer_alpha)
        self.polynomial_degree = int(polynomial_degree)
        self.num_modes = int(num_modes)
        self.trend_window = int(trend_window)
        self.spectral_feature_dim = int(input_dim)
        self.tgc_blocks = nn.ModuleList(
            [
                _TemporalGraphGegenConvBlock(
                    node_num=node_num,
                    feature_dim=self.spectral_feature_dim,
                    num_timesteps_in=self.num_timesteps_in,
                    gegenbauer_alpha=self.gegenbauer_alpha,
                    polynomial_degree=self.polynomial_degree,
                    num_modes=self.num_modes,
                    trend_window=self.trend_window,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.tgc_readout = nn.Linear(
            self.num_timesteps_in * self.spectral_feature_dim,
            int(output_dim),
        )

        matrix = torch.as_tensor(sum_matrix, dtype=torch.float32)
        if matrix.ndim != 2 or matrix.shape[0] != node_num:
            raise ValueError(
                f"DeepHGNN sum_matrix must have shape [{node_num}, B], got {tuple(matrix.shape)}"
            )
        if bool((matrix < 0).any()):
            raise ValueError("DeepHGNN requires a nonnegative summing matrix.")
        self.bottom_start_idx = int(bottom_start_idx)
        self.num_bottom_nodes = int(matrix.shape[1])
        if self.bottom_start_idx + self.num_bottom_nodes != node_num:
            raise ValueError(
                "DeepHGNN requires bottom nodes to be the final contiguous block: "
                f"bottom_start={self.bottom_start_idx}, B={self.num_bottom_nodes}, N={node_num}."
            )
        bottom_block = matrix[self.bottom_start_idx:, :]
        identity = torch.eye(self.num_bottom_nodes, dtype=matrix.dtype)
        if not torch.allclose(bottom_block, identity, atol=1e-7, rtol=0.0):
            raise ValueError("DeepHGNN requires the bottom block of sum_matrix to be identity.")
        if float(hierarchical_loss_weight) < 0:
            raise ValueError("hierarchical_loss_weight must be nonnegative.")
        self.hierarchical_loss_weight = float(hierarchical_loss_weight)
        self.register_buffer("deephgnn_sum_matrix", matrix)

    def _resolve_tgc_adjacency(
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
        adjacency = x.new_zeros((x.shape[1], x.shape[1]))
        adjacency[edge_index_use[0], edge_index_use[1]] = edge_weight_use.to(x.dtype)
        return torch.clamp(adjacency, min=0.0)

    @staticmethod
    def _symmetric_normalized_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        # TGC is defined on the symmetric normalized adjacency I - L_hat.
        adjacency = torch.maximum(adjacency, adjacency.T)
        adjacency = adjacency.clone()
        adjacency.fill_diagonal_(1.0)
        degree_inv_sqrt = adjacency.sum(dim=1).clamp_min(1e-8).pow(-0.5)
        return degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.dim() != 4 or x.shape[2] != self.spectral_feature_dim:
            raise ValueError(
                "DeepHGNN-SpecTGNN expected [batch,nodes,input_dim,time], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.num_timesteps_in:
            raise ValueError(
                f"DeepHGNN-SpecTGNN expected T={self.num_timesteps_in}, got {x.shape[-1]}."
            )
        adjacency = self._resolve_tgc_adjacency(x, edge_index, edge_weight)
        adjacency = self._symmetric_normalized_adjacency(adjacency)
        representation = x.permute(0, 1, 3, 2).contiguous()
        for block in self.tgc_blocks:
            representation = block(representation, adjacency)

        candidates_norm = self.tgc_readout(
            representation.reshape(
                representation.shape[0], representation.shape[1], -1
            )
        )
        candidates = self.inverse_log(self.denormalize(candidates_norm))
        bottom_forecasts = candidates[
            :, self.bottom_start_idx:self.bottom_start_idx + self.num_bottom_nodes, :
        ]
        return torch.einsum("nb,sbh->snh", self.deephgnn_sum_matrix, bottom_forecasts)

    def compute_training_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        criterion,
    ) -> torch.Tensor:
        """Bottom forecasting loss plus weighted aggregate hierarchy loss."""
        bottom_loss = criterion(
            y_pred[:, self.bottom_start_idx:, :],
            y_true[:, self.bottom_start_idx:, :],
        )
        if self.bottom_start_idx == 0 or self.hierarchical_loss_weight == 0:
            return bottom_loss
        hierarchy_loss = criterion(
            y_pred[:, :self.bottom_start_idx, :],
            y_true[:, :self.bottom_start_idx, :],
        )
        return bottom_loss + self.hierarchical_loss_weight * hierarchy_loss


class STGCNBaseline(_GraphBaselineBase):
    """Compact STGCN-style baseline with fixed project graph propagation."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        dropout: float = 0.2,
        stgnn_graph_source: str = "project",
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
        self.channels = max(32, int(hidden_dim))
        self.num_blocks = max(1, int(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.start_conv = nn.Conv2d(1, self.channels, kernel_size=(1, 1))
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dilations = []
        dilation = 1
        for _ in range(self.num_blocks):
            self.filter_convs.append(
                nn.Conv2d(self.channels, self.channels, kernel_size=(1, 3), dilation=(1, dilation))
            )
            self.gate_convs.append(
                nn.Conv2d(self.channels, self.channels, kernel_size=(1, 3), dilation=(1, dilation))
            )
            self.residual_convs.append(nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1)))
            self.norms.append(nn.BatchNorm2d(self.channels))
            self.dilations.append(dilation)
            dilation *= 2
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.channels),
            nn.Linear(self.channels, self.output_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = x[:, :, 0, :]  # [S, N, T]
        h = self.start_conv(target_series.unsqueeze(1))  # [S, C, N, T]
        adj = self._resolve_dense_adj(x, edge_index, edge_weight)

        for i in range(self.num_blocks):
            residual = h
            dilation = self.dilations[i]
            h_pad = F.pad(h, (2 * dilation, 0, 0, 0))
            filt = torch.tanh(self.filter_convs[i](h_pad))
            gate = torch.sigmoid(self.gate_convs[i](h_pad))
            h = filt * gate
            h = torch.einsum("nm,scmt->scnt", adj, h)
            h = self.dropout(h)
            h = self.residual_convs[i](h) + residual
            h = self.norms[i](h)

        last_state = h[:, :, :, -1].transpose(1, 2)  # [S, N, C]
        pred_norm = self.out_proj(last_state)
        pred_denorm = self.denormalize(pred_norm)
        return self.inverse_log(pred_denorm)


class _AVWGraphConv(nn.Module):
    """Node-adaptive graph convolution used by the full AGCRN baseline."""

    def __init__(self, input_dim: int, output_dim: int, cheb_k: int = 2, embed_dim: int = 16):
        super().__init__()
        self.cheb_k = max(1, int(cheb_k))
        self.weights_pool = nn.Parameter(torch.empty(int(embed_dim), self.cheb_k, int(input_dim), int(output_dim)))
        self.bias_pool = nn.Parameter(torch.empty(int(embed_dim), int(output_dim)))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C], adj: [N, N], node_emb: [N, E]
        supports = [torch.eye(adj.shape[0], device=adj.device, dtype=adj.dtype)]
        if self.cheb_k > 1:
            supports.append(adj)
            for _ in range(2, self.cheb_k):
                supports.append(torch.matmul(2.0 * adj, supports[-1]) - supports[-2])
        support = torch.stack(supports[: self.cheb_k], dim=0)  # [K, N, N]
        x_g = torch.einsum("knm,bmc->bknc", support, x)
        weights = torch.einsum("ne,ekco->nkco", node_emb, self.weights_pool)
        bias = torch.einsum("ne,eo->no", node_emb, self.bias_pool)
        return torch.einsum("bknc,nkco->bno", x_g, weights) + bias


class _AGCRNCell(nn.Module):
    """AGCRN recurrent cell with node-adaptive graph convolutional gates."""

    def __init__(self, input_dim: int, hidden_dim: int, cheb_k: int, embed_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gate = _AVWGraphConv(input_dim + hidden_dim, 2 * hidden_dim, cheb_k=cheb_k, embed_dim=embed_dim)
        self.update = _AVWGraphConv(input_dim + hidden_dim, hidden_dim, cheb_k=cheb_k, embed_dim=embed_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        hidden: torch.Tensor,
        adj: torch.Tensor,
        node_emb: torch.Tensor,
    ) -> torch.Tensor:
        gate_input = torch.cat([x_t, hidden], dim=-1)
        z_gate, r_gate = torch.sigmoid(self.gate(gate_input, adj, node_emb)).chunk(2, dim=-1)
        cand_input = torch.cat([x_t, r_gate * hidden], dim=-1)
        h_tilde = torch.tanh(self.update(cand_input, adj, node_emb))
        return z_gate * hidden + (1.0 - z_gate) * h_tilde


class AGCRNBaseline(_GraphBaselineBase):
    """Full AGCRN-style baseline with DAGG, NAPL, and stacked AGCRN cells."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        adaptive_emb_dim: int = 16,
        support_order: int = 2,
        stgnn_graph_source: str = "hybrid",
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
        self.state_dim = max(32, int(hidden_dim))
        self.embed_dim = max(4, int(adaptive_emb_dim))
        self.cheb_k = max(1, int(support_order) + 1)
        self.num_agcrn_layers = max(1, int(num_layers))
        self.node_emb = nn.Parameter(torch.randn(node_num, self.embed_dim) * 0.01)
        self.cells = nn.ModuleList(
            _AGCRNCell(
                input_dim=1 if layer_idx == 0 else self.state_dim,
                hidden_dim=self.state_dim,
                cheb_k=self.cheb_k,
                embed_dim=self.embed_dim,
            )
            for layer_idx in range(self.num_agcrn_layers)
        )
        self.output_norm = nn.LayerNorm(self.state_dim)
        self.readout = nn.Linear(self.state_dim, self.output_dim)

    def _adaptive_adj(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        scores = torch.relu(self.node_emb @ self.node_emb.T)
        adj = torch.softmax(scores, dim=1).to(device=device, dtype=dtype)
        adj = adj + torch.eye(adj.shape[0], device=device, dtype=dtype) * 1e-3
        return _row_normalize(adj)

    def get_adaptive_adjacency(self, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        param = self.node_emb
        return self._adaptive_adj(
            device=device or param.device,
            dtype=dtype or param.dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = x[:, :, 0, :]  # [B, N, T]
        ssz, n_nodes, t_in = target_series.shape

        project_adj = self._resolve_dense_adj(x, edge_index, edge_weight)
        native_adj = self._adaptive_adj(device=x.device, dtype=x.dtype)
        adj = self._combine_project_and_native_adj(project_adj, native_adj)

        states = [
            x.new_zeros((ssz, n_nodes, self.state_dim))
            for _ in range(self.num_agcrn_layers)
        ]
        node_emb = self.node_emb.to(device=x.device, dtype=x.dtype)
        for t in range(t_in):
            layer_input = target_series[:, :, t : t + 1]
            for layer_idx, cell in enumerate(self.cells):
                states[layer_idx] = cell(layer_input, states[layer_idx], adj, node_emb)
                layer_input = states[layer_idx]

        pred_norm = self.readout(self.output_norm(states[-1]))
        pred_denorm = self.denormalize(pred_norm)
        return self.inverse_log(pred_denorm)


class _GraphNativeTemporalBase(_GraphBaselineBase):
    """Shared GNN residual block for graph-native standalone temporal models."""

    def __init__(
        self,
        node_num: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        global_min: float,
        global_max: float,
        graph_dim: int,
        gnn_type: str = "gcn",
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
        self.graph_dim = int(graph_dim)
        self.graph_gnn_type = str(gnn_type).lower()
        if self.graph_gnn_type not in {"none", "gcn", "gatv2", "graphsage", "transformer"}:
            raise ValueError(f"Unsupported graph-native gnn_type: {gnn_type}")

        if self.graph_gnn_type == "none":
            self.graph_layer = None
        elif self.graph_gnn_type == "gcn":
            self.graph_layer = GCNConv(self.graph_dim, self.graph_dim)
        elif self.graph_gnn_type == "gatv2":
            self.graph_layer = GATv2Conv(self.graph_dim, self.graph_dim, heads=1)
        elif self.graph_gnn_type == "graphsage":
            self.graph_layer = SAGEConv(self.graph_dim, self.graph_dim)
        else:
            self.graph_layer = TransformerConv(self.graph_dim, self.graph_dim, heads=1)

        self.graph_norm = nn.LayerNorm(self.graph_dim)
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_gate = nn.Linear(self.graph_dim * 2, self.graph_dim)
        self.graph_residual_scale = nn.Parameter(torch.tensor(0.1))

    def _graph_residual_features(
        self,
        x: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        # node_features: [S, N, D]
        if self.graph_layer is None:
            return node_features.new_zeros(node_features.shape)

        ssz, n_nodes, feat_dim = node_features.shape
        if feat_dim != self.graph_dim:
            raise ValueError(f"Expected graph feature dim={self.graph_dim}, got {feat_dim}")

        edge_index_use, edge_weight_use = self._resolve_graph_edges(x, edge_index, edge_weight)
        edge_index_batch, edge_weight_batch = self._expand_edges_for_batch(
            edge_index_use,
            edge_weight_use,
            batch_size=ssz,
            num_nodes=n_nodes,
        )
        flat = node_features.reshape(ssz * n_nodes, feat_dim)
        if self.graph_gnn_type == "gcn":
            graph_out = self.graph_layer(flat, edge_index_batch, edge_weight=edge_weight_batch)
        else:
            graph_out = self.graph_layer(flat, edge_index_batch)
        graph_out = self.graph_dropout(self.graph_norm(graph_out)).view(ssz, n_nodes, feat_dim)
        gate = torch.sigmoid(self.graph_gate(torch.cat([node_features, graph_out], dim=-1)))
        return gate * graph_out


class GraphDLinearBaseline(_GraphNativeTemporalBase):
    """DLinear with a native GNN residual over node-level temporal summaries."""

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
        gnn_type: str = "gcn",
        moving_avg_window: int = 25,
        dropout: float = 0.1,
    ):
        graph_dim = max(32, int(hidden_dim))
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
            graph_dim=graph_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        kernel = max(1, int(moving_avg_window))
        if kernel % 2 == 0:
            kernel += 1
        self.moving_avg_window = kernel

        self.trend_linear = nn.Linear(self.num_timesteps_in, self.output_dim)
        self.seasonal_linear = nn.Linear(self.num_timesteps_in, self.output_dim)
        self.trend_feature = nn.Linear(self.num_timesteps_in, self.graph_dim)
        self.seasonal_feature = nn.Linear(self.num_timesteps_in, self.graph_dim)
        self.graph_head = nn.Sequential(nn.LayerNorm(self.graph_dim), nn.Linear(self.graph_dim, self.output_dim))
        nn.init.constant_(self.trend_linear.weight, 1.0 / self.num_timesteps_in)
        nn.init.constant_(self.seasonal_linear.weight, 1.0 / self.num_timesteps_in)
        nn.init.zeros_(self.trend_linear.bias)
        nn.init.zeros_(self.seasonal_linear.bias)

    def _moving_average(self, series: torch.Tensor) -> torch.Tensor:
        pad = self.moving_avg_window // 2
        x = series.unsqueeze(1)
        x_pad = F.pad(x, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.moving_avg_window, stride=1)
        return trend.squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        ssz, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"GraphDLinear expected T_in={self.num_timesteps_in}, got {t_in}")

        flat = target_series.reshape(ssz * n_nodes, t_in)
        trend = self._moving_average(flat)
        seasonal = flat - trend
        base_pred = self.trend_linear(trend) + self.seasonal_linear(seasonal)

        features = self.trend_feature(trend) + self.seasonal_feature(seasonal)
        features = features.view(ssz, n_nodes, self.graph_dim)
        graph_delta = self.graph_head(self._graph_residual_features(x, features, edge_index, edge_weight))
        pred_norm = base_pred.view(ssz, n_nodes, self.output_dim) + self.graph_residual_scale * graph_delta

        pred_denorm = self.denormalize(pred_norm)
        return self.inverse_log(pred_denorm)


class GraphPatchTSTBaseline(_GraphNativeTemporalBase):
    """PatchTST with a native GNN residual over patch-token summaries."""

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
        gnn_type: str = "gcn",
        patch_len: int = 8,
        patch_stride: int = 4,
        dropout: float = 0.1,
    ):
        self.patch_dim = max(32, int(hidden_dim))
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
            graph_dim=self.patch_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        self.patch_len = max(1, min(int(patch_len), self.num_timesteps_in))
        self.patch_stride = max(1, int(patch_stride))
        max_num_patches = 1 + max(0, (self.num_timesteps_in - self.patch_len) // self.patch_stride)
        self.patch_proj = nn.Linear(self.patch_len, self.patch_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_num_patches, self.patch_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)

        nhead = _choose_nhead(self.patch_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.patch_dim,
            nhead=nhead,
            dim_feedforward=self.patch_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.head = nn.Sequential(nn.LayerNorm(self.patch_dim), nn.Linear(self.patch_dim, self.output_dim))
        self.graph_head = nn.Sequential(nn.LayerNorm(self.patch_dim), nn.Linear(self.patch_dim, self.output_dim))

    def _get_positional_embedding(self, num_patches: int, device: torch.device) -> torch.Tensor:
        if num_patches <= self.pos_embedding.shape[1]:
            return self.pos_embedding[:, :num_patches]
        extra = torch.zeros(
            1,
            num_patches - self.pos_embedding.shape[1],
            self.patch_dim,
            device=device,
            dtype=self.pos_embedding.dtype,
        )
        return torch.cat([self.pos_embedding, extra], dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        ssz, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"GraphPatchTST expected T_in={self.num_timesteps_in}, got {t_in}")

        patches = target_series.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        _, _, num_patches, _ = patches.shape
        patch_tokens = patches.reshape(ssz * n_nodes, num_patches, self.patch_len)
        patch_tokens = self.patch_proj(patch_tokens)
        patch_tokens = patch_tokens + self._get_positional_embedding(num_patches, patch_tokens.device)
        patch_tokens = self.input_dropout(patch_tokens)

        encoded = self.encoder(patch_tokens)
        pooled = encoded.mean(dim=1).view(ssz, n_nodes, self.patch_dim)
        base_pred = self.head(pooled)
        graph_delta = self.graph_head(self._graph_residual_features(x, pooled, edge_index, edge_weight))
        pred_norm = base_pred + self.graph_residual_scale * graph_delta

        pred_denorm = self.denormalize(pred_norm)
        return self.inverse_log(pred_denorm)


class GraphTemporalAdapterBaseline(_GraphBaselineBase):
    """PatchTST/iTransformer encoder with a fixed-graph adapter.

    Data-driven multi-source fusion is implemented only by LAGTCN.
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
        num_timesteps_in: int,
        temporal_backbone: str = "patchtst",
        gnn_type: str = "gcn",
        patch_len: int = 8,
        patch_stride: int = 4,
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
        self.model_dim = max(64, int(hidden_dim))
        self.temporal_backbone = str(temporal_backbone).lower()
        if self.temporal_backbone not in {"patchtst", "itransformer"}:
            raise ValueError("GRAPH_ADAPTER requires temporal_backbone='patchtst' or 'itransformer'.")

        self.graph_gnn_type = str(gnn_type).lower()
        if self.graph_gnn_type not in {"none", "gcn", "gatv2", "graphsage", "transformer"}:
            raise ValueError(f"Unsupported GRAPH_ADAPTER gnn_type: {gnn_type}")

        self.patch_len = max(1, min(int(patch_len), self.num_timesteps_in))
        self.patch_stride = max(1, int(patch_stride))
        max_num_patches = 1 + max(0, (self.num_timesteps_in - self.patch_len) // self.patch_stride)
        self.patch_proj = nn.Linear(self.patch_len, self.model_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_num_patches, self.model_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)

        self.token_proj = nn.Linear(self.num_timesteps_in, self.model_dim)
        self.input_dropout = nn.Dropout(dropout)
        nhead = _choose_nhead(self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))

        self.base_head = nn.Sequential(nn.LayerNorm(self.model_dim), nn.Linear(self.model_dim, self.output_dim))
        self.graph_head = nn.Sequential(nn.LayerNorm(self.model_dim), nn.Linear(self.model_dim, self.output_dim))
        for head in (self.base_head, self.graph_head):
            nn.init.normal_(head[-1].weight, mean=0.0, std=1e-4)
            nn.init.zeros_(head[-1].bias)

        self.graph_part_order = ("base",)
        self.part_proj = nn.ModuleDict({
            "base": nn.Linear(self.model_dim, self.model_dim)
        })
        self.sage_proj = nn.ModuleDict({
            "base": nn.Linear(self.model_dim * 2, self.model_dim)
        })
        self.attn_q = nn.Linear(self.model_dim, self.model_dim)
        self.attn_k = nn.Linear(self.model_dim, self.model_dim)
        self.attn_v = nn.Linear(self.model_dim, self.model_dim)
        self.attn_out = nn.Linear(self.model_dim, self.model_dim)
        self.graph_norm = nn.LayerNorm(self.model_dim)
        self.graph_dropout = nn.Dropout(dropout)
        self.graph_gate = nn.Linear(self.model_dim * 2, self.model_dim)
        self.graph_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.forecast_residual_scale = 0.1

    def _get_positional_embedding(self, num_patches: int, device: torch.device) -> torch.Tensor:
        if num_patches <= self.pos_embedding.shape[1]:
            return self.pos_embedding[:, :num_patches]
        extra = torch.zeros(
            1,
            num_patches - self.pos_embedding.shape[1],
            self.model_dim,
            device=device,
            dtype=self.pos_embedding.dtype,
        )
        return torch.cat([self.pos_embedding, extra], dim=1)

    def _encode_patchtst(self, target_series: torch.Tensor) -> torch.Tensor:
        ssz, n_nodes, t_in = target_series.shape
        patches = target_series.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        _, _, num_patches, _ = patches.shape
        patch_tokens = patches.reshape(ssz * n_nodes, num_patches, self.patch_len)
        patch_tokens = self.patch_proj(patch_tokens)
        patch_tokens = patch_tokens + self._get_positional_embedding(num_patches, patch_tokens.device)
        patch_tokens = self.input_dropout(patch_tokens)
        encoded = self.encoder(patch_tokens)
        return encoded.mean(dim=1).view(ssz, n_nodes, self.model_dim)

    def _encode_itransformer(self, target_series: torch.Tensor) -> torch.Tensor:
        tokens = self.token_proj(target_series)
        tokens = self.input_dropout(tokens)
        return self.encoder(tokens)

    def _temporal_features(self, target_series: torch.Tensor) -> torch.Tensor:
        if self.temporal_backbone == "patchtst":
            return self._encode_patchtst(target_series)
        return self._encode_itransformer(target_series)

    def _dense_base_adj(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        n_nodes = x.shape[1]
        if self.base_adj is not None:
            adj = self.base_adj.to(device=x.device, dtype=x.dtype).clone()
        else:
            adj = x.new_zeros((n_nodes, n_nodes))
            if edge_weight is None:
                weights = torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
            else:
                weights = edge_weight.to(device=x.device, dtype=x.dtype)
            adj[edge_index[0].to(x.device), edge_index[1].to(x.device)] = weights
        adj = torch.clamp(adj, min=0.0)
        if self.graph_config is None or self.graph_config.get("include_self_loops", True):
            adj.fill_diagonal_(1.0)
        return _row_normalize(adj)

    def _graph_parts(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> list[tuple[str, torch.Tensor]]:
        adj = self._dense_base_adj(x, edge_index, edge_weight)
        return [("base", adj.to(device=x.device, dtype=x.dtype))]

    def _apply_graph_operator(self, features: torch.Tensor, adj: torch.Tensor, name: str) -> torch.Tensor:
        if self.graph_gnn_type == "none":
            return torch.zeros_like(features)

        if self.graph_gnn_type == "graphsage":
            neigh = torch.einsum("nm,bmd->bnd", adj, features)
            return self.sage_proj[name](torch.cat([features, neigh], dim=-1))

        if self.graph_gnn_type in {"gatv2", "transformer"}:
            q = self.attn_q(features)
            k = self.attn_k(features)
            v = self.attn_v(features)
            scores = torch.einsum("bnd,bmd->bnm", q, k) / (self.model_dim ** 0.5)
            edge_scores = adj.clamp_min(1e-8).log().unsqueeze(0)
            mask = adj <= 1e-12
            scores = (scores + edge_scores).masked_fill(mask.unsqueeze(0), torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1)
            msg = torch.einsum("bnm,bmd->bnd", weights, v)
            return self.attn_out(msg)

        msg = torch.einsum("nm,bmd->bnd", adj, features)
        return self.part_proj[name](msg)

    def _graph_adapter(
        self,
        x: torch.Tensor,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None,
    ) -> torch.Tensor:
        _, adj = self._graph_parts(x, edge_index, edge_weight)[0]
        mixed = self._apply_graph_operator(features, adj, "base")
        mixed = self.graph_dropout(self.graph_norm(mixed))
        gate = torch.sigmoid(self.graph_gate(torch.cat([features, mixed], dim=-1)))
        return gate * mixed

    def _clamp_normalized_forecast(self, pred_norm: torch.Tensor) -> torch.Tensor:
        if not self.use_log:
            return pred_norm
        if self.norm_method == "zscore":
            mean = self.norm_mean if self.norm_mean is not None else 0.0
            std = self.norm_std if self.norm_std not in (None, 0.0) else 1.0
            log_min = 0.0
            log_max = mean + 6.0 * std
            norm_min = (log_min - mean) / std
            norm_max = (log_max - mean) / std
            return pred_norm.clamp(min=norm_min, max=norm_max)
        if self.norm_min is not None and self.norm_max is not None:
            log_min = 0.0
            log_max = self.norm_max
            norm_min = (log_min - self.norm_min) / (self.norm_max - self.norm_min)
            norm_max = (log_max - self.norm_min) / (self.norm_max - self.norm_min)
            return pred_norm.clamp(min=norm_min, max=norm_max)
        return pred_norm

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        ssz, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"GRAPH_ADAPTER expected T_in={self.num_timesteps_in}, got {t_in}")

        features = self._temporal_features(target_series)
        adapter_features = self._graph_adapter(x, features, edge_index, edge_weight)
        base_delta = self.base_head(features)
        graph_delta = self.graph_head(adapter_features)
        last_value = target_series[:, :, -1:].expand(-1, -1, self.output_dim)
        pred_norm = last_value + self.forecast_residual_scale * (
            base_delta + self.graph_residual_scale * graph_delta
        )
        pred_norm = self._clamp_normalized_forecast(pred_norm.view(ssz, n_nodes, self.output_dim))
        pred_denorm = self.denormalize(pred_norm)
        return self.inverse_log(pred_denorm)


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
        attended, _ = self.self_attention(
            x, x, x, need_weights=False
        )
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
        self.n_heads = _choose_nhead(self.model_dim)

        # DataEmbedding_inverted without time-covariate tokens: each node's full
        # history is a variate token, exactly as in the target-only official path.
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
            raise ValueError(f"iTransformer expected T_in={self.num_timesteps_in}, got {t_in}")

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


class GraphITransformerBaseline(_GraphNativeTemporalBase):
    """iTransformer with a native GNN residual over inverted node tokens."""

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
        gnn_type: str = "gcn",
        dropout: float = 0.1,
    ):
        self.model_dim = max(64, int(hidden_dim))
        super().__init__(
            node_num=node_num,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            global_min=global_min,
            global_max=global_max,
            graph_dim=self.model_dim,
            gnn_type=gnn_type,
            dropout=dropout,
        )
        self.num_timesteps_in = int(num_timesteps_in)
        self.token_proj = nn.Linear(self.num_timesteps_in, self.model_dim)
        nhead = _choose_nhead(self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=nhead,
            dim_feedforward=self.model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.head = nn.Sequential(nn.LayerNorm(self.model_dim), nn.Linear(self.model_dim, self.output_dim))
        self.graph_head = nn.Sequential(nn.LayerNorm(self.model_dim), nn.Linear(self.model_dim, self.output_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_series = self._extract_target_series(x)
        ssz, n_nodes, t_in = target_series.shape
        if t_in != self.num_timesteps_in:
            raise ValueError(f"GraphiTransformer expected T_in={self.num_timesteps_in}, got {t_in}")

        tokens = self.token_proj(target_series)
        encoded = self.encoder(tokens)
        base_pred = self.head(encoded).view(ssz, n_nodes, self.output_dim)
        graph_delta = self.graph_head(self._graph_residual_features(x, encoded, edge_index, edge_weight))
        pred_norm = base_pred + self.graph_residual_scale * graph_delta

        pred_denorm = self.denormalize(pred_norm)
        return self.inverse_log(pred_denorm)
