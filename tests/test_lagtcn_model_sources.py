from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


def _install_torch_geometric_stub_if_missing() -> None:
    try:
        import torch_geometric  # noqa: F401
        return
    except ImportError:
        pass

    class _UnusedConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x, edge_index, *args, **kwargs):
            return x

    def dense_to_sparse(adj: torch.Tensor):
        indices = torch.nonzero(adj, as_tuple=False).t().contiguous()
        values = adj[indices[0], indices[1]]
        return indices, values

    package = types.ModuleType("torch_geometric")
    nn_module = types.ModuleType("torch_geometric.nn")
    utils_module = types.ModuleType("torch_geometric.utils")
    for name in ("GATv2Conv", "GCNConv", "SAGEConv", "TransformerConv"):
        setattr(nn_module, name, _UnusedConv)
    utils_module.dense_to_sparse = dense_to_sparse
    package.nn = nn_module
    package.utils = utils_module
    sys.modules["torch_geometric"] = package
    sys.modules["torch_geometric.nn"] = nn_module
    sys.modules["torch_geometric.utils"] = utils_module


_install_torch_geometric_stub_if_missing()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models_additional_baselines import LAGTCNBaseline
from graph_sparsity import FINAL_GRAPH_SOURCE_POLICY


class LAGTCNIndependentSourceTest(unittest.TestCase):
    def _make_model(self, graph_mode: str) -> LAGTCNBaseline:
        model = LAGTCNBaseline(
            node_num=4,
            input_dim=1,
            hidden_dim=16,
            output_dim=2,
            num_layers=2,
            global_min=0.0,
            global_max=1.0,
            num_timesteps_in=12,
            patch_len=4,
            patch_stride=2,
            dropout=0.0,
            hop_order=2,
        )
        config = {
            "graph_mode": graph_mode,
            "sim_type": "cosine",
            "adaptive_sim_type": "cosine",
            "dynamic_sim_type": "cosine",
            "adaptive_use_abs": False,
            "dynamic_use_abs": True,
            "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
            "adaptive_top_k": 2,
            "dynamic_threshold": 0.5,
            "adaptive_emb_dim": 4,
            "include_self_loops": True,
        }
        identity = torch.eye(4)
        model.set_graph_config(config, base_adj=identity)
        hierarchy = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )
        similarity = torch.tensor(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
            ]
        )
        model.set_static_graph_sources(hierarchy_adj=hierarchy, similarity_adj=similarity)
        model.set_norm_params({
            "norm_method": "minmax",
            "min": 0.0,
            "max": 1.0,
            "use_log": False,
        })
        return model

    def test_full_mode_uses_four_independent_sources(self) -> None:
        torch.manual_seed(7)
        model = self._make_model("H+S+A+D")
        x = torch.rand(3, 4, 1, 12)
        edge_index = torch.arange(4).repeat(2, 1)
        parts = model._graph_parts(x, edge_index, None)
        self.assertEqual(
            tuple(name for name, _ in parts),
            ("hierarchy", "similarity", "adaptive", "dynamic"),
        )
        for name, adj in parts:
            expected_shape = (3, 4, 4) if name == "dynamic" else (4, 4)
            self.assertEqual(tuple(adj.shape), expected_shape)
            self.assertTrue(torch.allclose(adj.sum(dim=-1), torch.ones_like(adj.sum(dim=-1)), atol=1e-5))

        prediction = model(x, edge_index)
        self.assertEqual(tuple(prediction.shape), (3, 4, 2))
        self.assertTrue(torch.isfinite(prediction).all())

        gates = model.get_graph_source_gates()
        self.assertEqual(len(gates), 2)
        for block_gates in gates:
            self.assertEqual(
                tuple(block_gates["gates"]),
                ("hierarchy", "similarity", "adaptive", "dynamic"),
            )
            self.assertAlmostEqual(block_gates["gates"]["hierarchy"], 1.0, places=6)
            for source_name in ("similarity", "adaptive", "dynamic"):
                self.assertAlmostEqual(block_gates["gates"][source_name], 0.5, places=6)

        prediction.mean().backward()
        for block in model.blocks:
            self.assertIsNotNone(block.source_logits.grad)
            self.assertIsNotNone(block.graph_mix.weight.grad)

    def test_identity_adjacency_produces_zero_relation_message(self) -> None:
        block = self._make_model("I").blocks[0]
        features = torch.randn(2, 4, block.model_dim)
        message = block.graph_update(features, torch.eye(4), "identity")
        self.assertTrue(torch.equal(message, torch.zeros_like(message)))

    def test_adaptive_and_dynamic_modes_exclude_identity(self) -> None:
        x = torch.rand(2, 4, 1, 12)
        edge_index = torch.arange(4).repeat(2, 1)
        for graph_mode, expected_name in (("A", "adaptive"), ("D", "dynamic")):
            with self.subTest(graph_mode=graph_mode):
                model = self._make_model(graph_mode)
                names = tuple(name for name, _ in model._graph_parts(x, edge_index, None))
                self.assertEqual(names, (expected_name,))

    def test_dynamic_graph_is_independent_of_batch_companions(self) -> None:
        torch.manual_seed(11)
        model = self._make_model("D").eval()
        anchor = torch.rand(1, 4, 1, 12)
        companion = torch.rand(1, 4, 1, 12) * 5.0
        edge_index = torch.arange(4).repeat(2, 1)

        adj_alone = model._graph_parts(anchor, edge_index, None)[0][1]
        adj_batched = model._graph_parts(torch.cat([anchor, companion]), edge_index, None)[0][1]
        self.assertEqual(tuple(adj_alone.shape), (1, 4, 4))
        self.assertEqual(tuple(adj_batched.shape), (2, 4, 4))
        self.assertTrue(torch.allclose(adj_alone[0], adj_batched[0], atol=1e-6))

        with torch.no_grad():
            pred_alone = model(anchor, edge_index)
            pred_batched = model(torch.cat([anchor, companion]), edge_index)[:1]
        self.assertTrue(torch.allclose(pred_alone, pred_batched, atol=1e-6))

    def test_identity_is_a_standalone_anchor(self) -> None:
        model = self._make_model("I")
        x = torch.rand(2, 4, 1, 12)
        edge_index = torch.arange(4).repeat(2, 1)
        parts = model._graph_parts(x, edge_index, None)
        self.assertEqual(tuple(name for name, _ in parts), ("identity",))
        self.assertTrue(torch.equal(parts[0][1], torch.eye(4)))


if __name__ == "__main__":
    unittest.main()
