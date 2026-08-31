from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from models_additional_baselines import DeepHGNNSpecTGNNBaseline


class DeepHGNNTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.S = np.array([
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ], dtype=np.float32)
        self.model = DeepHGNNSpecTGNNBaseline(
            node_num=5,
            input_dim=1,
            hidden_dim=8,
            output_dim=4,
            num_layers=1,
            global_min=0.0,
            global_max=1.0,
            sum_matrix=self.S,
            bottom_start_idx=2,
            hierarchical_loss_weight=0.5,
            num_timesteps_in=12,
            gegenbauer_alpha=1.2,
            polynomial_degree=4,
            num_modes=3,
            trend_window=4,
            stgnn_graph_source="project",
        )
        self.model.set_graph_config(
            {
                "graph_mode": "H",
                "stgnn_graph_source": "project",
                "include_self_loops": True,
                "native_top_k": 4,
                "graph_sparsity_policy": "source_specific_threshold_topk_v2",
            },
            base_adj=torch.ones(5, 5),
        )

    def test_forward_is_coherent_by_construction(self):
        x = torch.rand(3, 5, 1, 12)
        edge_index = torch.nonzero(torch.ones(5, 5), as_tuple=False).T
        pred = self.model(x, edge_index)
        self.assertEqual(tuple(pred.shape), (3, 5, 4))
        bottom = pred[:, 2:, :]
        rebuilt = torch.einsum("nb,sbh->snh", torch.tensor(self.S), bottom)
        self.assertTrue(torch.allclose(pred, rebuilt, atol=1e-6, rtol=0.0))

    def test_tgc_paper_components_receive_finite_gradients(self):
        self.assertEqual(self.model.backbone_variant, "SpecTGNN-TGC")
        block = self.model.tgc_blocks[0]
        self.assertEqual(block.polynomial_degree, 4)
        self.assertEqual(block.num_modes, 3)
        self.assertEqual(block.coarse_filter.frequency_indices.numel(), 3)
        self.assertEqual(block.fine_filter.frequency_indices.numel(), 3)
        # K+1 graph coefficients + four real/imaginary N*S*S filters
        # + a shared direct T-to-H readout. No inherited GRU/MLP parameters.
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 237)
        parameter_names = {name for name, _ in self.model.named_parameters()}
        self.assertFalse(any(name.startswith("node_encoder") for name in parameter_names))
        self.assertFalse(any(name.startswith("temporal_encoder") for name in parameter_names))

        x = torch.rand(2, 5, 1, 12)
        edge_index = torch.nonzero(torch.ones(5, 5), as_tuple=False).T
        loss = self.model(x, edge_index).square().mean()
        loss.backward()
        checked = (
            block.graph_filter_coefficients,
            block.coarse_filter.weight_real,
            block.coarse_filter.weight_imag,
            block.fine_filter.weight_real,
            self.model.tgc_readout.weight,
        )
        for parameter in checked:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_hierarchical_loss_matches_definition(self):
        criterion = nn.SmoothL1Loss()
        pred = torch.arange(40, dtype=torch.float32).view(2, 5, 4)
        true = pred + 1.0
        expected = criterion(pred[:, 2:, :], true[:, 2:, :])
        expected = expected + 0.5 * criterion(pred[:, :2, :], true[:, :2, :])
        actual = self.model.compute_training_loss(pred, true, criterion)
        self.assertTrue(torch.allclose(actual, expected))

    def test_invalid_bottom_block_rejected(self):
        bad = self.S.copy()
        bad[2, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "bottom block"):
            DeepHGNNSpecTGNNBaseline(
                5, 1, 8, 4, 1, 0.0, 1.0,
                sum_matrix=bad, bottom_start_idx=2,
            )


if __name__ == "__main__":
    unittest.main()
