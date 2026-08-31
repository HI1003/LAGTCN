from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lagtcn.models.graph_models import LAGTCNBaseline, _LAGTCNBlock
from lagtcn.core.graphs import FINAL_GRAPH_SOURCE_POLICY


class LAGTCNAblationTest(unittest.TestCase):
    def test_uniform_fusion_ignores_learned_logits(self):
        block = _LAGTCNBlock(
            model_dim=8, nhead=2, hop_order=1, dropout=0.0,
            uniform_source_fusion=True,
        )
        with torch.no_grad():
            block.source_logits[:] = torch.tensor([10.0, -10.0, 3.0, -3.0])
        patch = torch.randn(2, 3, 2, 8)
        state = torch.randn(2, 3, 8)
        emb = torch.randn(2, 3, 8)
        adj = torch.eye(3)
        _, weights = block(
            patch, state, [("hierarchy", adj), ("similarity", adj)], emb, emb
        )
        self.assertTrue(torch.allclose(weights, torch.tensor([1.0, 1.0])))

    def test_no_level_and_no_coevolution_forward(self):
        model = LAGTCNBaseline(
            node_num=4,
            input_dim=1,
            hidden_dim=16,
            output_dim=3,
            num_layers=1,
            global_min=0.0,
            global_max=1.0,
            num_timesteps_in=12,
            patch_len=4,
            patch_stride=2,
            dropout=0.0,
            use_level_awareness=False,
            use_coevolution=False,
        )
        model.set_graph_config(
            {
                "graph_mode": "H",
                "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
                "include_self_loops": True,
            },
            base_adj=torch.eye(4),
        )
        model.set_static_graph_sources(hierarchy_adj=torch.eye(4))
        model._level_embedding = lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("level embedding should be disabled")
        )
        x = torch.rand(2, 4, 1, 12)
        edge_index = torch.arange(4).repeat(2, 1)
        pred = model(x, edge_index)
        self.assertEqual(tuple(pred.shape), (2, 4, 3))
        self.assertTrue(torch.isfinite(pred).all())


if __name__ == "__main__":
    unittest.main()
