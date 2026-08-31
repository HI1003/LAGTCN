from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from graph_sparsity import (
    build_threshold_similarity_adj,
    build_threshold_similarity_adj_torch,
    build_topk_similarity_adj_torch,
    STATIC_THRESHOLD_CANDIDATES,
    DYNAMIC_THRESHOLD_CANDIDATES,
)


class GraphSourceProtocolTest(unittest.TestCase):
    def test_threshold_graph_does_not_force_edge_coverage(self) -> None:
        sim = np.array([
            [1.0, 0.9, 0.1, 0.1],
            [0.9, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.2],
            [0.1, 0.1, 0.2, 1.0],
        ], dtype=np.float32)
        adj = build_threshold_similarity_adj(sim, threshold=0.8)
        self.assertEqual(int(np.count_nonzero(np.triu(adj, 1))), 1)
        self.assertTrue(np.array_equal(np.diag(adj), np.ones(4)))
        self.assertEqual(int(np.count_nonzero(adj[2, :]) - 1), 0)

    def test_numpy_and_batched_torch_threshold_builders_match(self) -> None:
        rng = np.random.default_rng(8)
        raw = rng.random((3, 6, 6), dtype=np.float32)
        sim = 0.5 * (raw + raw.transpose(0, 2, 1))
        batched = build_threshold_similarity_adj_torch(
            torch.tensor(sim), threshold=0.7
        ).numpy()
        expected = np.stack([
            build_threshold_similarity_adj(matrix, threshold=0.7)
            for matrix in sim
        ])
        self.assertTrue(np.allclose(batched, expected))

    def test_adaptive_topk_uses_positive_cosine_and_supports_k_zero(self) -> None:
        sim = torch.tensor([
            [1.0, 0.8, -0.9, 0.1],
            [0.8, 1.0, 0.2, -0.7],
            [-0.9, 0.2, 1.0, 0.6],
            [0.1, -0.7, 0.6, 1.0],
        ])
        empty = build_topk_similarity_adj_torch(sim, top_k=0)
        self.assertTrue(torch.equal(empty, torch.eye(4)))
        adj = build_topk_similarity_adj_torch(sim, top_k=1)
        self.assertEqual(float(adj[0, 2]), 0.0)
        self.assertEqual(float(adj[1, 3]), 0.0)
        self.assertTrue(torch.equal(adj, adj.T))
        offdiag_degree = (adj > 0).sum(dim=1) - 1
        self.assertTrue(bool((offdiag_degree >= 1).all()))

    def test_topk_selected_weights_keep_gradients(self) -> None:
        raw = torch.rand(2, 5, 5, requires_grad=True)
        sim = 0.5 * (raw + raw.transpose(-1, -2))
        adj = build_topk_similarity_adj_torch(sim, top_k=2)
        adj.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)

    def test_adaptive_topk_does_not_force_zero_score_edges(self) -> None:
        sim = torch.eye(4)
        adj = build_topk_similarity_adj_torch(sim, top_k=3)
        self.assertTrue(torch.equal(adj, torch.eye(4)))


    def test_threshold_candidate_range_includes_point_99(self) -> None:
        self.assertEqual(STATIC_THRESHOLD_CANDIDATES[-1], 0.99)
        self.assertEqual(DYNAMIC_THRESHOLD_CANDIDATES[-1], 0.99)

if __name__ == "__main__":
    unittest.main()
