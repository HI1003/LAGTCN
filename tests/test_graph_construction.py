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
    graph_edge_diagnostics,
)


class CurrentGraphConstructionTest(unittest.TestCase):
    def test_threshold_numpy_and_torch_match(self):
        n = 21
        rng = np.random.default_rng(7)
        sim = rng.uniform(0.0, 1.0, size=(n, n)).astype(np.float32)
        sim = 0.5 * (sim + sim.T)
        a_np = build_threshold_similarity_adj(sim, threshold=0.7)
        a_t = build_threshold_similarity_adj_torch(torch.tensor(sim), threshold=0.7).numpy()
        self.assertTrue(np.array_equal(a_np > 0, a_t > 0))
        self.assertTrue(np.allclose(a_np, a_t, atol=1e-7))
        self.assertTrue(np.allclose(a_np, a_np.T))
        self.assertTrue(np.allclose(np.diag(a_np), 1.0))

    def test_batched_threshold_matches_individual_construction(self):
        n = 15
        rng = np.random.default_rng(17)
        raw = rng.uniform(0.0, 1.0, size=(3, n, n)).astype(np.float32)
        sim = 0.5 * (raw + raw.transpose(0, 2, 1))
        batched = build_threshold_similarity_adj_torch(torch.tensor(sim), 0.8).numpy()
        individual = np.stack([
            build_threshold_similarity_adj_torch(torch.tensor(matrix), 0.8).numpy()
            for matrix in sim
        ])
        self.assertTrue(np.array_equal(batched, individual))

    def test_topk_is_symmetric_and_has_finite_gradients(self):
        raw = torch.rand(3, 21, 21, requires_grad=True)
        sim = 0.5 * (raw + raw.transpose(-1, -2))
        adj = build_topk_similarity_adj_torch(sim, top_k=4)
        self.assertTrue(torch.allclose(adj, adj.transpose(-1, -2)))
        for sample in adj.detach().numpy():
            self.assertGreater(graph_edge_diagnostics(sample)["undirected_edge_count"], 0)
        adj.sum().backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(bool(torch.isfinite(raw.grad).all()))
        self.assertGreater(float(raw.grad.abs().sum()), 0.0)

    def test_topk_zero_has_only_self_loops(self):
        sim = torch.ones(5, 5)
        adj = build_topk_similarity_adj_torch(sim, top_k=0)
        self.assertTrue(torch.equal(adj, torch.eye(5)))


if __name__ == "__main__":
    unittest.main()
