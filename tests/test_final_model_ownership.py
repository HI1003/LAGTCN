from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lagtcn.core.graphs import FINAL_GRAPH_SOURCE_POLICY
from lagtcn.train import _resolve_graph_adjacency_file
from lagtcn.models.graph_models import (
    DCRNNBaseline,
    LAGTCNBaseline,
    MTGNNBaseline,
)
from lagtcn.models.temporal_baselines import (
    DLinearBaseline,
    ITransformerBaseline,
    NHiTSBaseline,
    PatchTSTBaseline,
)
from lagtcn.core.training import _compute_model_loss, _config_fingerprint, _forward_training_model


class FinalModelOwnershipTest(unittest.TestCase):
    def test_final_static_only_graph_modes_use_runtime_builder_seed(self):
        self.assertEqual(
            _resolve_graph_adjacency_file("S", "cosine", FINAL_GRAPH_SOURCE_POLICY),
            "adj_hierarchy.npy",
        )
        self.assertEqual(
            _resolve_graph_adjacency_file("S+A+D", "cosine", FINAL_GRAPH_SOURCE_POLICY),
            "adj_hierarchy.npy",
        )
        with self.assertRaises(ValueError):
            _resolve_graph_adjacency_file("S", "cosine", "unsupported")

    def test_non_lagtcn_baseline_rejects_data_driven_graph_sources(self):
        model = DCRNNBaseline(
            node_num=4, input_dim=1, hidden_dim=8, output_dim=3,
            num_layers=1, global_min=0.0, global_max=1.0,
        )
        with self.assertRaisesRegex(ValueError, "owned by LAGTCN"):
            model.set_graph_config({
                "graph_mode": "H+S",
                "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
                "static_threshold": 0.9,
            }, base_adj=np.eye(4, dtype=np.float32))

    def test_dedicated_graph_baseline_has_no_parent_placeholder_parameters(self):
        model = DCRNNBaseline(
            node_num=4, input_dim=1, hidden_dim=8, output_dim=3,
            num_layers=1, global_min=0.0, global_max=1.0,
        )
        names = {name for name, _ in model.named_parameters()}
        forbidden = {
            "layer_norm.weight", "layer_norm.bias", "projection.weight",
            "projection.bias", "node_encoder.weight", "node_encoder.bias",
            "temporal_encoder.weight_ih_l0", "temporal_encoder.weight_hh_l0",
        }
        self.assertTrue(names.isdisjoint(forbidden), sorted(names & forbidden))

    def test_nhits_keeps_official_multirate_identity_stacks(self):
        model = NHiTSBaseline(
            node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
            num_layers=2, global_min=0.0, global_max=1.0,
            num_timesteps_in=12,
        )
        self.assertEqual(model.stack_n_blocks, (1, 1, 1))
        self.assertEqual(model.stack_pooling_sizes, (2, 2, 1))
        self.assertEqual(model.stack_frequency_downsamples, (4, 2, 1))
        self.assertEqual([block.pooling_size for block in model.blocks], [2, 2, 1])
        self.assertTrue(all(block.basis.backcast_size == 12 for block in model.blocks))
        edge_index = torch.arange(4, dtype=torch.long).repeat(2, 1)
        output = model(torch.randn(2, 4, 1, 12), edge_index)
        output.sum().backward()
        missing_grad = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing_grad, [])

    def test_formal_lagtcn_uses_one_shared_relation_operator(self):
        torch.manual_seed(3)
        model = LAGTCNBaseline(
            node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
            num_layers=1, global_min=0.0, global_max=1.0,
            num_timesteps_in=12, patch_len=4, patch_stride=2,
        )
        adjacency = np.eye(4, dtype=np.float32)
        sum_matrix = np.array(
            [
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        )
        model.set_hierarchy_metadata(sum_matrix, middle_levels=[[1]], bottom_start_idx=2)
        self.assertEqual(model.hier_level_ids.tolist(), [0, 1, 2, 2])
        self.assertEqual(model.hierarchy_level_encoding_version, "explicit_middle_levels_v1")
        model.set_graph_config(
            {
                "graph_mode": "H",
                "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
                "include_self_loops": True,
            },
            base_adj=adjacency,
        )
        model.set_static_graph_sources(hierarchy_adj=adjacency)
        self.assertIsInstance(model.blocks[0].graph_mix, torch.nn.Linear)
        self.assertFalse(model.blocks[0].graph_mix.bias is not None)
        self.assertFalse(model.blocks[0].source_logits.requires_grad)
        state_keys = tuple(model.state_dict())
        self.assertEqual(sum(key.endswith("graph_mix.weight") for key in state_keys), 1)

        edge_index, edge_weight = dense_to_sparse(torch.as_tensor(adjacency))
        output = model(torch.randn(2, 4, 1, 12), edge_index, edge_weight)
        output.sum().backward()
        missing_grad = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing_grad, [])


    def test_temporal_baselines_expose_reference_architecture_components(self):
        dlinear = DLinearBaseline(
            node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
            num_layers=2, global_min=0.0, global_max=1.0,
            num_timesteps_in=12,
        )
        self.assertEqual(dlinear.architecture_version, "dlinear_official_shared_v1")
        self.assertEqual(dlinear.trend_linear.in_features, 12)
        self.assertEqual(dlinear.seasonal_linear.out_features, 3)

        patchtst = PatchTSTBaseline(
            node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
            num_layers=2, global_min=0.0, global_max=1.0,
            num_timesteps_in=12, patch_len=4, patch_stride=2,
        )
        self.assertEqual(
            patchtst.architecture_version, "patchtst_supervised_official_v1"
        )
        self.assertFalse(patchtst.revin.affine)
        self.assertIsInstance(patchtst.end_padding, torch.nn.ReplicationPad1d)
        self.assertEqual(patchtst.head.in_features, patchtst.model_dim * patchtst.patch_num)
        self.assertTrue(all(
            layer.attention.residual_attention for layer in patchtst.encoder.layers
        ))

        itransformer = ITransformerBaseline(
            node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
            num_layers=2, global_min=0.0, global_max=1.0,
            num_timesteps_in=12,
        )
        self.assertEqual(
            itransformer.architecture_version, "itransformer_official_finalnorm_v2"
        )
        self.assertEqual(itransformer.token_projection.in_features, 12)
        self.assertEqual(itransformer.projector.out_features, 3)
        self.assertIsInstance(itransformer.encoder_norm, torch.nn.LayerNorm)

        edge_index = torch.arange(4, dtype=torch.long).repeat(2, 1)
        output = itransformer(torch.randn(2, 4, 1, 12), edge_index)
        output.sum().backward()
        self.assertIsNotNone(itransformer.encoder_norm.weight.grad)
        self.assertIsNotNone(itransformer.encoder_norm.bias.grad)
        self.assertGreater(float(itransformer.encoder_norm.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(itransformer.encoder_norm.bias.grad.abs().sum()), 0.0)

    def test_dcrnn_training_hook_uses_seq2seq_curriculum(self):
        torch.manual_seed(7)
        model = DCRNNBaseline(
            node_num=4, input_dim=1, hidden_dim=8, output_dim=3,
            num_layers=2, global_min=0.0, global_max=1.0,
            cl_decay_steps=1000, use_curriculum_learning=True,
        )
        self.assertEqual(len(model.encoder_cells), 2)
        self.assertEqual(len(model.decoder_cells), 2)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])
        x = torch.randn(2, 4, 1, 12)
        targets = torch.randn(2, 4, 3)
        model.train()
        output = _forward_training_model(model, x, edge_index, targets, 0)
        self.assertEqual(tuple(output.shape), (2, 4, 3))
        self.assertTrue(torch.isfinite(output).all())
        self.assertAlmostEqual(
            model._last_sampling_threshold,
            DCRNNBaseline.sampling_threshold(0, 1000),
        )
        output.sum().backward()

    def test_mtgnn_keeps_native_graph_inception_and_trains_all_horizons(self):
        torch.manual_seed(11)
        model = MTGNNBaseline(
            node_num=4, input_dim=1, hidden_dim=8, output_dim=4,
            num_layers=1, global_min=0.0, global_max=1.0,
            num_timesteps_in=12, top_k=2,
            stgnn_graph_source="native",
        )
        self.assertEqual(model.filter_convs[0].kernel_set, (2, 3, 6, 7))
        self.assertEqual(len(model.mixprop_forward), 1)
        self.assertEqual(len(model.mixprop_backward), 1)
        pred = torch.zeros(2, 4, 4)
        true = torch.zeros_like(pred)
        true[:, :, -1] = 1.0
        self.assertGreater(
            _compute_model_loss(model, pred, true, torch.nn.L1Loss()).item(),
            0.0,
        )
        adjacency = model.get_adaptive_adjacency()
        self.assertLessEqual(int((adjacency > 0).sum(dim=1).max()), 2)
        self.assertFalse(torch.allclose(adjacency, adjacency.transpose(0, 1)))

    def test_mtgnn_native_graph_and_forecast_are_deterministic_in_eval(self):
        torch.manual_seed(17)
        model = MTGNNBaseline(
            node_num=6, input_dim=1, hidden_dim=8, output_dim=4,
            num_layers=1, global_min=0.0, global_max=1.0,
            num_timesteps_in=12, top_k=2,
            stgnn_graph_source="native",
        )
        self.assertEqual(
            model.architecture_version, "mtgnn_official_direct_multihorizon_v3"
        )
        edge_index = torch.arange(6, dtype=torch.long).repeat(2, 1)
        x = torch.randn(2, 6, 1, 12)
        model.eval()
        with torch.no_grad():
            first = model(x, edge_index)
            first_adj = model._last_adaptive_adjacency.clone()
            second = model(x, edge_index)
            second_adj = model._last_adaptive_adjacency.clone()
        torch.testing.assert_close(first_adj, second_adj, rtol=0.0, atol=0.0)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_lagtcn_patch_transformer_is_not_patchtst_alias(self):
        model = LAGTCNBaseline(
            node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
            num_layers=1, global_min=0.0, global_max=1.0,
            num_timesteps_in=12, patch_len=4, patch_stride=2,
        )
        self.assertEqual(model.temporal_backbone, "patch_transformer")
        self.assertEqual(
            model.architecture_version, "lagtcn_decoder_modes_hanchor_v3"
        )
        decoder = model.get_decoder_metadata()
        self.assertEqual(decoder["decoder_mode"], "persistence_residual")
        self.assertEqual(decoder["residual_scale_mode"], "unit")
        self.assertEqual(decoder["residual_scale_init"], 1.0)
        self.assertEqual(decoder["residual_scale_effective"], 1.0)
        self.assertTrue(hasattr(model, "patch_proj"))
        self.assertFalse(hasattr(model, "revin"))
        with self.assertRaises(ValueError):
            LAGTCNBaseline(
                node_num=4, input_dim=1, hidden_dim=16, output_dim=3,
                num_layers=1, global_min=0.0, global_max=1.0,
                num_timesteps_in=12, temporal_backbone="patchtst",
            )

    def test_checkpoint_fingerprint_tracks_architecture_version(self):
        config = {"model_name": "PATCHTST", "num_timesteps_in": 168}
        first = dict(config, model_architecture_version="patchtst_v1")
        second = dict(config, model_architecture_version="patchtst_v2")
        self.assertNotEqual(_config_fingerprint(first), _config_fingerprint(second))

        zero_coherence = dict(config, coherency_lambda=0.0)
        nonzero_coherence = dict(config, coherency_lambda=0.1)
        self.assertNotEqual(
            _config_fingerprint(zero_coherence), _config_fingerprint(nonzero_coherence)

        )
if __name__ == "__main__":
    unittest.main()
