from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lagtcn.models.graph_models import LAGTCNBaseline


class LAGTCNDecoderTest(unittest.TestCase):
    def _model(self, decoder_mode: str, scale_mode: str = "fixed") -> LAGTCNBaseline:
        return LAGTCNBaseline(
            node_num=2,
            input_dim=1,
            hidden_dim=16,
            output_dim=24,
            num_layers=1,
            global_min=0.0,
            global_max=1.0,
            num_timesteps_in=48,
            patch_len=8,
            patch_stride=4,
            dropout=0.0,
            decoder_mode=decoder_mode,
            residual_scale_mode=scale_mode,
            residual_scale_init=0.1,
            seasonal_lag=24,
        )

    def test_persistence_reference_repeats_last_observation(self) -> None:
        model = self._model("persistence_residual")
        history = torch.arange(96, dtype=torch.float32).view(1, 2, 48)
        reference = model._decoder_reference(history)
        self.assertEqual(tuple(reference.shape), (1, 2, 24))
        self.assertTrue(torch.equal(reference, history[:, :, -1:].expand(-1, -1, 24)))

    def test_seasonal_reference_aligns_each_forecast_hour(self) -> None:
        model = self._model("seasonal_residual")
        history = torch.arange(96, dtype=torch.float32).view(1, 2, 48)
        reference = model._decoder_reference(history)
        self.assertTrue(torch.equal(reference, history[:, :, -24:]))

    def test_direct_decoder_has_no_reference_or_residual_scale(self) -> None:
        model = self._model("direct", scale_mode="learnable")
        history = torch.zeros(1, 2, 48)
        self.assertIsNone(model._decoder_reference(history))
        metadata = model.get_decoder_metadata()
        self.assertEqual(metadata["residual_scale_mode"], "not_applicable")
        self.assertIsNone(metadata["residual_scale_effective"])
        self.assertIsNone(model.residual_scale_logit)

    def test_learnable_scale_starts_at_requested_value_and_receives_gradient(self) -> None:
        model = self._model("seasonal_residual", scale_mode="learnable")
        scale = model.residual_scale()
        self.assertAlmostEqual(float(scale.detach()), 0.1, places=6)
        (2.0 * scale).backward()
        self.assertIsNotNone(model.residual_scale_logit.grad)
        self.assertNotEqual(float(model.residual_scale_logit.grad), 0.0)

    def test_unit_scale_removes_the_manual_multiplier(self) -> None:
        model = self._model("seasonal_residual", scale_mode="unit")
        self.assertEqual(float(model.residual_scale()), 1.0)
        self.assertIsNone(model.residual_scale_logit)

    def test_seasonal_decoder_rejects_short_history(self) -> None:
        with self.assertRaises(ValueError):
            LAGTCNBaseline(
                node_num=2,
                input_dim=1,
                hidden_dim=16,
                output_dim=3,
                num_layers=1,
                global_min=0.0,
                global_max=1.0,
                num_timesteps_in=12,
                decoder_mode="seasonal_residual",
                seasonal_lag=24,
            )


if __name__ == "__main__":
    unittest.main()
