from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lagtcn.core.training import train_model


class _ToyForecast(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, x, edge_index):
        del edge_index
        return self.linear(x).unsqueeze(-1)

    @staticmethod
    def transform_target(y):
        return y.unsqueeze(-1) if y.dim() == 2 else y


class GradientAccumulationTest(unittest.TestCase):
    def _train(self, initial_state, signal, batch_size, accumulation_steps):
        model = _ToyForecast()
        model.load_state_dict(initial_state)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as output_dir:
            config = {
                "epochs": 1,
                "patience": 2,
                "batch_size": batch_size,
                "gradient_accumulation_steps": accumulation_steps,
                "output_dir": output_dir,
                "model_name": "toy",
                "timestamp": "test",
                "training_loss_space": "original",
                "checkpoint_every_epochs": 0,
            }
            torch.manual_seed(123)
            *_, efficiency = train_model(
                model,
                signal,
                signal,
                torch.empty((2, 0), dtype=torch.long),
                optimizer,
                torch.nn.SmoothL1Loss(),
                config,
                "cpu",
            )
        return copy.deepcopy(model.state_dict()), efficiency

    def test_micro_batches_match_the_same_effective_batch_update(self):
        rng = np.random.default_rng(7)
        signal = SimpleNamespace(
            features=rng.normal(size=(10, 3)).astype("float32"),
            targets=rng.normal(size=(10, 2)).astype("float32"),
        )
        initial_state = copy.deepcopy(_ToyForecast().state_dict())

        full_state, full_efficiency = self._train(initial_state, signal, 4, 1)
        accumulated_state, accumulated_efficiency = self._train(
            initial_state, signal, 2, 2
        )

        for name in full_state:
            torch.testing.assert_close(full_state[name], accumulated_state[name])
        self.assertEqual(full_efficiency["nominal_effective_batch_size"], 4)
        self.assertEqual(accumulated_efficiency["nominal_effective_batch_size"], 4)
        self.assertEqual(accumulated_efficiency["optimizer_steps_per_epoch"], 3)

    def test_invalid_accumulation_steps_are_rejected(self):
        signal = SimpleNamespace(
            features=np.zeros((2, 3), dtype="float32"),
            targets=np.zeros((2, 2), dtype="float32"),
        )
        model = _ToyForecast()
        with tempfile.TemporaryDirectory() as output_dir:
            config = {
                "epochs": 1,
                "batch_size": 1,
                "gradient_accumulation_steps": 0,
                "output_dir": output_dir,
            }
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                train_model(
                    model, signal, signal, torch.empty((2, 0), dtype=torch.long),
                    torch.optim.Adam(model.parameters()), torch.nn.SmoothL1Loss(),
                    config, "cpu",
                )


if __name__ == "__main__":
    unittest.main()
