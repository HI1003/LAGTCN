from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from train_eval import _capture_rng_state, _restore_rng_state


class CheckpointRngStateTest(unittest.TestCase):
    def test_restores_python_numpy_and_torch_rng_streams(self) -> None:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        state = _capture_rng_state(use_cuda=False)

        expected_python = random.random()
        expected_numpy = np.random.random(3)
        expected_torch = torch.rand(3)

        random.random()
        np.random.random(3)
        torch.rand(3)
        _restore_rng_state(state, use_cuda=False)

        self.assertEqual(random.random(), expected_python)
        np.testing.assert_array_equal(np.random.random(3), expected_numpy)
        torch.testing.assert_close(torch.rand(3), expected_torch)


if __name__ == "__main__":
    unittest.main()
