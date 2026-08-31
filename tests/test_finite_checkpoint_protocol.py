from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from lagtcn.core.training import (
    CONFIG_FINGERPRINT_PROTOCOL_VERSION,
    _atomic_json_dump,
    _config_fingerprint,
    _legacy_config_fingerprint,
    _match_config_fingerprint,
    _sha256_file,
    assert_finite,
    load_best_model_strict,
)


class FiniteCheckpointProtocolTest(unittest.TestCase):
    def test_git_revision_is_provenance_only_in_current_fingerprint(self):
        first = {"model_name": "DCRNN", "dataset": "D", "seed": 42, "lr": 1e-3,
                 "source_git_commit": "a" * 40, "source_git_branch": "paper/applied-energy"}
        second = dict(first, source_git_commit="b" * 40, source_git_branch="DETACHED")
        self.assertEqual(_config_fingerprint(first), _config_fingerprint(second))
        self.assertNotEqual(_legacy_config_fingerprint(first), _legacy_config_fingerprint(second))
        self.assertEqual(_match_config_fingerprint(second, _config_fingerprint(first)),
                         CONFIG_FINGERPRINT_PROTOCOL_VERSION)
        self.assertNotEqual(_config_fingerprint(first), _config_fingerprint(dict(first, lr=5e-4)))

    def test_nonfinite_writes_failure_and_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = {"output_dir": tmp, "model_name": "X", "dataset": "D", "seed": 1}
            with self.assertRaises(FloatingPointError):
                assert_finite(torch.tensor([1.0, float("nan")]), "prediction", "test", config)
            payload = json.loads((Path(tmp) / "failure.json").read_text())
            self.assertEqual(payload["failure_type"], "nonfinite")
            self.assertEqual(payload["summary"]["nonfinite_count"], 1)

    def test_strict_loader_checks_hash_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best_model.pth"
            model = nn.Linear(2, 1)
            torch.save(model.state_dict(), path)
            config = {"model_name": "DLINEAR", "dataset": "D", "seed": 42}
            metadata = {
                "config_fingerprint": _config_fingerprint(config),
                "checkpoint_sha256": _sha256_file(path),
            }
            _atomic_json_dump(metadata, f"{path}.metadata.json")
            clone = nn.Linear(2, 1)
            load_best_model_strict(clone, str(path), config, torch.device("cpu"))
            metadata["config_fingerprint"] = _legacy_config_fingerprint(config)
            _atomic_json_dump(metadata, f"{path}.metadata.json")
            load_best_model_strict(clone, str(path), config, torch.device("cpu"))
            for p1, p2 in zip(model.parameters(), clone.parameters()):
                self.assertTrue(torch.equal(p1, p2))
            with path.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_best_model_strict(clone, str(path), config, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
