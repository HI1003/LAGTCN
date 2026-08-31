from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT,):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from reproduction.manifests import build_model_matrix as build_ae_final_manifest
from lagtcn.core.naming import RUN_LABEL_MAX_LEN, shorten_existing_run_label


class OutputNamingTest(unittest.TestCase):
    def test_short_label_is_unchanged(self):
        label = "baseline_stgnn_dcrnn_H_h24_s42_proj"
        self.assertEqual(shorten_existing_run_label(label), label)

    def test_long_labels_have_stable_distinct_hash_suffixes(self):
        common = (
            "ae_final_tuning_v1_lagtcn_gcn-patch_transformer_H_h24_"
            "s42_proj_lr0.0005_hidden"
        )
        first = shorten_existing_run_label(f"{common}64")
        second = shorten_existing_run_label(f"{common}128")
        self.assertLessEqual(len(first), RUN_LABEL_MAX_LEN)
        self.assertLessEqual(len(second), RUN_LABEL_MAX_LEN)
        self.assertNotEqual(first, second)
        self.assertEqual(first, shorten_existing_run_label(f"{common}64"))

    def test_formal_manifest_rejects_same_effective_label_in_one_output(self):
        run = {
            "dataset": "GEFCom2012_2level",
            "graph": "H",
            "run_label": "duplicate",
        }
        with self.assertRaisesRegex(ValueError, "duplicate effective run label"):
            build_ae_final_manifest.validate_unique_run_labels([run, dict(run)])

    def test_formal_tuning_labels_keep_only_operational_identifiers(self):
        lagtcn = {
            "model": "LAGTCN",
            "graph": "H",
            "gnn": "gcn",
            "temporal": "patch_transformer",
            "source": "project",
        }
        temporal = {
            "model": "DLINEAR",
            "graph": "H",
            "gnn": "none",
            "temporal": "gru",
            "source": "project",
        }
        self.assertEqual(
            build_ae_final_manifest.formal_run_label(
                "ae_final_tuning_v1",
                lagtcn,
                {"lr": 5e-4, "hidden_dim": 64},
                42,
            ),
            "lagtcn_H_h24_s42_lr5e-4_d64",
        )
        self.assertEqual(
            build_ae_final_manifest.formal_run_label(
                "ae_final_tuning_v1",
                temporal,
                {"lr": 2.5e-4, "hidden_dim": 64},
                42,
            ),
            "dlin_h24_s42_lr2.5e-4",
        )


if __name__ == "__main__":
    unittest.main()
