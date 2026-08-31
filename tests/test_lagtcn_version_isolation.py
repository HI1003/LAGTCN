from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import summarize_experiments
import run_reconcile_matrix
from output_naming import LAGTCN_GRAPH_SOURCE_VERSION_CURRENT


class LAGTCNVersionIsolationTest(unittest.TestCase):
    def _summary_record(self, version: str = LAGTCN_GRAPH_SOURCE_VERSION_CURRENT) -> dict:
        record = {field: None for field in summarize_experiments.GROUP_FIELDS}
        record.update({
            "paper_scope": "journal_applied_energy",
            "experiment_stage": "stage_c_graph_temporal",
            "model_family": "graph-enhanced-temporal",
            "dataset": "GEFCom2012_2level",
            "num_timesteps_out": 24,
            "graph_sparsity_policy": "source_specific_threshold_topk_v2",
            "graph_mode": "H+S+A+D",
            "lagtcn_graph_source_version": version,
            "model_name": "LAGTCN",
            "gnn_type": "gcn",
            "temporal_type": "patchtst",
            "st_mode": "sequential",
        })
        for metric in summarize_experiments.METRIC_COLUMNS:
            record[metric] = 1.0
        return record

    def test_summary_accepts_only_current_layout(self) -> None:
        rows = summarize_experiments._aggregate([self._summary_record()])
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["lagtcn_graph_source_version"],
            LAGTCN_GRAPH_SOURCE_VERSION_CURRENT,
        )

    def test_current_lagtcn_pairs_with_unversioned_baseline(self) -> None:
        baseline = self._summary_record(None)
        baseline.update({
            "model_name": "DLINEAR",
            "model_family": "temporal-only",
            "seed": 42,
            "WAPE": 10.0,
        })
        candidate = self._summary_record()
        candidate.update({"seed": 42, "WAPE": 8.0})

        delta_rows = summarize_experiments._paired_delta(
            [baseline, candidate], baseline_model="DLINEAR"
        )
        self.assertEqual(len(delta_rows), 1)
        self.assertEqual(
            delta_rows[0]["lagtcn_graph_source_version"],
            LAGTCN_GRAPH_SOURCE_VERSION_CURRENT,
        )
        self.assertEqual(delta_rows[0]["delta_WAPE_count"], 1)
        self.assertAlmostEqual(delta_rows[0]["delta_WAPE_mean"], -2.0)

        significance_rows = summarize_experiments._paired_significance(
            [baseline, candidate], baseline_model="DLINEAR"
        )
        wape_rows = [row for row in significance_rows if row["metric"] == "WAPE"]
        self.assertEqual(len(wape_rows), 1)
        self.assertEqual(wape_rows[0]["n_pairs"], 1)

    def test_summary_rejects_missing_lagtcn_version(self) -> None:
        with self.assertRaises(ValueError):
            summarize_experiments._aggregate([self._summary_record(None)])

    def test_reconcile_summary_rejects_wrong_lagtcn_version(self) -> None:
        record = {
            "base_model": "LAGTCN",
            "lagtcn_graph_source_version": "unsupported",
            "WAPE": 9.0,
        }
        with self.assertRaises(ValueError):
            run_reconcile_matrix._summarize(
                [record],
                ["base_model", "lagtcn_graph_source_version"],
                ["WAPE"],
            )

if __name__ == "__main__":
    unittest.main()
