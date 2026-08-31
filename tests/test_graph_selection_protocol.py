from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT / "code", ROOT / "scripts"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

import select_ae_graph_hparams as selector
from build_ae_final_manifest import (
    DATASETS,
    dataset_manifest_suffix,
    graph_flags,
    load_selected_graphs,
)
from build_ae_graph_tuning_manifest import candidates
from graph_sparsity import GRAPH_DESIGN_PROTOCOL_VERSION
from select_ae_graph_hparams import (
    SELECTION_RULE,
    SELECTION_VERSION,
    source_revision_provenance,
)


class GraphSelectionProtocolTest(unittest.TestCase):
    def test_source_revision_is_provenance_only(self) -> None:
        config_path = Path("/tmp/config.json")
        clean = {
            "source_git_commit": "abc123",
            "source_git_branch": "paper/applied-energy",
            "source_git_tracked_dirty": False,
            "source_git_untracked_code": False,
        }
        self.assertEqual(
            source_revision_provenance(clean, config_path),
            ("abc123", "paper/applied-energy"),
        )
        self.assertEqual(
            source_revision_provenance(
                {**clean, "source_git_tracked_dirty": True}, config_path
            ),
            ("abc123", "paper/applied-energy"),
        )
        with self.assertRaisesRegex(ValueError, "absent"):
            source_revision_provenance({}, config_path)

    def test_candidate_matrix_removes_only_clipped_duplicates(self) -> None:
        counts = {dataset: len(tuple(candidates(dataset))) for dataset in DATASETS}
        self.assertEqual(counts, {
            "GEFCom2012_2level": 21,
            "GEFCom2017QualifyingMatch_3level": 20,
            "GEFCom2017FinalMatch_4level": 25,
        })
        self.assertEqual(sum(counts.values()), 66)
        four_level_a = [
            params["adaptive_top_k"]
            for source, params, _ in candidates("GEFCom2017FinalMatch_4level")
            if source == "A"
        ]
        self.assertEqual(
            four_level_a,
            [0, 1, 2, 4, 8, 16, 32, 64, 96, 128, 157],
        )

    def test_fusion_flags_reuse_frozen_single_source_values(self) -> None:
        frozen = {
            "static_threshold": 0.9,
            "adaptive_top_k": 4,
            "dynamic_threshold": 0.8,
        }
        flags = graph_flags("H+S+A+D", frozen)
        self.assertEqual(flags, [
            "--static-threshold", "0.9",
            "--adaptive-top-k", "4",
            "--dynamic-threshold", "0.8",
        ])
        self.assertEqual(graph_flags("H", frozen), [])

    def test_frozen_graph_selection_requires_complete_audit(self) -> None:
        selected = {
            dataset: {
                "static_threshold": 0.9,
                "adaptive_top_k": 4,
                "dynamic_threshold": 0.8,
            }
            for dataset in DATASETS
        }
        selected_key = {"S": "static_threshold", "A": "adaptive_top_k", "D": "dynamic_threshold"}
        audit = []
        for dataset in DATASETS:
            by_source = {"S": [], "A": [], "D": []}
            for source, graph_hp, _ in candidates(dataset):
                value = next(iter(graph_hp.values()))
                by_source[source].append({
                    "value": value,
                    "status": "finite",
                    "validation_objective": 1.0 if value == selected[dataset][selected_key[source]] else 2.0,
                    "validation_smase": 0.5 if value == selected[dataset][selected_key[source]] else 0.6,
                    "run_dir": f"/{dataset}/{source}/{value}",
                })
            for source, candidate_rows in by_source.items():
                audit.append({
                    "dataset": dataset,
                    "graph_source": source,
                    "selection_metric": "all_level_mean_1_24_validation_WAPE_pct",
                    "secondary_tie_break_metric": "all_level_mean_1_24_validation_smase",
                    "selection_rule": SELECTION_RULE,
                    "selected_value": selected[dataset][selected_key[source]],
                    "candidate_count": len(candidate_rows),
                    "candidates": candidate_rows,
                })
        payload = {
            "selection_protocol_version": SELECTION_VERSION,
            "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
            "source_experiment_id": "graph_tune",
            "source_git_commit": "abc",
            "source_git_branch": "paper/applied-energy",
            "test_results_accessed": False,
            "selected": selected,
            "audit": audit,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected_graph.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            actual, metadata = load_selected_graphs(path)
            self.assertEqual(actual, selected)
            self.assertEqual(metadata["graph_selection_source_experiment_id"], "graph_tune")
            payload["audit"].pop()
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "auditable"):
                load_selected_graphs(path)

    def test_dataset_scoped_graph_selection_is_strict_within_scope(self) -> None:
        scope = DATASETS[:2]
        selected = {
            dataset: {
                "static_threshold": 0.9,
                "adaptive_top_k": 4,
                "dynamic_threshold": 0.8,
            }
            for dataset in scope
        }
        selected_key = {
            "S": "static_threshold",
            "A": "adaptive_top_k",
            "D": "dynamic_threshold",
        }
        audit = []
        for dataset in scope:
            by_source = {"S": [], "A": [], "D": []}
            for source, graph_hp, _ in candidates(dataset):
                value = next(iter(graph_hp.values()))
                chosen = selected[dataset][selected_key[source]]
                by_source[source].append({
                    "value": value,
                    "status": "finite",
                    "validation_objective": 1.0 if value == chosen else 2.0,
                    "validation_smase": 0.5 if value == chosen else 0.6,
                    "run_dir": f"/{dataset}/{source}/{value}",
                })
            for source, candidate_rows in by_source.items():
                audit.append({
                    "dataset": dataset,
                    "graph_source": source,
                    "selection_metric": "all_level_mean_1_24_validation_WAPE_pct",
                    "secondary_tie_break_metric": "all_level_mean_1_24_validation_smase",
                    "selection_rule": SELECTION_RULE,
                    "selected_value": selected[dataset][selected_key[source]],
                    "candidate_count": len(candidate_rows),
                    "candidates": candidate_rows,
                })
        payload = {
            "selection_protocol_version": SELECTION_VERSION,
            "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
            "source_experiment_id": "graph_tune",
            "selection_scope_datasets": list(scope),
            "test_results_accessed": False,
            "selected": selected,
            "audit": audit,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected_graph_2l3l.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            actual, metadata = load_selected_graphs(path, datasets=scope)
            self.assertEqual(actual, selected)
            self.assertEqual(metadata["graph_selection_scope_datasets"], list(scope))
            with self.assertRaisesRegex(ValueError, "does not audit"):
                load_selected_graphs(path)
            with self.assertRaisesRegex(ValueError, "does not audit"):
                load_selected_graphs(path, datasets=(DATASETS[2],))

    def test_partial_manifest_filename_has_dataset_suffix(self) -> None:
        self.assertEqual(dataset_manifest_suffix(DATASETS), "")
        self.assertEqual(dataset_manifest_suffix(DATASETS[:2]), "_2l3l")
        self.assertEqual(dataset_manifest_suffix((DATASETS[2],)), "_4l")

    def test_selector_main_compares_complete_validation_truth(self) -> None:
        candidate_values = {
            "S": (0.5, 0.9),
            "A": (1, 2),
            "D": (0.6, 0.95),
        }
        config_key = {
            "S": "static_threshold",
            "A": "adaptive_top_k",
            "D": "dynamic_threshold",
        }
        true = np.asarray([[[10.0], [20.0]], [[11.0], [21.0]]], dtype=np.float64)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = root / "runs"
            output = root / "selected.json"
            for source, values in candidate_values.items():
                for candidate_index, value in enumerate(values):
                    run_dir = runs_root / source / str(value)
                    run_dir.mkdir(parents=True)
                    config = {
                        "experiment_stage": selector.STAGE,
                        "experiment_id": "graph_tuning_e2e",
                        "validation_only": True,
                        "graph_sparsity_policy": selector.FINAL_GRAPH_SOURCE_POLICY,
                        "graph_design_protocol_version": selector.GRAPH_DESIGN_PROTOCOL_VERSION,
                        "split_protocol_version": selector.TARGET_TIMESTAMP_SPLIT_VERSION,
                        "dataset": "toy",
                        "graph_mode": source,
                        "model_name": "LAGTCN",
                        "num_timesteps_in": 168,
                        "num_timesteps_out": 24,
                        "feature_set": "target",
                        "training_loss_space": "original",
                        "lagtcn_decoder_mode": "persistence_residual",
                        "lagtcn_residual_scale_mode": "unit",
                        "lagtcn_residual_scale_init": 1.0,
                        "sim_type": "cosine",
                        "seed": 42,
                        "lr": 1e-3,
                        "hidden_dim": 16,
                        "num_layers": 2,
                        "batch_size": 8,
                        "epochs": 2,
                        "patience": 1,
                        "lagtcn_graph_source_version": "test_current",
                        "selection_source_experiment_id": "model_tuning",
                        "selection_protocol_version": "model_selection_v1",
                        "source_git_commit": "abc123",
                        "source_git_branch": "paper/applied-energy",
                        "source_git_tracked_dirty": False,
                        "source_git_untracked_code": False,
                        "smase_scale_metadata": {"scale_per_node": [1.0, 1.0]},
                        config_key[source]: value,
                    }
                    (run_dir / "config.json").write_text(
                        json.dumps(config), encoding="utf-8"
                    )
                    prediction = true + float(2 - candidate_index)
                    columns = ["n1", "n2"]
                    index = ["2020-01-01 00:00:00", "2020-01-01 01:00:00"]
                    pd.DataFrame(
                        prediction[:, :, 0], index=index, columns=columns
                    ).to_csv(run_dir / "validation_pred.csv")
                    pd.DataFrame(
                        true[:, :, 0], index=index, columns=columns
                    ).to_csv(run_dir / "validation_true.csv")

            def expected_values(_dataset: str, source: str):
                return candidate_values[source]

            argv = [
                "select_ae_graph_hparams.py",
                "--runs-root", str(runs_root),
                "--experiment-id", "graph_tuning_e2e",
                "--datasets", "toy",
                "--output", str(output),
            ]
            with (
                mock.patch.object(selector, "DATASETS", ("toy",)),
                mock.patch.object(selector, "expected_values", side_effect=expected_values),
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                selector.main()

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["selected"]["toy"], {
                "static_threshold": 0.9,
                "adaptive_top_k": 2,
                "dynamic_threshold": 0.95,
            })
            self.assertEqual(payload["selection_scope_datasets"], ["toy"])
            serialized = json.dumps(payload)
            self.assertNotIn("true_values", serialized)
            self.assertNotIn("denominator_by_origin", serialized)

            mismatched = runs_root / "S" / "0.9" / "validation_true.csv"
            frame = pd.read_csv(mismatched, index_col=0)
            frame.iloc[0, 0] += 1.0
            frame.to_csv(mismatched)
            with (
                mock.patch.object(selector, "DATASETS", ("toy",)),
                mock.patch.object(selector, "expected_values", side_effect=expected_values),
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaisesRegex(RuntimeError, "validation truth differs"),
            ):
                selector.main()


if __name__ == "__main__":
    unittest.main()
