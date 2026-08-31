from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "code", ROOT / "scripts"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import benchmark_ae_final
from ae_protocol import is_formal_ae_stage
import build_ae_final_manifest
import postprocess_ae_final
import select_ae_final_hparams
from graph_sparsity import (
    FINAL_GRAPH_SOURCE_POLICY,
    GRAPH_DESIGN_PROTOCOL_VERSION,
)
from output_naming import LAGTCN_GRAPH_SOURCE_VERSION_CURRENT
from data_loader import TARGET_TIMESTAMP_SPLIT_VERSION
from mase import MASE_VERSION
from metrics import calculate_level_metrics
from train_eval import _compute_metrics


class FinalBatchProtocolTest(unittest.TestCase):
    def test_phase1_counts_exclude_optional_phase2_extensions(self):
        current = build_ae_final_manifest.CURRENT_MODEL_CONFIGS
        phase2 = build_ae_final_manifest.PHASE2_MODEL_CONFIGS
        self.assertEqual(len(current), 6)
        self.assertNotIn("DEEPHGNN_SPECTGNN", {item["model"] for item in current})
        self.assertEqual({item["model"] for item in phase2}, {"DEEPHGNN_SPECTGNN"})
        self.assertEqual(len(benchmark_ae_final.MODELS), 7)
        datasets = len(build_ae_final_manifest.DATASETS)
        seeds = len(build_ae_final_manifest.SEEDS)
        tuning = datasets * 7 * 4
        graph_tuning = 66
        main = datasets * seeds * (
            len(build_ae_final_manifest.LAGTCN_GRAPHS) + len(current)
        )
        reconciliation_trajectories = datasets * seeds * (1 + len(current))
        benchmark = datasets * seeds * len(benchmark_ae_final.MODELS)
        self.assertEqual((tuning, graph_tuning, main), (84, 66, 144))
        self.assertEqual(tuning + graph_tuning + main, 294)
        self.assertEqual(reconciliation_trajectories * 3, 189)
        self.assertEqual(benchmark, 63)

    def test_four_level_stgnn_batch_protocol(self):
        expected = {
            ("DCRNN", 128): (64, 2),
            ("DCRNN", 256): (32, 4),
            ("MTGNN", 128): (32, 4),
            ("MTGNN", 256): (16, 8),
        }
        for (model, hidden), (physical, accumulation) in expected.items():
            actual = build_ae_final_manifest.formal_batch_protocol(
                "GEFCom2017FinalMatch_4level", model, hidden, 128
            )
            self.assertEqual(actual["physical_batch_size"], physical)
            self.assertEqual(actual["gradient_accumulation_steps"], accumulation)
            self.assertEqual(actual["effective_batch_size"], 128)
    def test_metric_pipeline_uses_frozen_nodewise_smase_scale(self):
        truth = np.array([
            [[2.0, 4.0], [10.0, 14.0]],
            [[4.0, 6.0], [12.0, 16.0]],
        ])
        pred = np.array([
            [[1.0, 2.0], [8.0, 10.0]],
            [[3.0, 4.0], [10.0, 12.0]],
        ])
        scale = np.array([2.0, 4.0])
        result = _compute_metrics(truth, pred, mase_scale=scale)
        per_node = np.abs(truth - pred).mean(axis=(0, 2)) / scale
        self.assertAlmostEqual(result["MASE"], float(per_node.mean()))
        self.assertEqual(result["MASE_version"], MASE_VERSION)
        self.assertEqual(result["MASE_n_excluded"], 0)

    def test_phase2_is_formal_and_cannot_fall_back_to_legacy_mase(self):
        self.assertTrue(is_formal_ae_stage("ae_final_main_v1"))
        self.assertTrue(is_formal_ae_stage("ae_phase2_ablation_v1"))
        self.assertTrue(is_formal_ae_stage("ae_phase2_deephgnn_v1"))
        self.assertFalse(is_formal_ae_stage("pilot"))
        truth = np.ones((3, 2, 24), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "refusing legacy MASE fallback"):
            _compute_metrics(
                truth,
                truth,
                mase_scale=None,
                require_mase_scale=True,
            )

        with self.assertRaisesRegex(ValueError, "refusing legacy MASE fallback"):
            calculate_level_metrics(
                truth,
                truth,
                {
                    "experiment_stage": "ae_phase2_ablation_v1",
                    "bottom_start_idx": 1,
                    "num_bottom_nodes": 1,
                },
            )


    def test_postprocess_recomputes_old_versions_and_rejects_nnls_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reconciliation_diagnostics.json"
            path.write_text(json.dumps({"postprocess_version": "old"}))
            self.assertFalse(postprocess_ae_final.has_current_diagnostics(path))
            path.write_text(json.dumps({"postprocess_version": postprocess_ae_final.VERSION}))
            self.assertTrue(postprocess_ae_final.has_current_diagnostics(path))
        good = {"n_failures": 0, "horizon_diagnostics": []}
        postprocess_ae_final.assert_mint_nnls_success(Path("run"), good)
        bad = {
            "n_failures": 1,
            "horizon_diagnostics": [{"horizon": 24, "n_failures": 1}],
        }
        with self.assertRaisesRegex(RuntimeError, "refusing fallback output"):
            postprocess_ae_final.assert_mint_nnls_success(Path("run"), bad)

    def test_postprocess_validates_frozen_prediction_index_and_shape(self):
        index = pd.date_range("2026-01-01", periods=3, freq="h")
        config = {
            "node_names": ["Total", "Bottom"],
            "time_index": [str(value) for value in index],
            "split_provenance": {"segments": {"test": {"origin_count": 3}}},
        }
        values = np.zeros((3, 2, 24), dtype=np.float64)
        postprocess_ae_final.validate_prediction_contract(
            Path("run"), config, values, config["node_names"], index, "test"
        )
        bad_index = index.copy()
        bad_index = bad_index.delete(1).append(pd.DatetimeIndex([index[-1] + pd.Timedelta(hours=2)]))
        with self.assertRaisesRegex(ValueError, "CSV index differs"):
            postprocess_ae_final.validate_prediction_contract(
                Path("run"), config, values, config["node_names"], bad_index, "test"
            )
        with self.assertRaisesRegex(ValueError, r"\[origins,nodes,24\]"):
            postprocess_ae_final.validate_prediction_contract(
                Path("run"), config, values[:, :, :6], config["node_names"], index, "test"
            )

    def test_complete_formal_batch_contract_accepts_only_frozen_model_specs(self):
        common = {
            "paper_scope": "journal_applied_energy",
            "experiment_stage": postprocess_ae_final.FORMAL_STAGE,
            "feature_set": "target",
            "num_timesteps_in": 168,
            "num_timesteps_out": 24,
            "training_loss_space": "original",
            "validation_only": False,
            "checkpoint_every_epochs": 1,
            "resume": "auto",
            "coherency_lambda": 0.0,
            "source_git_branch": "paper/applied-energy",
            "batch_size": 128,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 128,
            "epochs": 150,
            "patience": 20,
            "num_layers": 2,
        }
        controls = {
            "static_threshold": 0.9,
            "adaptive_top_k": 8,
            "dynamic_threshold": 0.95,
        }
        selected = []
        for dataset in build_ae_final_manifest.DATASETS:
            for seed in build_ae_final_manifest.SEEDS:
                for graph in build_ae_final_manifest.LAGTCN_GRAPHS:
                    config = {
                        **common,
                        **controls,
                        "dataset": dataset,
                        "seed": seed,
                        "model_name": "LAGTCN",
                        "graph_mode": graph,
                        "gnn_type": "gcn",
                        "temporal_type": "patch_transformer",
                        "stgnn_graph_source": "project",
                        "lagtcn_ablation": "none",
                        "lagtcn_decoder_mode": "persistence_residual",
                        "lagtcn_residual_scale_mode": "unit",
                        "lagtcn_residual_scale_init": 1.0,
                        "lr": 0.001,
                        "hidden_dim": 64,
                    }
                    selected.append((Path(f"{dataset}/{seed}/{graph}"), config))
                for item in build_ae_final_manifest.CURRENT_MODEL_CONFIGS:
                    config = {
                        **common,
                        "dataset": dataset,
                        "seed": seed,
                        "model_name": item["model"],
                        "graph_mode": item["graph"],
                        "gnn_type": item["gnn"],
                        "temporal_type": item["temporal"],
                        "stgnn_graph_source": item["source"],
                        "lr": 0.001,
                        "hidden_dim": 128 if dataset == "GEFCom2017FinalMatch_4level" else 64,
                    }
                    resource = build_ae_final_manifest.formal_batch_protocol(
                        dataset, item["model"], config["hidden_dim"], 128
                    )
                    config["batch_size"] = resource["physical_batch_size"]
                    config["gradient_accumulation_steps"] = resource[
                        "gradient_accumulation_steps"
                    ]
                    config["effective_batch_size"] = resource["effective_batch_size"]
                    if item["role"] == "end_to_end_coherent":
                        config["prediction_role"] = item["role"]
                    selected.append((Path(f"{dataset}/{seed}/{item['model']}"), config))
        postprocess_ae_final.validate_complete_batch_contract(selected)
        mtgnn_config = next(
            config for _, config in selected
            if config["model_name"] == "MTGNN"
        )
        mtgnn_config["stgnn_graph_source"] = "project"
        with self.assertRaisesRegex(RuntimeError, "formal model contract mismatch"):
            postprocess_ae_final.validate_complete_batch_contract(selected)

    def test_formal_manifest_explicitly_freezes_loss_and_lagtcn_decoder(self):
        args = mock.Mock(
            num_layers=2,
            epochs=150,
            patience=20,
            batch_size=128,
            device="cuda:0",
            selection_source_experiment_id=None,
            selection_protocol_version=None,
            graph_selection_source_experiment_id=None,
            graph_selection_protocol_version=None,
        )
        item = {
            "model": "LAGTCN",
            "graph": "H",
            "gnn": "gcn",
            "temporal": "patch_transformer",
            "source": "project",
        }
        _, command = build_ae_final_manifest.common_command(
            args,
            "GEFCom2012_2level",
            42,
            item,
            {"hidden_dim": 64, "lr": 0.001},
            "formal_batch",
        )
        rendered = " ".join(command)
        self.assertIn("--training-loss-space original", rendered)
        self.assertIn("--lagtcn-decoder-mode persistence_residual", rendered)
        self.assertIn("--resume auto", rendered)
        self.assertIn("--batch-size 128", rendered)
        self.assertIn("--gradient-accumulation-steps 1", rendered)
        self.assertIn("--lagtcn-residual-scale-mode unit", rendered)
        self.assertIn("--lagtcn-residual-scale-init 1", rendered)
        self.assertNotIn("--expected-git-commit", rendered)
        self.assertNotIn("--require-clean-worktree", rendered)

    def test_postprocess_rejects_ambiguous_formal_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, experiment_id in enumerate(("batch_a", "batch_b")):
                run = root / f"run_{index}"
                run.mkdir()
                (run / "config.json").write_text(json.dumps({
                    "experiment_stage": postprocess_ae_final.FORMAL_STAGE,
                    "experiment_id": experiment_id,
                    "num_timesteps_out": 24,
                    "dataset": "GEFCom2012_2level",
                    "model_name": "LAGTCN",
                    "graph_mode": "H",
                    "seed": 42 + index,
                    "source_git_commit": "abc123",
                    "source_git_branch": "paper/applied-energy",
                    "source_git_tracked_dirty": False,
                    "split_protocol_version": TARGET_TIMESTAMP_SPLIT_VERSION,
                    "selection_source_experiment_id": "tuning_a",
                    "selection_protocol_version": build_ae_final_manifest.MODEL_SELECTION_PROTOCOL_VERSION,
                    "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
                    "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
                    "lagtcn_graph_source_version": LAGTCN_GRAPH_SOURCE_VERSION_CURRENT,
                }))
            with self.assertRaises(RuntimeError):
                postprocess_ae_final.matching_runs(root)
            experiment_id, rows = postprocess_ae_final.matching_runs(root, "batch_a")
            self.assertEqual(experiment_id, "batch_a")
            self.assertEqual(len(rows), 1)
            with self.assertRaisesRegex(RuntimeError, "not the complete 144-run"):
                postprocess_ae_final.matching_runs(
                    root, "batch_a", require_complete=True
                )
            legacy_path = root / "run_0" / "config.json"
            legacy = json.loads(legacy_path.read_text())
            legacy.pop("split_protocol_version")
            legacy_path.write_text(json.dumps(legacy))
            with self.assertRaisesRegex(RuntimeError, "legacy or missing split"):
                postprocess_ae_final.matching_runs(root, "batch_a")

    def test_postprocess_dataset_scope_parser(self):
        two_level = "GEFCom2012_2level"
        three_level = "GEFCom2017QualifyingMatch_3level"
        self.assertEqual(
            postprocess_ae_final.parse_dataset_scope(f"{three_level},{two_level}"),
            (two_level, three_level),
        )
        with self.assertRaisesRegex(ValueError, "Unknown formal datasets"):
            postprocess_ae_final.parse_dataset_scope("unknown")

    def test_postprocess_uses_run_frozen_smase_scale_after_precision_check(self):
        meta = mock.Mock()
        meta.scale_metadata = {"mase_version": MASE_VERSION}
        meta.train_length = 10
        meta.naive_scale = np.array([1.0, 2.0])
        recorded = np.array([1.0000005, 2.0000005])
        config = {
            "smase_scale_metadata": {
                "mase_version": MASE_VERSION,
                "seasonal_period": 24,
                "train_length": 10,
                "scale_per_node": recorded.tolist(),
            },
        }
        actual = postprocess_ae_final.validate_smase_metadata(Path("run"), config, meta)
        np.testing.assert_array_equal(actual, recorded)
        config["smase_scale_metadata"]["scale_per_node"] = [1.01, 2.0]
        with self.assertRaisesRegex(ValueError, "scales differ"):
            postprocess_ae_final.validate_smase_metadata(Path("run"), config, meta)

    def test_manifest_builder_requires_terminal_invalid_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.json"
            finite = [{
                "hyperparameters": {"lr": 0.001, "hidden_dim": 8},
                "run_dir": "/finite/0",
                "status": "finite",
                "validation_objective": 1.0,
                "validation_smase": 0.1,
            }]
            terminal = []
            for index in range(1, 4):
                invalid_hp = {"lr": 0.001 * (index + 1), "hidden_dim": 8}
                terminal.append({
                    "hyperparameters": invalid_hp,
                    "run_dir": f"/invalid/{index}",
                    "status": "terminal_invalid",
                    "validation_objective": None,
                    "validation_smase": None,
                    "terminal_decision": {
                        "status": "terminal_invalid_configuration",
                        "failure_type": "numeric_divergence",
                        "selection_action": "exclude_from_validation_selection_without_retry",
                        "hyperparameters": invalid_hp,
                        "status_file": "/audit/status.json",
                        "source_manifest": "/audit/tuning.jsonl",
                    },
                })
            payload = {
                "selection_protocol_version": build_ae_final_manifest.MODEL_SELECTION_PROTOCOL_VERSION,
                "source_experiment_id": "tuning_a",
                "test_results_accessed": False,
                "selected": {"D": {"LAGTCN": finite[0]["hyperparameters"]}},
                "audit": [{
                    "dataset": "D",
                    "model": "LAGTCN",
                    "selection_metric": "all_level_mean_1_24_validation_WAPE_pct",
                    "secondary_tie_break_metric": "all_level_mean_1_24_validation_smase",
                    "selection_rule": "minimum_WAPE_then_smase_on_exact_tie",
                    "winner": finite[0],
                    "candidate_count": 4,
                    "finite_candidate_count": 1,
                    "ineligible_candidate_count": 3,
                    "all_candidates_terminal": True,
                    "candidates": [*finite, *terminal],
                }],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(build_ae_final_manifest, "DATASETS", ("D",)):
                selected, _ = build_ae_final_manifest.load_selected(path, model_configs=())
                self.assertEqual(selected["D"]["LAGTCN"], finite[0]["hyperparameters"])
                payload["audit"][0]["candidates"][-1].pop("terminal_decision")
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Invalid candidate records"):
                    build_ae_final_manifest.load_selected(path, model_configs=())

    def test_manifest_builder_rejects_selection_without_candidate_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "selected.json"
            path.write_text(json.dumps({
                "selection_protocol_version": build_ae_final_manifest.MODEL_SELECTION_PROTOCOL_VERSION,
                "source_experiment_id": "tuning_a",
                "source_git_commit": "abc123",
                "source_git_branch": "paper/applied-energy",
                "test_results_accessed": False,
                "selected": {},
            }))
            with self.assertRaisesRegex(ValueError, "candidate audit"):
                build_ae_final_manifest.load_selected(path)

    def test_hparam_selection_requires_four_unique_candidates_from_one_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "selected.json"
            for index in range(4):
                run = root / f"candidate_{index}"
                run.mkdir()
                (run / "config.json").write_text(json.dumps({
                    "experiment_stage": select_ae_final_hparams.STAGE,
                    "experiment_id": "tuning_a",
                    "validation_only": True,
                    "num_timesteps_out": 24,
                    "training_loss_space": "original",
                    "dataset": "D",
                    "model_name": "M",
                    "lr": 0.001 * (index + 1),
                    "hidden_dim": 8,
                    "source_git_commit": f"abc{index % 2}",
                    "source_git_branch": "paper/applied-energy",
                    "source_git_tracked_dirty": bool(index % 2),
                    "split_protocol_version": TARGET_TIMESTAMP_SPLIT_VERSION,
                }))
                (run / "validation_metrics.json").write_text(json.dumps({
                    "WAPE": 1.0,
                    "MASE": float(4 - index),
                }))
            argv = [
                "select_ae_final_hparams.py", "--runs-root", str(root),
                "--experiment-id", "tuning_a", "--output", str(output),
            ]
            with mock.patch.object(select_ae_final_hparams, "DATASETS", ("D",)), \
                 mock.patch.object(select_ae_final_hparams, "MODELS", ("M",)), \
                 mock.patch.object(sys, "argv", argv):
                select_ae_final_hparams.main()
            payload = json.loads(output.read_text())
            self.assertEqual(payload["source_experiment_id"], "tuning_a")
            self.assertEqual(payload["runtime_git_binding"], "provenance_only")
            self.assertEqual(len(payload["source_git_revisions"]), 2)
            self.assertEqual(payload["audit"][0]["candidate_count"], 4)
            self.assertEqual(payload["audit"][0]["finite_candidate_count"], 4)
            self.assertEqual(
                payload["audit"][0]["selection_rule"],
                "minimum_WAPE_then_smase_on_exact_tie",
            )
            self.assertAlmostEqual(payload["selected"]["D"]["M"]["lr"], 0.004)

    def test_hparam_selection_rejects_legacy_lagtcn_eta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "legacy_eta"
            run.mkdir()
            (run / "config.json").write_text(json.dumps({
                "experiment_stage": select_ae_final_hparams.STAGE,
                "experiment_id": "tuning_a",
                "validation_only": True,
                "num_timesteps_out": 24,
                "training_loss_space": "original",
                "dataset": "D",
                "model_name": "LAGTCN",
                "lr": 0.001,
                "hidden_dim": 8,
                "lagtcn_decoder_mode": "persistence_residual",
                "lagtcn_residual_scale_mode": "fixed",
                "lagtcn_residual_scale_init": 0.1,
                "source_git_commit": "abc123",
                "source_git_branch": "paper/applied-energy",
                "split_protocol_version": TARGET_TIMESTAMP_SPLIT_VERSION,
            }))
            (run / "validation_metrics.json").write_text(json.dumps({
                "WAPE": 1.0,
                "MASE": 1.0,
            }))
            argv = [
                "select_ae_final_hparams.py", "--runs-root", str(root),
                "--experiment-id", "tuning_a",
                "--output", str(root / "selected.json"),
            ]
            with mock.patch.object(select_ae_final_hparams, "DATASETS", ("D",)), \
                 mock.patch.object(select_ae_final_hparams, "MODELS", ("LAGTCN",)), \
                 mock.patch.object(sys, "argv", argv), \
                 self.assertRaisesRegex(ValueError, "eta=1"):
                select_ae_final_hparams.main()

    def test_hparam_selection_deduplicates_retry_and_accepts_audited_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            output = root / "selected.json"
            source_manifest = root / "tuning.jsonl"
            manifest_rows = []
            for index in range(4):
                manifest_rows.append({
                    "experiment_id": "tuning_a",
                    "dataset": "D",
                    "model": "M",
                    "run_label": f"candidate_{index}",
                    "hyperparameters": {
                        "lr": 0.001 * (index + 1),
                        "hidden_dim": 8,
                    },
                })
            source_manifest.write_text(
                "\n".join(json.dumps(row) for row in manifest_rows) + "\n",
                encoding="utf-8",
            )
            terminal_status = root / "status.json"
            terminal_status.write_text(json.dumps({
                "status_protocol_version": select_ae_final_hparams.TERMINAL_STATUS_VERSION,
                "experiment_id": "tuning_a",
                "source_manifest": str(source_manifest),
                "terminal_invalid_configurations": [{
                    "source_index_zero_based": index,
                    "model": "M",
                    "run_label": f"candidate_{index}",
                    "hyperparameters": {
                        "lr": 0.001 * (index + 1),
                        "hidden_dim": 8,
                    },
                    "status": "terminal_invalid_configuration",
                    "failure_type": "numeric_divergence",
                    "selection_action": "exclude_from_validation_selection_without_retry",
                } for index in range(1, 4)],
            }), encoding="utf-8")

            def write_attempt(name, index, *, finite=False, failed=False):
                run = runs / name
                run.mkdir(parents=True)
                (run / "config.json").write_text(json.dumps({
                    "experiment_stage": select_ae_final_hparams.STAGE,
                    "experiment_id": "tuning_a",
                    "validation_only": True,
                    "num_timesteps_out": 24,
                    "training_loss_space": "original",
                    "dataset": "D",
                    "model_name": "M",
                    "lr": 0.001 * (index + 1),
                    "hidden_dim": 8,
                    "source_git_commit": "abc123",
                    "source_git_branch": "paper/applied-energy",
                    "split_protocol_version": TARGET_TIMESTAMP_SPLIT_VERSION,
                }))
                if finite:
                    (run / "validation_metrics.json").write_text(json.dumps({
                        "WAPE": float(3 - index),
                        "MASE": float(3 - index) / 10.0,
                    }))
                if failed:
                    (run / "failure.json").write_text(json.dumps({
                        "failure_type": "cuda_out_of_memory",
                    }))

            write_attempt("candidate_0_original", 0, failed=True)
            write_attempt("candidate_0_retry", 0, finite=True)
            write_attempt("candidate_1", 1)
            write_attempt("candidate_2", 2)
            write_attempt("candidate_3", 3)

            argv = [
                "select_ae_final_hparams.py",
                "--runs-root", str(runs),
                "--experiment-id", "tuning_a",
                "--terminal-status", str(terminal_status),
                "--output", str(output),
            ]
            with mock.patch.object(select_ae_final_hparams, "DATASETS", ("D",)), \
                 mock.patch.object(select_ae_final_hparams, "MODELS", ("M",)), \
                 mock.patch.object(sys, "argv", argv):
                select_ae_final_hparams.main()
            payload = json.loads(output.read_text())
            row = payload["audit"][0]
            self.assertEqual(row["candidate_count"], 4)
            self.assertEqual(row["finite_candidate_count"], 1)
            self.assertEqual(row["ineligible_candidate_count"], 3)
            self.assertTrue(row["all_candidates_terminal"])
            retry_candidate = next(
                candidate for candidate in row["candidates"]
                if candidate["hyperparameters"]["lr"] == 0.001
            )
            self.assertEqual(len(retry_candidate["attempts"]), 2)
            self.assertAlmostEqual(payload["selected"]["D"]["M"]["lr"], 0.001)
    def test_hparam_selection_rejects_any_nonfinite_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "selected.json"
            for index in range(4):
                run = root / f"candidate_{index}"
                run.mkdir()
                (run / "config.json").write_text(json.dumps({
                    "experiment_stage": select_ae_final_hparams.STAGE,
                    "experiment_id": "tuning_a",
                    "validation_only": True,
                    "num_timesteps_out": 24,
                    "training_loss_space": "original",
                    "dataset": "D",
                    "model_name": "M",
                    "lr": 0.001 * (index + 1),
                    "hidden_dim": 8,
                    "source_git_commit": "abc123",
                    "source_git_branch": "paper/applied-energy",
                    "split_protocol_version": TARGET_TIMESTAMP_SPLIT_VERSION,
                }))
                if index < 3:
                    (run / "validation_metrics.json").write_text(json.dumps({
                        "WAPE": float(4 - index)
                    }))
            argv = [
                "select_ae_final_hparams.py", "--runs-root", str(root),
                "--experiment-id", "tuning_a", "--output", str(output),
            ]
            with mock.patch.object(select_ae_final_hparams, "DATASETS", ("D",)), \
                 mock.patch.object(select_ae_final_hparams, "MODELS", ("M",)), \
                 mock.patch.object(sys, "argv", argv), \
                 self.assertRaisesRegex(RuntimeError, "all four preregistered candidates"):
                select_ae_final_hparams.main()


if __name__ == "__main__":
    unittest.main()
