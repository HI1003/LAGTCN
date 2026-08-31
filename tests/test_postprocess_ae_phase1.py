from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (CODE_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import mase
import materialize_ae_run_predictions as materialize
import postprocess_ae_phase1 as pp


def build_synthetic_dataset(data_root: Path, name: str = "Synth_3level") -> dict:
    """3-level dataset: nodes [T, M1, M2, B1, B2, B3]; coherent original series."""
    rng = np.random.default_rng(7)
    dataset_dir = data_root / name
    dataset_dir.mkdir(parents=True)

    T = 60
    bottoms = 50.0 + np.cumsum(rng.normal(0.0, 2.0, size=(T, 3)), axis=0)
    bottoms = np.abs(bottoms) + 1.0
    m1 = bottoms[:, :2].sum(axis=1)
    m2 = bottoms[:, 2]
    top = bottoms.sum(axis=1)
    original = np.column_stack([top, m1, m2, bottoms])  # [T, 6]

    train_T = 48
    log_values = np.log1p(original)
    mean = float(log_values[:train_T].mean())
    std = float(log_values[:train_T].std())
    normalized = (log_values - mean) / std
    np.save(dataset_dir / "node_values.npy", normalized[:, :, None].astype(np.float32))
    np.save(dataset_dir / "normalization_params.npy", {
        "use_log": True, "log_offset": 1.0, "norm_method": "zscore",
        "mean": mean, "std": std, "train_ratio": 0.8, "train_T": train_T, "total_T": T,
    })

    node_order = ["TOP", "M1", "M2", "B1", "B2", "B3"]
    S = np.vstack([
        np.ones((1, 3)),
        np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
    ])
    pd.DataFrame(S).to_csv(dataset_dir / "sum_matrix.csv", header=False, index=False)
    (dataset_dir / "hierarchy_info.json").write_text(json.dumps({
        "num_total_nodes": 6, "num_bottom_nodes": 3, "bottom_start_idx": 3,
        "num_mid_nodes": 2, "top_nodes": ["TOP"], "mid_nodes": ["M1", "M2"],
        "bottom_nodes": ["B1", "B2", "B3"], "node_order": node_order,
        "middle_levels": [[1, 2]],
    }))
    return {"original": original, "train_T": train_T, "node_order": node_order, "S": S}


def _wide_frame(values: np.ndarray, node_order: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    num_samples, num_nodes, num_h = values.shape
    if num_h == 1:
        return pd.DataFrame(values[:, :, 0], index=index, columns=node_order)
    columns = [f"{node}_t+{h + 1}" for h in range(num_h) for node in node_order]
    return pd.DataFrame(
        values.transpose(0, 2, 1).reshape(num_samples, num_h * num_nodes),
        index=index, columns=columns)


def write_run(run_dir: Path, dataset_dir_name: str, node_order: list[str],
              y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "LAGTCN",
              seed: int = 42, true_index_shift_hours: int = 0) -> None:
    """y_true/y_pred: [S, N, H] written in the save_predictions wide format."""
    run_dir.mkdir(parents=True)
    index = pd.date_range("2017-06-01", periods=y_pred.shape[0], freq="h")
    true_index = index + pd.Timedelta(hours=true_index_shift_hours)
    _wide_frame(y_pred, node_order, index).to_csv(run_dir / "pred.csv")
    _wide_frame(y_true, node_order, true_index).to_csv(run_dir / "true.csv")
    (run_dir / "config.json").write_text(json.dumps({
        "raw_data_dir": f"/somewhere/{dataset_dir_name}",
        "model_name": model_name, "seed": seed,
        "graph_mode": "H+A", "stgnn_graph_source": "hybrid",
        "feature_set": "target", "experiment_id": "fixture-exp",
        "output_namespace": "ae/fixture",
        "timestamp": "20260701_000000",
    }))


def write_legacy_run(run_dir: Path, dataset_dir_name: str, node_order: list[str],
                     y_true: np.ndarray, y_base: np.ndarray, y_reconciled: np.ndarray,
                     model_name: str = "GCN-GRU-LP", seed: int = 43) -> None:
    """Legacy layout: {prefix}_{model}_{timestamp}.csv + model_info with nested config."""
    run_dir.mkdir(parents=True)
    stem = f"{model_name}_20260601-000000"
    index = pd.date_range("2017-06-01", periods=y_base.shape[0], freq="h")
    _wide_frame(y_base, node_order, index).to_csv(run_dir / f"base_predictions_{stem}.csv")
    _wide_frame(y_reconciled, node_order, index).to_csv(run_dir / f"predictions_{stem}.csv")
    _wide_frame(y_true, node_order, index).to_csv(run_dir / f"true_values_{stem}.csv")
    (run_dir / f"model_info_{stem}.json").write_text(json.dumps({
        "model_name": model_name,
        "timestamp": "20260601-000000",
        "params": {"total": 1, "trainable": 1},
        "config": {"raw_data_dir": f"/somewhere/{dataset_dir_name}", "seed": seed},
    }))


class BoundaryMaskTest(unittest.TestCase):
    def test_triangular_drop_counts(self) -> None:
        mask = pp.boundary_cell_mask(5, 3, "triangular")
        # horizon h (1-based) drops the first H-h origins
        np.testing.assert_array_equal(mask[:, 0], [False, False, True, True, True])
        np.testing.assert_array_equal(mask[:, 1], [False, True, True, True, True])
        np.testing.assert_array_equal(mask[:, 2], [True, True, True, True, True])

    def test_row_trim_and_h1_empty(self) -> None:
        mask = pp.boundary_cell_mask(5, 3, "row_trim")
        self.assertFalse(mask[:2].any())
        self.assertTrue(mask[2:].all())
        self.assertTrue(pp.boundary_cell_mask(5, 1, "triangular").all())
        self.assertTrue(pp.boundary_cell_mask(5, 1, "row_trim").all())

    def test_h1_sanity_tolerates_only_machine_precision(self) -> None:
        records = [
            {
                "method": "base",
                "level": "bottom_level",
                "horizon_label": "all",
                "boundary_variant": variant,
                "wape": value,
            }
            for variant, value in zip(
                pp.BOUNDARY_VARIANTS,
                [4.331453040458503, 4.3314530404584985, 4.3314530404584985],
            )
        ]
        pp.assert_h1_boundary_metrics_equal(records, "fixture")
        records[-1]["wape"] += 1e-6
        with self.assertRaises(RuntimeError):
            pp.assert_h1_boundary_metrics_equal(records, "fixture")



class PostprocessEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data_root = root / "Data"
        self.runs_root = root / "runs"
        self.output_dir = root / "out"
        self.fixture = build_synthetic_dataset(self.data_root)

        node_order = self.fixture["node_order"]
        original = self.fixture["original"]
        rng = np.random.default_rng(11)

        # Run 1 (H=3): true block from the series tail, pred = true + 1 on TOP only.
        S_count, H = 5, 3
        base = np.stack([original[50 + s: 50 + s + H].T for s in range(S_count)])  # [S, 6, H]
        self.y_true = base
        self.y_pred = base.copy()
        self.y_pred[:, 0, :] += 1.0
        write_run(self.runs_root / "Synth_3level" / "run_h3", "Synth_3level",
                  node_order, self.y_true, self.y_pred)

        # Run 2 (H=3): contains a NaN -> quarantined.
        bad_pred = self.y_pred.copy()
        bad_pred[1, 2, 1] = np.nan
        write_run(self.runs_root / "Synth_3level" / "run_nan", "Synth_3level",
                  node_order, self.y_true, bad_pred, model_name="DLINEAR", seed=43)

        # Run 3 (H=1): plain columns; identical pred/true -> all zero errors.
        y1 = np.abs(rng.normal(80.0, 5.0, size=(6, 6, 1))) + 1.0
        write_run(self.runs_root / "Synth_3level" / "run_h1", "Synth_3level",
                  node_order, y1, y1.copy(), model_name="PATCHTST", seed=44)

        # Run 4 (legacy layout): base file must win over the reconciled
        # predictions_* file, and config comes from nested model_info config.
        write_legacy_run(self.runs_root / "Synth_3level" / "run_legacy", "Synth_3level",
                         node_order, self.y_true, self.y_pred,
                         y_reconciled=self.y_true * 2.0, seed=43)

        # Run 5: true.csv time index shifted by one hour -> quarantined.
        write_run(self.runs_root / "Synth_3level" / "run_badidx", "Synth_3level",
                  node_order, y1, y1.copy(), model_name="NHITS", seed=45,
                  true_index_shift_hours=1)

        self.manifest = pp.run_postprocess(
            self.runs_root, self.data_root, self.output_dir, ["ols"])

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_manifest_counts(self) -> None:
        self.assertEqual(self.manifest["n_runs_discovered"], 5)
        self.assertEqual(self.manifest["n_runs_ok"], 3)
        self.assertEqual(self.manifest["n_runs_quarantined"], 2)
        self.assertEqual(self.manifest["n_runs_errored"], 0)
        self.assertEqual(self.manifest["postprocess_version"], pp.POSTPROCESS_VERSION)

    def test_deterministic_shards_partition_runs(self) -> None:
        shard_manifests = []
        shard_run_ids = []
        for shard_index in range(2):
            output = self.output_dir.parent / f"shard_{shard_index}"
            manifest = pp.run_postprocess(
                self.runs_root,
                self.data_root,
                output,
                ["ols"],
                shard_count=2,
                shard_index=shard_index,
            )
            shard_manifests.append(manifest)
            shard_run_ids.append(
                {
                    summary["run_id"]
                    for summary in manifest["run_summaries"]
                }
                | {
                    item["run_id"]
                    for item in json.loads(
                        (output / "quarantine.json").read_text()
                    )
                }
            )
        self.assertTrue(shard_run_ids[0].isdisjoint(shard_run_ids[1]))
        self.assertEqual(shard_run_ids[0] | shard_run_ids[1], {
            str(path.relative_to(self.runs_root))
            for path in pp.discover_runs(self.runs_root)
        })
        self.assertEqual([m["shard_index"] for m in shard_manifests], [0, 1])


    def test_quarantine_reasons(self) -> None:
        quarantine = json.loads((self.output_dir / "quarantine.json").read_text())
        by_run = {q["run_id"]: q for q in quarantine}
        self.assertEqual(by_run["Synth_3level/run_nan"]["reason"], "non_finite_values")
        self.assertEqual(by_run["Synth_3level/run_nan"]["n_nonfinite_pred"], 1)
        self.assertEqual(by_run["Synth_3level/run_badidx"]["reason"],
                         "time_index_mismatch_between_pred_and_true")
        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        self.assertNotIn("Synth_3level/run_nan", set(metrics["run_id"]))
        self.assertNotIn("Synth_3level/run_badidx", set(metrics["run_id"]))

    def test_legacy_layout_prefers_base_and_merges_nested_config(self) -> None:
        summary = {s["run_id"]: s for s in self.manifest["run_summaries"]}
        legacy = summary["Synth_3level/run_legacy"]
        self.assertEqual(legacy["status"], "ok")
        self.assertEqual(legacy["seed"], 43)  # from model_info's nested config
        self.assertEqual(legacy["model_name"], "GCN-GRU-LP")

        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        row = metrics[(metrics.run_id == "Synth_3level/run_legacy") & (metrics.method == "base")
                      & (metrics.boundary_variant == "none") & (metrics.level == "top_level")
                      & (metrics.horizon_label == "all")].iloc[0]
        # base_predictions_* (true + 1 on TOP) was used, not predictions_* (true * 2).
        self.assertAlmostEqual(row["mae"], 1.0, places=9)

    def test_csv_outputs_carry_provisional_tag(self) -> None:
        for name in ["phase1_metrics_long.csv", "negative_prediction_rates.csv",
                     "clamp_trigger_rates.csv"]:
            df = pd.read_csv(self.output_dir / name)
            self.assertTrue((df["postprocess_version"] == pp.POSTPROCESS_VERSION).all(), name)
            self.assertTrue((df["result_tag"] == "provisional_diagnostic").all(), name)
        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        self.assertTrue((metrics["mase_version"] == mase.MASE_VERSION).all())
        self.assertTrue((metrics["mase_label"] == "sMASE-24").all())
        self.assertTrue((metrics["mase_seasonal_period"] == 24).all())

    def test_diagnostic_weights_are_gated_and_separated(self) -> None:
        out2 = self.output_dir.parent / "out_diag"
        pp.run_postprocess(self.runs_root, self.data_root, out2, ["ols", "wls"])
        main_csv = pd.read_csv(out2 / "phase1_metrics_long.csv")
        self.assertNotIn("mint_wls", set(main_csv["method"]))
        diag_csv = pd.read_csv(out2 / "phase1_metrics_diagnostic_weights.csv")
        self.assertEqual(set(diag_csv["method"]), {"mint_wls"})
        self.assertTrue((diag_csv["weight_estimation_source"] == "test_targets").all())
        diagnostics = json.loads((out2 / "reconciliation_diagnostics.json").read_text())
        self.assertEqual(
            diagnostics["Synth_3level/run_h3"]["mint_wls"]["weight_estimation_source"],
            "test_targets")

    def test_main_rejects_ungated_wls_and_uses_smoke_dir_with_limit(self) -> None:
        with self.assertRaises(SystemExit):
            pp.main(["--runs-root", str(self.runs_root), "--mint-weights", "ols,wls"])
        rc = pp.main(["--runs-root", str(self.runs_root), "--limit", "1"])
        self.assertEqual(rc, 0)
        smoke = self.runs_root / "_postprocess_ae_phase1_smoke"
        self.assertTrue((smoke / "postprocess_manifest.json").exists())
        self.assertFalse((self.runs_root / "_postprocess_ae_phase1").exists())

    def test_base_metrics_hand_computed(self) -> None:
        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        row = metrics[(metrics.run_id == "Synth_3level/run_h3") & (metrics.method == "base")
                      & (metrics.boundary_variant == "none") & (metrics.level == "top_level")
                      & (metrics.horizon_label == "all")].iloc[0]
        top_true = np.abs(self.y_true[:, 0, :])
        self.assertAlmostEqual(row["wape"], 100.0 * top_true.size / top_true.sum(), places=9)
        expected_scale = mase.compute_naive_scale(
            self.fixture["original"][: self.fixture["train_T"]])
        # node_values.npy round-trips through float32, so allow ~1e-6 relative error.
        self.assertAlmostEqual(row["mase"] / (1.0 / expected_scale[0]), 1.0, places=5)
        self.assertAlmostEqual(row["mae"], 1.0, places=9)

    def test_bu_repairs_top_only_error(self) -> None:
        # Bottom forecasts equal the truth, so BU rebuilds the truth everywhere.
        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        rows = metrics[(metrics.run_id == "Synth_3level/run_h3") & (metrics.method == "bu")
                       & (metrics.boundary_variant == "none") & (metrics.horizon_label == "all")]
        for _, row in rows.iterrows():
            self.assertAlmostEqual(row["wape"], 0.0, places=9)
            self.assertAlmostEqual(row["mase"], 0.0, places=9)

    def test_methods_and_variants_present(self) -> None:
        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        run_rows = metrics[metrics.run_id == "Synth_3level/run_h3"]
        self.assertEqual(set(run_rows.method), {"base", "bu", "td_fp", "mint_ols"})
        self.assertEqual(set(run_rows.boundary_variant), {"none", "triangular", "row_trim"})
        levels = set(run_rows.level)
        self.assertIn("middle1_level", levels)
        self.assertNotIn("middle2_level", levels)

    def test_triangular_masks_change_h3_but_not_h1_task(self) -> None:
        metrics = pd.read_csv(self.output_dir / "phase1_metrics_long.csv")
        h1 = metrics[(metrics.run_id == "Synth_3level/run_h1") & (metrics.method == "base")
                     & (metrics.level == "All")]
        wapes = h1.groupby("boundary_variant")["wape"].first()
        self.assertAlmostEqual(wapes["none"], wapes["triangular"], places=12)
        self.assertAlmostEqual(wapes["none"], wapes["row_trim"], places=12)

    def test_diagnostics_and_scale_metadata_written(self) -> None:
        diagnostics = json.loads((self.output_dir / "reconciliation_diagnostics.json").read_text())
        run_diag = diagnostics["Synth_3level/run_h3"]
        self.assertLessEqual(run_diag["mint_ols"]["coherence_residual_max_abs"], 1e-6)
        self.assertEqual(run_diag["bu"]["n_failures"], 0)
        # Base forecast is incoherent (+1 on TOP only); truth is coherent.
        self.assertGreater(run_diag["base"]["coherence_residual_max_abs"], 0.5)
        self.assertLessEqual(run_diag["true_values"]["coherence_residual_max_abs"], 1e-4)

        scale_meta = json.loads((self.output_dir / "mase_scale_Synth_3level.json").read_text())
        self.assertEqual(scale_meta["train_length"], self.fixture["train_T"])
        self.assertEqual(scale_meta["num_degenerate_nodes"], 0)
        self.assertEqual(scale_meta["result_tag"], "provisional_diagnostic")
        self.assertEqual(scale_meta["mase_label"], "sMASE-24")
        self.assertEqual(scale_meta["seasonal_period"], 24)


    def test_provenance_fields_and_reconciled_archives(self) -> None:
        archive_out = self.output_dir.parent / "out_archives"
        manifest = pp.run_postprocess(
            self.runs_root,
            self.data_root,
            archive_out,
            ["ols"],
            save_reconciled=True,
        )
        self.assertEqual(manifest["n_reconciled_archives"], 3)
        table = pd.read_csv(archive_out / "reconciled_predictions_manifest.csv")
        self.assertEqual(len(table), 3)
        row = table[table.run_id == "Synth_3level/run_h3"].iloc[0]
        self.assertEqual(row["graph_mode"], "H+A")
        self.assertEqual(row["stgnn_graph_source"], "hybrid")
        self.assertEqual(row["feature_set"], "target")

        archive = archive_out / row["archive_file"]
        self.assertTrue(archive.exists())
        with np.load(archive, allow_pickle=False) as saved:
            self.assertEqual(
                set(saved.files),
                {"bu", "td_fp", "mint_ols", "time_index", "node_order",
                 "archive_version", "postprocess_version", "reconcile_version",
                 "source_run_id", "source_prediction_file"},
            )
            self.assertEqual(saved["bu"].dtype, np.float64)
            np.testing.assert_allclose(
                saved["bu"], self.y_true, rtol=0.0, atol=1e-10)
            self.assertEqual(
                saved["archive_version"].item(), pp.RECONCILED_ARCHIVE_VERSION)

        metrics = pd.read_csv(archive_out / "phase1_metrics_long.csv")
        for field in ["graph_mode", "stgnn_graph_source", "feature_set",
                      "prediction_file", "hierarchy_source"]:
            self.assertIn(field, metrics.columns)

    def test_validated_reconciliation_archives_are_reused_for_rescoring(self) -> None:
        archive_out = self.output_dir.parent / "out_archive_reuse"
        first = pp.run_postprocess(
            self.runs_root, self.data_root, archive_out, ["ols"],
            save_reconciled=True,
        )
        self.assertEqual(first["n_reconciled_archives"], 3)
        before_metrics = pd.read_csv(archive_out / "phase1_metrics_long.csv")
        archive_table = pd.read_csv(
            archive_out / "reconciled_predictions_manifest.csv"
        )
        mtimes = {
            row.run_id: (archive_out / row.archive_file).stat().st_mtime_ns
            for row in archive_table.itertuples(index=False)
        }

        with mock.patch.object(
            pp.reconcile_ae,
            "apply_reconciliation_ae",
            side_effect=AssertionError("solver must not run for a valid cache"),
        ):
            second = pp.run_postprocess(
                self.runs_root, self.data_root, archive_out, ["ols"],
                save_reconciled=True, reuse_reconciled=True,
            )

        self.assertEqual(second["n_reconciled_archives_reused"], 3)
        self.assertEqual(second["n_reconciled_archives_computed"], 0)
        after_table = pd.read_csv(
            archive_out / "reconciled_predictions_manifest.csv"
        )
        self.assertTrue(after_table["archive_reused"].all())
        for row in after_table.itertuples(index=False):
            self.assertEqual(
                (archive_out / row.archive_file).stat().st_mtime_ns,
                mtimes[row.run_id],
            )
        after_metrics = pd.read_csv(archive_out / "phase1_metrics_long.csv")
        keys = [
            "run_id", "method", "boundary_variant", "level", "horizon_label"
        ]
        numeric = ["mase", "mae", "rmse", "wape"]
        left = before_metrics.sort_values(keys).reset_index(drop=True)
        right = after_metrics.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(left[keys + numeric], right[keys + numeric])
        diagnostics = json.loads(
            (archive_out / "reconciliation_diagnostics.json").read_text()
        )
        self.assertTrue(
            diagnostics["Synth_3level/run_h3"]["mint_ols"][
                "reused_reconciled_archive"
            ]
        )

    def test_invalid_reconciliation_archive_falls_back_to_solver(self) -> None:
        archive_out = self.output_dir.parent / "out_archive_fallback"
        pp.run_postprocess(
            self.runs_root, self.data_root, archive_out, ["ols"],
            save_reconciled=True,
        )
        table = pd.read_csv(
            archive_out / "reconciled_predictions_manifest.csv"
        )
        row = table[table.run_id == "Synth_3level/run_h3"].iloc[0]
        archive_path = archive_out / row["archive_file"]
        with np.load(archive_path, allow_pickle=False) as saved:
            payload = {key: saved[key] for key in saved.files}
        payload["node_order"] = payload["node_order"][::-1]
        np.savez_compressed(archive_path, **payload)

        manifest = pp.run_postprocess(
            self.runs_root, self.data_root, archive_out, ["ols"],
            save_reconciled=True, reuse_reconciled=True,
        )
        self.assertEqual(manifest["n_reconciled_archives_reused"], 2)
        self.assertEqual(manifest["n_reconciled_archives_computed"], 1)
        summaries = {row["run_id"]: row for row in manifest["run_summaries"]}
        self.assertIn(
            "node_order mismatch",
            summaries["Synth_3level/run_h3"][
                "reconciled_archive_reuse_error"
            ],
        )



    def test_clamp_provenance_uses_introduction_time_and_refreshes(self) -> None:
        pre = pp.clamp_provenance("TIMESNET", "20260619_120000")
        post = pp.clamp_provenance("TIMESNET", "20260623_120000")
        self.assertTrue(pre["clamp_model_family"])
        self.assertFalse(pre["clamp_expected"])
        self.assertTrue(post["clamp_expected"])
        self.assertIn("inferred", pre["clamp_provenance"])

        path = self.output_dir / "clamp_trigger_rates.csv"
        table = pd.read_csv(path)
        table["clamp_expected"] = False
        table.to_csv(path, index=False)
        summary = pp.refresh_clamp_provenance_table(
            self.runs_root, self.output_dir)
        refreshed = pd.read_csv(path).set_index("run_id")
        self.assertEqual(summary["n_rows"], 3)
        self.assertTrue(
            bool(refreshed.loc["Synth_3level/run_h3", "clamp_expected"]))
        self.assertIn("clamp_introduced_commit", refreshed.columns)
    def test_negative_and_clamp_reports(self) -> None:
        negative = pd.read_csv(self.output_dir / "negative_prediction_rates.csv")
        run_neg = negative[negative.run_id == "Synth_3level/run_h3"]
        self.assertTrue((run_neg["negative_fraction"] == 0.0).all())

        clamp = pd.read_csv(self.output_dir / "clamp_trigger_rates.csv")
        by_run = clamp.set_index("run_id")
        self.assertTrue(bool(by_run.loc["Synth_3level/run_h3", "clamp_expected"]))
        self.assertFalse(bool(by_run.loc["Synth_3level/run_h1", "clamp_expected"]))
        self.assertEqual(float(by_run.loc["Synth_3level/run_h3", "lower_hit_fraction"]), 0.0)




class MaterializeRunPredictionsTest(unittest.TestCase):
    def test_materializes_four_predictions_and_explicit_metrics_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "Data"
            dataset = "Synth_3level"
            fixture = build_synthetic_dataset(data_root, dataset)
            ae_root = data_root / dataset / "output" / "ae"
            run_dir = ae_root / "fixture" / "run_h3"
            original = fixture["original"]
            y_true = np.stack([
                original[50 + sample: 53 + sample].T
                for sample in range(5)
            ])
            y_pred = y_true.copy()
            y_pred[:, 0, :] += 1.0
            write_run(
                run_dir,
                dataset,
                fixture["node_order"],
                y_true,
                y_pred,
            )
            output_dir = ae_root / "_postprocess_ae_phase1"
            pp.run_postprocess(
                ae_root,
                data_root,
                output_dir,
                ["ols"],
                save_reconciled=True,
            )

            summary = materialize.materialize_dataset(
                dataset,
                data_root=data_root,
                workers=1,
            )
            self.assertEqual(summary["n_runs"], 1)
            self.assertEqual(summary["n_base_renamed"], 1)
            self.assertEqual(summary["n_written"], 3)
            self.assertFalse((run_dir / "pred.csv").exists())

            expected_files = {
                "base": "base_pred.csv",
                "bu": "bu_recon_pred.csv",
                "td_fp": "td_recon_pred.csv",
                "mint_ols": "mint_recon_pred.csv",
            }
            for filename in expected_files.values():
                self.assertTrue((run_dir / filename).is_file(), filename)

            base, nodes, index = pp.parse_prediction_csv(run_dir / "base_pred.csv")
            np.testing.assert_allclose(base, y_pred, rtol=0.0, atol=1e-10)
            archive = (
                output_dir / "reconciled_predictions" / "fixture" / "run_h3"
                / "reconciled_predictions.npz"
            )
            with np.load(archive, allow_pickle=False) as saved:
                for method in ("bu", "td_fp", "mint_ols"):
                    actual, actual_nodes, actual_index = pp.parse_prediction_csv(
                        run_dir / expected_files[method]
                    )
                    np.testing.assert_allclose(
                        actual, saved[method], rtol=0.0, atol=1e-10
                    )
                    self.assertEqual(actual_nodes, nodes)
                    self.assertEqual(list(actual_index), list(index))

            local_metrics = pd.read_csv(run_dir / "reconciliation_metrics.csv")
            self.assertEqual(set(local_metrics["method"]), set(expected_files))
            for method, filename in expected_files.items():
                rows = local_metrics[local_metrics.method == method]
                self.assertTrue((rows.prediction_file == filename).all())
                expected_role = "base" if method == "base" else "reconciled"
                expected_reconciliation = "none" if method == "base" else method
                self.assertTrue((rows.prediction_role == expected_role).all())
                self.assertTrue(
                    (rows.reconciliation_method == expected_reconciliation).all()
                )

            global_metrics = pd.read_csv(output_dir / "phase1_metrics_long.csv")
            self.assertEqual(
                set(global_metrics.prediction_file), set(expected_files.values())
            )
            file_manifest = pd.read_csv(
                output_dir / "prediction_files_manifest.csv"
            )
            self.assertEqual(len(file_manifest), 4)
            self.assertEqual(
                set(file_manifest.prediction_file), set(expected_files.values())
            )

            rerun = materialize.materialize_dataset(
                dataset,
                data_root=data_root,
                workers=1,
            )
            self.assertEqual(rerun["n_base_renamed"], 0)
            self.assertEqual(rerun["n_written"], 0)
            self.assertEqual(rerun["n_skipped"], 3)
            self.assertEqual(
                len(pd.read_csv(output_dir / "prediction_files_manifest.csv")), 4
            )
if __name__ == "__main__":
    unittest.main()
