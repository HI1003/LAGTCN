#!/usr/bin/env python3
"""Submission postprocessing for frozen Applied Energy direct-24 forecasts.

Consumes saved test Base forecasts and saved validation forecasts. It never
loads or updates a forecasting model. BU, TD-FP and horizon-wise MinT-SHR are
nonnegative and coherent by construction; MinT covariance sees validation
residuals only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT / "code", ROOT / "scripts"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

import postprocess_ae_phase1 as phase1
import reconcile_ae
from data_loader import TARGET_TIMESTAMP_SPLIT_VERSION
from graph_sparsity import (
    FINAL_GRAPH_SOURCE_POLICY,
    GRAPH_DESIGN_PROTOCOL_VERSION,
)
from output_naming import LAGTCN_GRAPH_SOURCE_VERSION_CURRENT, graph_components
from postprocess_ae_mint_shrink import (
    estimate_horizon_shrink_covariances,
    reconcile_horizonwise_mint_shrink,
)
from build_ae_final_manifest import (
    CURRENT_MODEL_CONFIGS,
    FORMAL_EFFECTIVE_BATCH_SIZE,
    DATASETS,
    GRAPH_SELECTION_PROTOCOL_VERSION,
    LAGTCN_GRAPHS,
    formal_batch_protocol,
    SEEDS,
)

VERSION = "ae_final_postprocess_direct24_v4_scoped_provenance"
FORMAL_STAGE = "ae_final_main_v1"


def parse_dataset_scope(value: str | None) -> tuple[str, ...]:
    """Return a canonical, non-empty subset of the formal datasets."""
    if value is None:
        return tuple(DATASETS)
    requested = {item.strip() for item in value.split(",") if item.strip()}
    if not requested:
        raise ValueError("--datasets must name at least one formal dataset.")
    unknown = sorted(requested - set(DATASETS))
    if unknown:
        raise ValueError(f"Unknown formal datasets: {unknown}.")
    return tuple(dataset for dataset in DATASETS if dataset in requested)


def write_prediction(path: Path, values: np.ndarray, nodes: list[str], index: pd.Index) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(f"Expected [origins,nodes,horizons], got {values.shape}")
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{path}: prediction contains NaN or Inf.")
    origins, num_nodes, horizons = values.shape
    if len(nodes) != num_nodes or len(index) != origins:
        raise ValueError(f"{path}: prediction metadata does not match shape {values.shape}")
    columns = [f"{node}_t+{h}" for h in range(1, horizons + 1) for node in nodes]
    frame = pd.DataFrame(
        values.transpose(0, 2, 1).reshape(origins, horizons * num_nodes),
        index=index,
        columns=columns,
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary)
    temporary.replace(path)


def validate_smase_metadata(run_dir: Path, config: dict, meta) -> np.ndarray:
    recorded = config.get("smase_scale_metadata")
    if not isinstance(recorded, dict):
        raise ValueError(f"{run_dir}: missing frozen smase_scale_metadata.")
    if recorded.get("mase_version") != meta.scale_metadata.get("mase_version"):
        raise ValueError(f"{run_dir}: sMASE version differs from dataset metadata.")
    if int(recorded.get("seasonal_period", 0)) != 24:
        raise ValueError(f"{run_dir}: formal sMASE seasonal period is not 24.")
    if int(recorded.get("train_length", -1)) != int(meta.train_length):
        raise ValueError(f"{run_dir}: sMASE train boundary differs from the frozen dataset.")
    scale = np.asarray(recorded.get("scale_per_node"), dtype=np.float64)
    # Training freezes this scale after the float32 tensor inverse transform,
    # whereas dataset metadata independently reconstructs it in float64. Keep
    # the run-frozen denominator after verifying float32-level equivalence.
    if scale.shape != meta.naive_scale.shape or not np.allclose(
        scale, meta.naive_scale, rtol=1e-6, atol=1e-8
    ):
        raise ValueError(f"{run_dir}: run and postprocessor sMASE scales differ.")
    return scale


def validate_output(values: np.ndarray, S: np.ndarray, bottom_start: int, method: str) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{method}: output contains NaN or Inf.")
    minimum = float(values.min())
    if minimum < -1e-10:
        raise RuntimeError(f"{method}: negative reconciled forecast {minimum}.")
    bottom = values[:, bottom_start:bottom_start + S.shape[1], :]
    rebuilt = np.einsum("nb,sbh->snh", S, bottom)
    residual = np.abs(values - rebuilt)
    maximum = float(residual.max()) if residual.size else 0.0
    if maximum > 1e-8:
        raise RuntimeError(f"{method}: coherence residual {maximum} exceeds 1e-8.")
    return {
        "minimum_prediction": minimum,
        "coherence_residual_max_abs": maximum,
        "coherence_residual_mean_abs": float(residual.mean()) if residual.size else 0.0,
    }



def validate_coherence_only(
    values: np.ndarray,
    S: np.ndarray,
    bottom_start: int,
    method: str,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError(f"{method}: output contains NaN or Inf.")
    bottom = values[:, bottom_start:bottom_start + S.shape[1], :]
    rebuilt = np.einsum("nb,sbh->snh", S, bottom)
    residual = np.abs(values - rebuilt)
    maximum = float(residual.max()) if residual.size else 0.0
    if maximum > 1e-8:
        raise RuntimeError(f"{method}: coherence residual {maximum} exceeds 1e-8.")
    return {
        "coherence_residual_max_abs": maximum,
        "coherence_residual_mean_abs": float(residual.mean()) if residual.size else 0.0,
    }


def has_current_diagnostics(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return payload.get("postprocess_version") == VERSION


def assert_mint_nnls_success(run_dir: Path, diagnostic: dict) -> None:
    if int(diagnostic.get("n_failures", 0)) == 0:
        return
    failed_horizons = [
        {"horizon": item.get("horizon"), "n_failures": item.get("n_failures")}
        for item in diagnostic.get("horizon_diagnostics", [])
        if int(item.get("n_failures", 0)) > 0
    ]
    raise RuntimeError(
        f"{run_dir}: MinT-SHR NNLS failed; refusing fallback output: {failed_horizons}."
    )


def validate_prediction_contract(
    run_dir: Path,
    config: dict,
    values: np.ndarray,
    nodes: list[str],
    index: pd.Index,
    split_name: str,
) -> None:
    """Verify a saved formal prediction against its frozen split metadata."""
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[2] != 24:
        raise ValueError(
            f"{run_dir}: {split_name} prediction must have shape [origins,nodes,24], "
            f"got {values.shape}."
        )
    recorded_nodes = config.get("node_names")
    if not isinstance(recorded_nodes, list) or [str(v) for v in recorded_nodes] != nodes:
        raise ValueError(f"{run_dir}: {split_name} node order differs from config.json.")
    provenance = config.get("split_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{run_dir}: missing split_provenance in config.json.")
    segment = provenance.get("segments", {}).get(split_name)
    if not isinstance(segment, dict):
        raise ValueError(f"{run_dir}: missing {split_name} split provenance.")
    expected_count = int(segment.get("origin_count", -1))
    if values.shape[0] != expected_count:
        raise ValueError(
            f"{run_dir}: {split_name} has {values.shape[0]} origins, expected {expected_count}."
        )
    time_key = "validation_time_index" if split_name == "validation" else "time_index"
    expected_index = config.get(time_key)
    if not isinstance(expected_index, list) or len(expected_index) != expected_count:
        raise ValueError(f"{run_dir}: invalid frozen {time_key} metadata.")
    try:
        actual_times = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
        expected_times = pd.DatetimeIndex(pd.to_datetime(expected_index, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{run_dir}: unparseable {split_name} timestamp index.") from exc
    if not actual_times.equals(expected_times):
        raise ValueError(f"{run_dir}: {split_name} CSV index differs from config.json.")
    if len(actual_times) > 1 and not np.all(
        np.diff(actual_times.asi8) == pd.Timedelta(hours=1).value
    ):
        raise ValueError(f"{run_dir}: {split_name} forecast origins are not hourly stride-1.")


def validate_complete_batch_contract(selected: list[tuple[Path, dict]]) -> None:
    """Fail closed if a nominally complete formal batch mixes configurations."""
    allowed_external = {
        (item["model"], item["graph"]): item for item in CURRENT_MODEL_CONFIGS
    }
    common_budget = set()
    model_hparams: dict[tuple[str, str], set[tuple]] = {}
    graph_controls: dict[tuple[str, str], set[float | int]] = {}
    for run_dir, config in selected:
        model = str(config.get("model_name", "")).upper()
        graph = str(config.get("graph_mode", ""))
        dataset = str(config.get("dataset", ""))
        common_expected = {
            "paper_scope": "journal_applied_energy",
            "experiment_stage": FORMAL_STAGE,
            "feature_set": "target",
            "num_timesteps_in": 168,
            "num_timesteps_out": 24,
            "training_loss_space": "original",
            "validation_only": False,
            "checkpoint_every_epochs": 1,
            "resume": "auto",
        }
        mismatches = {
            key: (config.get(key), expected)
            for key, expected in common_expected.items()
            if config.get(key) != expected
        }
        if mismatches or not np.isclose(float(config.get("coherency_lambda", np.nan)), 0.0):
            raise RuntimeError(
                f"{run_dir}: formal invocation contract mismatch: {mismatches}, "
                f"coherency_lambda={config.get('coherency_lambda')}."
            )
        if str(config.get("source_git_branch")) not in {"main", "paper/applied-energy"}:
            raise RuntimeError(
                f"{run_dir}: formal run was not launched from the public main branch "
                "or the archived paper/applied-energy branch."
            )
        if model == "LAGTCN":
            expected = {
                "gnn_type": "gcn",
                "temporal_type": "patch_transformer",
                "stgnn_graph_source": "project",
                "lagtcn_ablation": "none",
                "lagtcn_decoder_mode": "persistence_residual",
                "lagtcn_residual_scale_mode": "unit",
                "lagtcn_residual_scale_init": 1.0,
            }
        else:
            item = allowed_external.get((model, graph))
            if item is None:
                raise RuntimeError(f"{run_dir}: unsupported formal model/graph pair {model}/{graph}.")
            expected = {
                "gnn_type": item["gnn"],
                "temporal_type": item["temporal"],
                "stgnn_graph_source": item["source"],
            }
            if item["role"] == "end_to_end_coherent":
                expected["prediction_role"] = item["role"]
        model_mismatches = {
            key: (config.get(key), value)
            for key, value in expected.items()
            if config.get(key) != value
        }
        if model_mismatches:
            raise RuntimeError(f"{run_dir}: formal model contract mismatch: {model_mismatches}.")

        physical_batch = int(config.get("batch_size", 0))
        accumulation = int(config.get("gradient_accumulation_steps", 0))
        effective_batch = int(config.get("effective_batch_size", 0))
        if (
            physical_batch < 1
            or accumulation < 1
            or physical_batch * accumulation != effective_batch
            or effective_batch != FORMAL_EFFECTIVE_BATCH_SIZE
        ):
            raise RuntimeError(
                f"{run_dir}: invalid formal batch protocol: physical={physical_batch}, "
                f"accumulation={accumulation}, effective={effective_batch}."
            )
        expected_resource = formal_batch_protocol(
            dataset, model, int(config.get("hidden_dim", 0)), effective_batch
        )
        actual_resource = {
            "physical_batch_size": physical_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
        }
        if actual_resource != expected_resource:
            raise RuntimeError(
                f"{run_dir}: batch resource policy mismatch: "
                f"{actual_resource} != {expected_resource}."
            )
        common_budget.add(tuple(config.get(key) for key in (
            "effective_batch_size", "epochs", "patience", "num_layers"
        )))
        model_hparams.setdefault((dataset, model), set()).add(tuple(
            config.get(key) for key in ("lr", "hidden_dim")
        ))
        components = graph_components(graph)
        for source, field in (
            ("S", "static_threshold"),
            ("A", "adaptive_top_k"),
            ("D", "dynamic_threshold"),
        ):
            if source in components:
                value = config.get(field)
                if value is None:
                    raise RuntimeError(f"{run_dir}: active graph source {source} lacks {field}.")
                graph_controls.setdefault((dataset, field), set()).add(value)

    if len(common_budget) != 1:
        raise RuntimeError(f"Formal batch mixes training budgets: {sorted(common_budget)}.")
    mixed_hparams = {key: values for key, values in model_hparams.items() if len(values) != 1}
    if mixed_hparams:
        raise RuntimeError(f"Formal batch mixes frozen model hyperparameters: {mixed_hparams}.")
    mixed_graphs = {key: values for key, values in graph_controls.items() if len(values) != 1}
    if mixed_graphs:
        raise RuntimeError(f"Formal batch mixes frozen graph controls: {mixed_graphs}.")


def matching_runs(
    runs_root: Path,
    experiment_id: str | None = None,
    *,
    require_complete: bool = False,
    datasets: tuple[str, ...] | None = None,
):
    dataset_scope = tuple(DATASETS) if datasets is None else tuple(datasets)
    if not dataset_scope or not set(dataset_scope).issubset(DATASETS):
        raise ValueError(f"Invalid formal dataset scope: {dataset_scope}.")
    matched = []
    for config_path in sorted(runs_root.rglob("config.json")):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if config.get("experiment_stage") != FORMAL_STAGE:
            continue
        if int(config.get("num_timesteps_out", 0)) != 24:
            continue
        matched.append((config_path.parent, config))
    experiment_ids = sorted({
        str(config.get("experiment_id"))
        for _, config in matched if config.get("experiment_id")
    })
    if experiment_id is None:
        if len(experiment_ids) != 1:
            raise RuntimeError(
                "Formal postprocessing requires exactly one experiment_id; "
                f"found {experiment_ids}. Pass --experiment-id explicitly."
            )
        experiment_id = experiment_ids[0]
    selected = [
        (run_dir, config) for run_dir, config in matched
        if str(config.get("experiment_id")) == experiment_id
        and str(config.get("dataset")) in dataset_scope
    ]
    if not selected:
        raise RuntimeError(f"No {FORMAL_STAGE} runs found for experiment_id={experiment_id!r}.")
    keys = [
        (str(c.get("dataset")), str(c.get("model_name")), str(c.get("graph_mode")), int(c.get("seed")))
        for _, c in selected
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            f"Duplicate formal run keys detected within experiment_id={experiment_id!r}."
        )
    lagtcn_expected = {
        (dataset, "LAGTCN", graph, seed)
        for dataset in dataset_scope for graph in LAGTCN_GRAPHS for seed in SEEDS
    }
    expected = lagtcn_expected | {
        (dataset, item["model"], item["graph"], seed)
        for dataset in dataset_scope for item in CURRENT_MODEL_CONFIGS for seed in SEEDS
    }
    actual_keys = set(keys)
    unexpected = sorted(actual_keys - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected formal run keys: {unexpected}.")
    if require_complete and actual_keys != expected:
        missing = sorted(expected - actual_keys)
        raise RuntimeError(
            f"Formal main batch is not the complete {len(expected)}-run phase-1 "
            f"matrix for datasets={list(dataset_scope)}; "
            f"missing={missing[:10]}."
        )
    selections = {
        (str(c.get("selection_source_experiment_id")), str(c.get("selection_protocol_version")))
        for _, c in selected
    }
    if (
        len(selections) != 1
        or any(value in {"", "None"} for pair in selections for value in pair)
    ):
        raise RuntimeError(
            f"Formal batch spans multiple or missing selections {sorted(selections)}."
        )
    if any(
        c.get("graph_sparsity_policy") != FINAL_GRAPH_SOURCE_POLICY
        for _, c in selected
    ):
        raise RuntimeError(
            f"Formal batch must use graph policy {FINAL_GRAPH_SOURCE_POLICY}."
        )
    if any(
        c.get("split_protocol_version") != TARGET_TIMESTAMP_SPLIT_VERSION
        for _, c in selected
    ):
        raise RuntimeError("Formal batch uses a legacy or missing split protocol.")


    lagtcn_configs = [
        c for _, c in selected if str(c.get("model_name", "")).upper() == "LAGTCN"
    ]
    if any(
        c.get("graph_design_protocol_version") != GRAPH_DESIGN_PROTOCOL_VERSION
        for c in lagtcn_configs
    ):
        raise RuntimeError(
            "Formal batch mixes legacy and current graph-design protocols."
        )

    if any(
        c.get("lagtcn_graph_source_version") != LAGTCN_GRAPH_SOURCE_VERSION_CURRENT
        for c in lagtcn_configs
    ):
        raise RuntimeError(
            "Formal batch mixes legacy and current LAGTCN graph-source implementations."
        )
    data_graph_configs = [
        c for c in lagtcn_configs
        if graph_components(str(c.get("graph_mode", ""))).intersection({"S", "A", "D"})
    ]
    graph_selections = {
        (
            str(c.get("graph_selection_source_experiment_id")),
            str(c.get("graph_selection_protocol_version")),
        )
        for c in data_graph_configs
    }
    if data_graph_configs and (len(graph_selections) != 1 or any(
        value in {"", "None"} for pair in graph_selections for value in pair
    )):
        raise RuntimeError(
            f"Formal data-driven graph runs span graph selections {sorted(graph_selections)}."
        )
    if data_graph_configs and next(iter(graph_selections))[1] != GRAPH_SELECTION_PROTOCOL_VERSION:
        raise RuntimeError(
            "Formal data-driven graph runs use a legacy graph-selection protocol."
        )

    if require_complete:
        validate_complete_batch_contract(selected)

    return experiment_id, selected



def process_deephgnn(run_dir: Path, config: dict, data_root: Path, force: bool):
    """Prepare the nonnegative operational output for the coherent DeepHGNN baseline.

    DeepHGNN has no unreconciled all-node Base forecast, so TD-FP and MinT-SHR
    are not meaningful same-Base comparisons. For deployment, BU simply clips
    its predicted bottom block and rebuilds the hierarchy.
    """
    prediction_path = run_dir / "pred.csv"
    true_path = run_dir / "true.csv"
    output_path = run_dir / "bu_recon_pred.csv"
    metrics_path = run_dir / "reconciliation_metrics_long.csv"
    diagnostics_path = run_dir / "reconciliation_diagnostics.json"
    if (
        not force
        and output_path.is_file()
        and metrics_path.is_file()
        and diagnostics_path.is_file()
        and has_current_diagnostics(diagnostics_path)
    ):
        return {"run_dir": str(run_dir), "status": "skipped_existing_deephgnn"}

    dataset = str(config["dataset"])
    meta = phase1.load_dataset_meta(data_root / dataset)
    meta.naive_scale = validate_smase_metadata(run_dir, config, meta)
    coherent, nodes, index = phase1.parse_prediction_csv(prediction_path)
    true, true_nodes, true_index = phase1.parse_prediction_csv(true_path)
    if coherent.shape != true.shape or nodes != true_nodes or not index.equals(true_index):
        raise ValueError(f"{run_dir}: DeepHGNN prediction/truth mismatch.")
    if nodes != meta.node_order:
        raise ValueError(f"{run_dir}: DeepHGNN node order differs from hierarchy metadata.")
    validate_prediction_contract(run_dir, config, coherent, nodes, index, "test")
    for name, values in (("prediction", coherent), ("true", true)):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{run_dir}: DeepHGNN {name} contains NaN or Inf.")
    e2e_check = validate_coherence_only(
        coherent, meta.sum_matrix, meta.bottom_start_idx, "e2e_coherent"
    )

    bu, bu_diag = reconcile_ae.apply_reconciliation_ae(
        "bu", coherent, meta.sum_matrix, bottom_start_idx=meta.bottom_start_idx
    )
    bu_check = validate_output(bu, meta.sum_matrix, meta.bottom_start_idx, "bu")
    write_prediction(output_path, bu, nodes, index)

    rows = []
    identity = {
        "dataset": dataset,
        "experiment_id": config.get("experiment_id"),
        "model_name": config.get("model_name"),
        "graph_mode": config.get("graph_mode"),
        "seed": int(config.get("seed")),
        "output_length": 24,
        "postprocess_version": VERSION,
        "same_base_rq3_eligible": False,
    }
    for method, values in (("e2e_coherent", coherent), ("bu", bu)):
        for row in phase1.metric_records(true, values, meta, method=method, variant="none"):
            rows.append({**identity, **row})
    frame = pd.DataFrame(rows)
    if frame.isna().any().any():
        raise FloatingPointError(f"{run_dir}: DeepHGNN metrics contain non-finite values.")
    frame.to_csv(metrics_path, index=False)
    diagnostics = {
        "postprocess_version": VERSION,
        "prediction_role": "end_to_end_coherent",
        "rq3_exclusion_reason": "no_unreconciled_all_node_base_forecast",
        "rq4_operational_output": "nonnegative_bottom_projection_then_sum_matrix",
        "methods": {
            "e2e_coherent": e2e_check,
            "bu": {**bu_diag, **bu_check},
        },
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {"run_dir": str(run_dir), "status": "completed_deephgnn_bu_only"}



def process_run(run_dir: Path, config: dict, data_root: Path, nnls_workers: int, force: bool):
    if config.get("prediction_role") == "end_to_end_coherent":
        return process_deephgnn(run_dir, config, data_root, force)

    required = {
        "base": run_dir / "base_pred.csv",
        "true": run_dir / "true.csv",
        "validation": run_dir / "validation_pred.csv",
        "validation_true": run_dir / "validation_true.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing required artifacts {missing}")

    output_paths = {
        "bu": run_dir / "bu_recon_pred.csv",
        "td_fp": run_dir / "td_recon_pred.csv",
        "mint_shrink": run_dir / "mint_recon_pred.csv",
    }
    metrics_path = run_dir / "reconciliation_metrics_long.csv"
    diagnostics_path = run_dir / "reconciliation_diagnostics.json"
    if (
        not force
        and metrics_path.is_file()
        and diagnostics_path.is_file()
        and all(path.is_file() for path in output_paths.values())
        and has_current_diagnostics(diagnostics_path)
    ):
        return {"run_dir": str(run_dir), "status": "skipped_existing"}

    dataset = str(config["dataset"])
    meta = phase1.load_dataset_meta(data_root / dataset)
    meta.naive_scale = validate_smase_metadata(run_dir, config, meta)
    base, nodes, test_index = phase1.parse_prediction_csv(required["base"])
    true, true_nodes, true_index = phase1.parse_prediction_csv(required["true"])
    validation, validation_nodes, validation_index = phase1.parse_prediction_csv(
        required["validation"]
    )
    validation_true, validation_true_nodes, validation_true_index = phase1.parse_prediction_csv(
        required["validation_true"]
    )
    if base.shape != true.shape or nodes != true_nodes or not test_index.equals(true_index):
        raise ValueError(f"{run_dir}: Base/test truth shape or metadata mismatch.")
    if (
        validation.shape != validation_true.shape
        or validation_nodes != validation_true_nodes
        or not validation_index.equals(validation_true_index)
    ):
        raise ValueError(f"{run_dir}: validation prediction/truth mismatch.")
    if nodes != meta.node_order or validation_nodes != meta.node_order:
        raise ValueError(f"{run_dir}: prediction node order differs from hierarchy metadata.")
    validate_prediction_contract(run_dir, config, base, nodes, test_index, "test")
    validate_prediction_contract(
        run_dir, config, validation, validation_nodes, validation_index, "validation"
    )
    for name, values in (
        ("base", base), ("true", true),
        ("validation", validation), ("validation_true", validation_true),
    ):
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{run_dir}: {name} contains NaN or Inf.")

    started = time.perf_counter()
    bu, bu_diag = reconcile_ae.apply_reconciliation_ae(
        "bu", base, meta.sum_matrix, bottom_start_idx=meta.bottom_start_idx
    )
    td, td_diag = reconcile_ae.apply_reconciliation_ae(
        "td_fp", base, meta.sum_matrix, bottom_start_idx=meta.bottom_start_idx
    )
    covariances, covariance_diag = estimate_horizon_shrink_covariances(
        validation, validation_true
    )
    mint, mint_diag = reconcile_horizonwise_mint_shrink(
        base,
        meta.sum_matrix,
        covariances,
        bottom_start_idx=meta.bottom_start_idx,
        nnls_workers=nnls_workers,
    )
    assert_mint_nnls_success(run_dir, mint_diag)
    forecasts = {"base": base, "bu": bu, "td_fp": td, "mint_shrink": mint}
    diagnostics = {
        "postprocess_version": VERSION,
        "batch_identity": {
            "experiment_id": config.get("experiment_id"),
            "dataset": dataset,
            "run_label": config.get("run_label"),
            "source_git_commit": config.get("source_git_commit"),
            "split_protocol_version": config.get("split_protocol_version"),
        },
        "weight_estimation_source": "all_target_timestamp_validation_residuals",
        "test_targets_used_for_weights": False,
        "runtime_sec": float(time.perf_counter() - started),
        "methods": {
            "bu": {**bu_diag, **validate_output(bu, meta.sum_matrix, meta.bottom_start_idx, "bu")},
            "td_fp": {**td_diag, **validate_output(td, meta.sum_matrix, meta.bottom_start_idx, "td_fp")},
            "mint_shrink": {
                **mint_diag,
                **validate_output(mint, meta.sum_matrix, meta.bottom_start_idx, "mint_shrink"),
                "covariance_by_horizon": covariance_diag,
            },
        },
    }
    for method, path in output_paths.items():
        write_prediction(path, forecasts[method], nodes, test_index)

    rows = []
    identity = {
        "dataset": dataset,
        "experiment_id": config.get("experiment_id"),
        "model_name": config.get("model_name"),
        "graph_mode": config.get("graph_mode"),
        "seed": int(config.get("seed")),
        "output_length": 24,
        "postprocess_version": VERSION,
    }
    for method, values in forecasts.items():
        for row in phase1.metric_records(true, values, meta, method=method, variant="none"):
            rows.append({**identity, **row})
    frame = pd.DataFrame(rows)
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise FloatingPointError(f"{run_dir}: non-finite metric columns {bad}")
    frame.to_csv(metrics_path, index=False)
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "status": "completed",
        "metrics": str(metrics_path),
        "diagnostics": str(diagnostics_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("Data"))
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--nnls-workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--datasets",
        help="Comma-separated formal dataset scope; completeness is enforced within it.",
    )
    parser.add_argument("--allow-partial-matrix", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/final_postprocess_summary.json"),
    )
    args = parser.parse_args()
    dataset_scope = parse_dataset_scope(args.datasets)

    experiment_id, full_batch = matching_runs(
        args.runs_root,
        args.experiment_id,
        require_complete=args.limit is None and not args.allow_partial_matrix,
        datasets=dataset_scope,
    )
    discovered = [
        (run_dir, config)
        for run_dir, config in full_batch
        if str(config.get("model_name", "")).upper() != "LAGTCN"
        or str(config.get("graph_mode")) == "H"
    ]
    expected_count = len(dataset_scope) * len(SEEDS) * (1 + len(CURRENT_MODEL_CONFIGS))
    if (
        args.limit is None
        and not args.allow_partial_matrix
        and len(discovered) != expected_count
    ):
        raise RuntimeError(
            f"Formal main matrix {experiment_id!r} contains {len(discovered)} configs; "
            f"expected {expected_count} phase-1 reconciliation trajectories for "
            f"datasets={list(dataset_scope)}. "
            "Use --allow-partial-matrix only for diagnostics."
        )
    if args.limit is not None:
        discovered = discovered[:max(0, args.limit)]
    results = []
    for run_dir, config in discovered:
        try:
            results.append(
                process_run(run_dir, config, args.data_root, args.nnls_workers, args.force)
            )
        except Exception as exc:
            results.append({"run_dir": str(run_dir), "status": "error", "error": repr(exc)})
    payload = {
        "postprocess_version": VERSION,
        "formal_stage": FORMAL_STAGE,
        "experiment_id": experiment_id,
        "dataset_scope": list(dataset_scope),
        "batch_identity": f"{experiment_id}:{'+'.join(dataset_scope)}:{VERSION}",
        "formal_main_runs": len(full_batch),
        "reconciliation_trajectories": len(discovered),
        "reconciled_outputs": 3 * len(discovered),
        "includes_deephgnn": False,
        "completed": sum(row["status"].startswith("completed") for row in results),
        "skipped": sum(row["status"].startswith("skipped") for row in results),
        "errors": sum(row["status"] == "error" for row in results),
        "runs": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "runs"}, indent=2))
    if payload["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
