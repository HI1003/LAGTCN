"""Leakage-free, nonnegative MinT-SHR for completed Applied Energy runs.

This is an inference-only post-processing path.  It never trains or updates a
forecasting model.  For each selected run it:

1. reconstructs the historical 80/10/10 rolling-window split;
2. loads the frozen best-model checkpoint;
3. replays validation forecasts and verifies a test prefix against the saved
   base forecast;
4. estimates one Ledoit-Wolf error covariance matrix per forecast horizon from
   validation cells whose targets overlap neither training nor test targets;
5. applies nonnegative MinT-SHR to the already-saved test base forecast; and
6. writes a separate prediction, metric, covariance, and diagnostic artifact.

The test target is never passed to covariance estimation.  Existing Base, BU,
TD-FP, and MinT-OLS files are not overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from reproduction.evaluation import reconcile_forecasts as phase1
from lagtcn.reconciliation import methods as reconcile_ae
from lagtcn.core.data import LoadDatasetLoader, TARGET_TIMESTAMP_SPLIT_VERSION
from lagtcn.models.graph_models import (
    DCRNNBaseline,
    LAGTCNBaseline,
    MTGNNBaseline,
)
from lagtcn.models.temporal_baselines import (
    DLinearBaseline,
    ITransformerBaseline,
    NHiTSBaseline,
    PatchTSTBaseline,
)
from lagtcn.core.training import _align_target, _make_loader


POSTPROCESS_VERSION = "ae_mint_shrink_valscale_v3_all_validation_origins"
PREDICTION_FILENAME = "mint_recon_pred.csv"
METRICS_FILENAME = "mint_shrink_metrics.csv"
DIAGNOSTICS_FILENAME = "mint_shrink_diagnostics.json"
COVARIANCE_FILENAME = "mint_shrink_covariances.npz"

FORMAL_TEMPORAL = {"DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER"}
FORMAL_STGNN = {"DCRNN", "MTGNN"}
FORMAL_FAMILIES = FORMAL_TEMPORAL | FORMAL_STGNN | {"LAGTCN"}
DATASET_IO = {
    "GEFCom2012_2level": ("Load_GEFCom2012_hourly.csv", "node_values.npy"),
    "GEFCom2017QualifyingMatch_3level": (
        "GEFCom2017QualifyingMatchDemand.csv",
        "node_values.npy",
    ),
    "GEFCom2017FinalMatch_4level": ("load_final_filled.csv", "node_values.npy"),
}


@dataclass
class DatasetBundle:
    loader: LoadDatasetLoader
    validation: object
    test: object
    edge_index: torch.Tensor
    num_train_origins: int
    num_validation_origins: int
    num_test_origins: int


def strict_validation_slice(
    num_validation_origins: int,
    num_horizons: int,
    horizon_index: int,
) -> slice:
    """Return every validation origin for one lead.

    The target-timestamp split constructs an origin only when all ``H`` target
    timestamps lie inside the validation segment. Therefore no validation
    target can overlap train or test, and discarding ``H-1`` boundary origins
    would only waste leakage-free covariance residuals. The historical function
    name is retained for callers of the Phase-1 postprocessor.
    """
    n_val = int(num_validation_origins)
    H = int(num_horizons)
    h = int(horizon_index)
    if H < 1 or not 0 <= h < H:
        raise ValueError(f"Invalid horizon index {h} for H={H}.")
    if n_val < 2:
        raise ValueError(
            f"Need at least two validation origins for covariance estimation, got {n_val}."
        )
    return slice(0, n_val)


def estimate_horizon_shrink_covariances(
    validation_predictions: np.ndarray,
    validation_true: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Estimate leakage-free Ledoit-Wolf covariance separately for each lead."""
    predictions = np.asarray(validation_predictions, dtype=np.float64)
    true_values = np.asarray(validation_true, dtype=np.float64)
    if predictions.shape != true_values.shape or predictions.ndim != 3:
        raise ValueError(
            "Validation predictions and truth must share [origins,nodes,horizons], "
            f"got {predictions.shape} and {true_values.shape}."
        )
    n_val, num_nodes, num_horizons = predictions.shape
    covariances = np.empty(
        (num_horizons, num_nodes, num_nodes), dtype=np.float64
    )
    diagnostics: list[dict] = []
    for horizon_index in range(num_horizons):
        selection = strict_validation_slice(n_val, num_horizons, horizon_index)
        errors = (
            true_values[selection, :, horizon_index]
            - predictions[selection, :, horizon_index]
        )
        if errors.shape[0] < 2:
            raise ValueError(
                f"Need at least two validation residuals at h={horizon_index + 1}."
            )
        estimator = LedoitWolf(assume_centered=False).fit(errors)
        covariance = np.asarray(estimator.covariance_, dtype=np.float64)
        if covariance.shape != (num_nodes, num_nodes):
            raise RuntimeError(
                f"Unexpected covariance shape at h={horizon_index + 1}: "
                f"{covariance.shape}."
            )
        if not np.all(np.isfinite(covariance)):
            raise RuntimeError(
                f"Non-finite covariance at h={horizon_index + 1}."
            )
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues.min() <= 0.0:
            raise RuntimeError(
                f"Covariance is not positive definite at h={horizon_index + 1}: "
                f"min eigenvalue={eigenvalues.min()}."
            )
        covariances[horizon_index] = covariance
        diagnostics.append(
            {
                "horizon": horizon_index + 1,
                "validation_origin_start_zero_based": int(selection.start),
                "validation_origin_stop_exclusive": int(selection.stop),
                "n_residual_vectors": int(errors.shape[0]),
                "n_nodes": int(num_nodes),
                "shrinkage": float(estimator.shrinkage_),
                "error_mean_abs": float(np.abs(errors).mean()),
                "error_bias_mean_abs_across_nodes": float(
                    np.abs(errors.mean(axis=0)).mean()
                ),
                "covariance_min_eigenvalue": float(eigenvalues.min()),
                "covariance_max_eigenvalue": float(eigenvalues.max()),
                "covariance_condition_number": float(
                    eigenvalues.max() / eigenvalues.min()
                ),
            }
        )
    return covariances, diagnostics


def reconcile_horizonwise_mint_shrink(
    base_predictions: np.ndarray,
    sum_matrix: np.ndarray,
    covariances: np.ndarray,
    *,
    bottom_start_idx: int,
    nnls_workers: int,
) -> tuple[np.ndarray, dict]:
    """Apply nonnegative MinT-SHR with a frozen covariance for each horizon."""
    base = np.asarray(base_predictions, dtype=np.float64)
    weights = np.asarray(covariances, dtype=np.float64)
    if base.ndim != 3:
        raise ValueError(f"Expected Base [origins,nodes,horizons], got {base.shape}.")
    if weights.shape != (base.shape[2], base.shape[1], base.shape[1]):
        raise ValueError(
            f"Covariance shape {weights.shape} is incompatible with Base {base.shape}."
        )
    reconciled = np.empty_like(base)
    horizon_diagnostics = []
    started = time.perf_counter()
    for horizon_index in range(base.shape[2]):
        forecast, diagnostic = reconcile_ae.reconcile_mint_nnls(
            base[:, :, horizon_index : horizon_index + 1],
            sum_matrix,
            bottom_start_idx=bottom_start_idx,
            weight_mode="shrink",
            W=weights[horizon_index],
            nnls_workers=nnls_workers,
        )
        reconciled[:, :, horizon_index] = forecast[:, :, 0]
        horizon_diagnostics.append(
            {"horizon": horizon_index + 1, **diagnostic}
        )
    stats = reconcile_ae.coherence_stats(
        reconciled, sum_matrix, bottom_start_idx=bottom_start_idx
    )
    diagnostic = {
        "method": "mint_shrink",
        "method_display": "nonnegative MinT-SHR",
        "weight_estimation_source": "all_target_timestamp_validation_residuals",
        "weight_estimation_target_access": "validation_only_no_test_targets",
        "horizon_specific_covariance": True,
        "n_samples": int(base.shape[0]),
        "n_nodes": int(base.shape[1]),
        "n_horizons": int(base.shape[2]),
        "n_columns": int(base.shape[0] * base.shape[2]),
        "n_nnls_solves": int(
            sum(item["n_nnls_solves"] for item in horizon_diagnostics)
        ),
        "n_failures": int(
            sum(item["n_failures"] for item in horizon_diagnostics)
        ),
        "runtime_sec": float(time.perf_counter() - started),
        **stats,
        "horizon_diagnostics": horizon_diagnostics,
    }
    return reconciled, diagnostic


def _load_config(run_dir: Path) -> tuple[dict, Path]:
    candidates = sorted(run_dir.glob("config*.json"))
    if not candidates:
        candidates = sorted(run_dir.glob("model_info*.json"))
    if not candidates:
        raise FileNotFoundError(f"No config/model_info JSON in {run_dir}.")
    path = candidates[0]
    payload = json.loads(path.read_text())
    if isinstance(payload.get("config"), dict):
        config = dict(payload["config"])
        for key in ("model_name", "timestamp"):
            if key in payload:
                config.setdefault(key, payload[key])
    else:
        config = payload
    return config, path


def _checkpoint_path(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("best_model*.pth"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one best-model checkpoint in {run_dir}, "
            f"found {[path.name for path in candidates]}."
        )
    return candidates[0]


def _torch_state(path: Path, device: torch.device) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"{path}: checkpoint does not contain a state dict.")
    return state


def _load_checkpoint_compatibly(
    model: torch.nn.Module,
    state: dict,
) -> dict:
    """Load only checkpoints produced by the current architecture."""
    model.load_state_dict(state, strict=True)
    return {"checkpoint_layout": "current", "migration": None}


def _dataset_bundle(
    dataset_name: str,
    config: dict,
    *,
    device: torch.device,
) -> DatasetBundle:
    if dataset_name not in DATASET_IO:
        raise ValueError(f"Unsupported Applied Energy dataset {dataset_name!r}.")
    default_csv, default_values = DATASET_IO[dataset_name]
    raw_dir = PROJECT_ROOT / "Data" / dataset_name
    feature_set = str(config.get("feature_set") or "target")
    if feature_set != "target":
        raise ValueError(
            f"{dataset_name}: this formal MinT-SHR path expects feature_set='target', "
            f"got {feature_set!r}."
        )
    loader = LoadDatasetLoader(
        str(raw_dir),
        input_dim=int(config.get("input_dim", 1)),
        adj_file=str(config.get("adj_file") or "adj_hierarchy.npy"),
        value_file=str(config.get("value_file") or default_values),
        raw_csv_file=str(config.get("raw_csv_file") or default_csv),
    )
    dataset = loader.get_dataset(
        num_timesteps_in=int(config.get("num_timesteps_in", 168)),
        num_timesteps_out=int(config.get("num_timesteps_out", 24)),
    )
    (
        train,
        validation,
        test,
        split_provenance,
        _,
    ) = loader.split_dataset_by_target_timestamp(
        dataset,
        train_ratio=0.8,
        validation_ratio=0.1,
    )
    expected_split = str(config.get("split_protocol_version") or "")
    if expected_split != TARGET_TIMESTAMP_SPLIT_VERSION:
        raise ValueError(
            "Run config uses a legacy or missing split protocol: "
            f"{expected_split!r}."
        )
    if split_provenance["split_protocol_version"] != expected_split:
        raise ValueError(
            "Reconstructed split protocol differs from the frozen run config: "
            f"{split_provenance['split_protocol_version']} != {expected_split}."
        )
    edge_index = torch.as_tensor(loader.edges, dtype=torch.long, device=device)
    return DatasetBundle(
        loader=loader,
        validation=validation,
        test=test,
        edge_index=edge_index,
        num_train_origins=len(train.features),
        num_validation_origins=len(validation.features),
        num_test_origins=len(test.features),
    )


def _build_model(
    config: dict,
    bundle: DatasetBundle,
    *,
    device: torch.device,
) -> torch.nn.Module:
    loader = bundle.loader
    name = str(config["model_name"]).upper()
    common = (
        int(config["node_num"]),
        int(config["input_dim"]),
        int(config["hidden_dim"]),
        int(config["output_dim"]),
        int(config["num_layers"]),
        loader.global_min,
        loader.global_max,
    )
    if name == "DLINEAR":
        model = DLinearBaseline(
            *common, num_timesteps_in=int(config["num_timesteps_in"])
        )
    elif name == "PATCHTST":
        model = PatchTSTBaseline(
            *common, num_timesteps_in=int(config["num_timesteps_in"])
        )
    elif name == "NHITS":
        model = NHiTSBaseline(
            *common, num_timesteps_in=int(config["num_timesteps_in"])
        )
    elif name == "ITRANSFORMER":
        model = ITransformerBaseline(
            *common, num_timesteps_in=int(config["num_timesteps_in"])
        )
    elif name == "DCRNN":
        model = DCRNNBaseline(*common)
    elif name == "MTGNN":
        model = MTGNNBaseline(
            *common,
            num_timesteps_in=int(config["num_timesteps_in"]),
            top_k=min(int(config["node_num"]) - 1, int(config.get("native_top_k", 5))),
            stgnn_graph_source=str(config.get("stgnn_graph_source") or "hybrid"),
        )
    elif name == "LAGTCN":
        model = LAGTCNBaseline(
            *common,
            num_timesteps_in=int(config["num_timesteps_in"]),
            temporal_backbone=str(config.get("temporal_type") or "patchtst"),
        )
    else:
        raise ValueError(f"Unsupported formal model {name!r}.")

    model = model.to(device)
    if hasattr(model, "set_norm_params"):
        model.set_norm_params(loader.norm_params)
    model.set_graph_config(
        config,
        base_adj=loader.A,
        base_edge_index=loader.edges,
        base_edge_weight=loader.edge_weights,
    )
    if isinstance(model, LAGTCNBaseline):
        if str(config.get("graph_mode") or "H") != "H":
            raise ValueError(
                "The formal RQ3 MinT-SHR preview fixes LAGTCN to graph H."
            )
        model.set_static_graph_sources(
            hierarchy_adj=loader.A.detach().cpu().numpy(),
            similarity_adj=None,
        )
    if hasattr(model, "set_hierarchy_metadata"):
        model.set_hierarchy_metadata(
            loader.sum_matrix,
            middle_levels=(loader.hierarchy_info or {}).get("middle_levels", []),
            bottom_start_idx=loader.bottom_start_idx,
        )
    if hasattr(model, "set_st_mode"):
        model.set_st_mode(str(config.get("st_mode") or "sequential"))
    return model


def _predict(
    model: torch.nn.Module,
    signal,
    edge_index: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    loader = _make_loader(signal, batch_size=batch_size, shuffle=False)
    predictions = []
    true_values = []
    model.eval()
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, edge_index)
            y_true = _align_target(model.transform_target(y), y_pred)
            predictions.append(y_pred.detach().cpu().numpy())
            true_values.append(y_true.detach().cpu().numpy())
    if not predictions:
        raise RuntimeError("Inference produced no batches.")
    return np.concatenate(predictions), np.concatenate(true_values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(payload: dict, target: Path) -> None:
    temporary = target.with_name(f".{target.name}.mint-shrink.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
        )
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_table(frame: pd.DataFrame, target: Path, *, index: bool) -> None:
    temporary = target.with_name(f".{target.name}.mint-shrink.tmp")
    try:
        frame.to_csv(temporary, index=index)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_covariances(
    covariances: np.ndarray,
    covariance_diagnostics: list[dict],
    target: Path,
) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=".mint_shrink_covariances_",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(
                handle,
                covariances=np.asarray(covariances, dtype=np.float64),
                shrinkage=np.asarray(
                    [row["shrinkage"] for row in covariance_diagnostics],
                    dtype=np.float64,
                ),
                n_residual_vectors=np.asarray(
                    [
                        row["n_residual_vectors"]
                        for row in covariance_diagnostics
                    ],
                    dtype=np.int64,
                ),
                postprocess_version=np.asarray(POSTPROCESS_VERSION),
                weight_estimation_source=np.asarray(
                    "all_target_timestamp_validation_residuals"
                ),
            )
            os.replace(temporary, target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise


def _metric_frame(
    *,
    run_id: str,
    dataset_name: str,
    config: dict,
    true_values: np.ndarray,
    reconciled: np.ndarray,
    meta: phase1.DatasetMeta,
) -> pd.DataFrame:
    records = []
    for boundary_variant in phase1.BOUNDARY_VARIANTS:
        records.extend(
            phase1.metric_records(
                true_values,
                reconciled,
                meta,
                method="mint_shrink",
                variant=boundary_variant,
            )
        )
    frame = pd.DataFrame(records)
    provenance = {
        "run_id": run_id,
        "dataset": dataset_name,
        "model_name": config.get("model_name"),
        "seed": config.get("seed"),
        "num_timesteps_out": config.get("num_timesteps_out"),
        "graph_mode": config.get("graph_mode"),
        "gnn_type": config.get("gnn_type"),
        "temporal_type": config.get("temporal_type"),
        "st_mode": config.get("st_mode"),
        "stgnn_graph_source": config.get("stgnn_graph_source"),
        "prediction_file": PREDICTION_FILENAME,
        "source_prediction_file": "base_pred.csv",
        "prediction_role": "reconciled",
        "reconciliation_method": "mint_shrink",
        "weight_estimation_source": "all_target_timestamp_validation_residuals",
        "postprocess_version": POSTPROCESS_VERSION,
        "result_tag": "submission_validation_weighted",
    }
    for key, value in reversed(list(provenance.items())):
        frame.insert(0, key, value)
    return frame


def _formal_manifest_rows(
    manifest_path: Path,
    *,
    datasets: set[str] | None,
    models: set[str] | None,
    seeds: set[int] | None,
    limit: int | None,
) -> list[dict]:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        model_name = str(row.get("configuration") or "").upper()
        family = str(row.get("family") or "")
        is_formal = (
            (family == "temporal" and model_name in FORMAL_TEMPORAL)
            or (family == "stgnn" and model_name in FORMAL_STGNN)
            or (family == "lagtcn" and model_name == "H")
        )
        if not is_formal or str(row.get("horizon")) != "24":
            continue
        if str(row.get("analysis_ready")).lower() != "true":
            continue
        if datasets is not None and row["dataset"] not in datasets:
            continue
        config_model = (
            "LAGTCN" if family == "lagtcn" else model_name
        )
        if models is not None and config_model not in models:
            continue
        if seeds is not None and int(row["seed"]) not in seeds:
            continue
        selected.append(row)
    selected.sort(
        key=lambda row: (
            row["dataset"],
            row["family"],
            row["configuration"],
            int(row["seed"]),
        )
    )
    return selected[:limit] if limit is not None else selected


def _parse_csv_set(value: str | None, transform=str) -> set | None:
    if value is None:
        return None
    return {transform(item.strip()) for item in value.split(",") if item.strip()}


def process_run(
    row: dict,
    bundle: DatasetBundle,
    *,
    device: torch.device,
    batch_size: int,
    nnls_workers: int,
    replay_batches: int,
    force: bool,
) -> dict:
    dataset_name = row["dataset"]
    run_dir = PROJECT_ROOT / row["run_dir"]
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    config, config_path = _load_config(run_dir)
    checkpoint = _checkpoint_path(run_dir)
    base_path = run_dir / "base_pred.csv"
    true_path = phase1._find_artifact(
        run_dir, phase1.TRUE_CANDIDATES, phase1.LEGACY_TRUE_PATTERNS
    )
    if not base_path.is_file() or true_path is None:
        raise FileNotFoundError(f"{run_dir}: missing Base or true prediction CSV.")
    output_paths = {
        "prediction": run_dir / PREDICTION_FILENAME,
        "metrics": run_dir / METRICS_FILENAME,
        "diagnostics": run_dir / DIAGNOSTICS_FILENAME,
        "covariances": run_dir / COVARIANCE_FILENAME,
    }
    if not force and all(path.is_file() for path in output_paths.values()):
        return {
            "run_id": str(row["run_id"]),
            "dataset": dataset_name,
            "model_name": config.get("model_name"),
            "seed": int(config.get("seed")),
            "status": "skipped_existing",
            **{f"{key}_file": str(path.relative_to(PROJECT_ROOT)) for key, path in output_paths.items()},
        }

    model = _build_model(config, bundle, device=device)
    checkpoint_compatibility = _load_checkpoint_compatibly(
        model, _torch_state(checkpoint, device)
    )

    saved_base, nodes, time_index = phase1.parse_prediction_csv(base_path)
    saved_true, true_nodes, true_time_index = phase1.parse_prediction_csv(true_path)
    if saved_base.shape != saved_true.shape:
        raise ValueError(
            f"{run_dir}: Base/true shapes differ: {saved_base.shape}/{saved_true.shape}."
        )
    if nodes != true_nodes or not time_index.equals(true_time_index):
        raise ValueError(f"{run_dir}: Base/true metadata differ.")
    if saved_base.shape[0] != bundle.num_test_origins:
        raise ValueError(
            f"{run_dir}: saved test origins={saved_base.shape[0]} but "
            f"reconstructed split has {bundle.num_test_origins}."
        )

    replay_predictions, replay_true = _predict(
        model,
        bundle.test,
        bundle.edge_index,
        device=device,
        batch_size=batch_size,
        max_batches=replay_batches,
    )
    replay_count = replay_predictions.shape[0]
    prediction_delta = np.abs(replay_predictions - saved_base[:replay_count])
    true_delta = np.abs(replay_true - saved_true[:replay_count])
    is_mtgnn_native = (
        str(config.get("model_name") or "").upper() == "MTGNN"
        and str(config.get("stgnn_graph_source") or "").lower() == "native"
    )
    replay_rtol = 5e-3 if is_mtgnn_native else 1e-5
    replay_atol = 10.0 if is_mtgnn_native else 1e-2
    replay_tolerance_policy = (
        "mtgnn_native_cross_hardware_numerical_equivalence"
        if is_mtgnn_native
        else "default_checkpoint_replay"
    )
    replay_prediction_ok = bool(
        np.allclose(
            replay_predictions,
            saved_base[:replay_count],
            rtol=replay_rtol,
            atol=replay_atol,
        )
    )
    replay_true_ok = bool(
        np.allclose(
            replay_true,
            saved_true[:replay_count],
            rtol=1e-5,
            atol=1e-2,
        )
    )
    if not replay_prediction_ok or not replay_true_ok:
        raise RuntimeError(
            f"{run_dir}: checkpoint replay mismatch; "
            f"prediction max abs={prediction_delta.max()}, "
            f"true max abs={true_delta.max()}."
        )

    validation_predictions, validation_true = _predict(
        model,
        bundle.validation,
        bundle.edge_index,
        device=device,
        batch_size=batch_size,
    )
    covariances, covariance_diagnostics = (
        estimate_horizon_shrink_covariances(
            validation_predictions, validation_true
        )
    )
    reconciled, reconciliation_diagnostic = (
        reconcile_horizonwise_mint_shrink(
            saved_base,
            bundle.loader.sum_matrix,
            covariances,
            bottom_start_idx=int(bundle.loader.bottom_start_idx),
            nnls_workers=nnls_workers,
        )
    )
    if reconciliation_diagnostic["n_failures"]:
        raise RuntimeError(
            f"{run_dir}: MinT-SHR had "
            f"{reconciliation_diagnostic['n_failures']} NNLS failures."
        )
    if reconciliation_diagnostic["min_value"] < -1e-10:
        raise RuntimeError(f"{run_dir}: MinT-SHR violates nonnegativity.")
    if reconciliation_diagnostic["coherence_residual_max_abs"] > 1e-8:
        raise RuntimeError(f"{run_dir}: MinT-SHR violates coherence.")

    meta = phase1.load_dataset_meta(PROJECT_ROOT / "Data" / dataset_name)
    if nodes != meta.node_order:
        raise ValueError(
            f"{run_dir}: saved node order differs from dataset metadata."
        )
    base_header = pd.read_csv(base_path, index_col=0, nrows=1)
    prediction_frame = pd.DataFrame(
        reconciled.transpose(0, 2, 1).reshape(
            reconciled.shape[0], reconciled.shape[2] * reconciled.shape[1]
        ),
        index=pd.Index(time_index, name=base_header.index.name),
        columns=list(base_header.columns),
    )
    metrics = _metric_frame(
        run_id=str(row["run_id"]),
        dataset_name=dataset_name,
        config=config,
        true_values=saved_true,
        reconciled=reconciled,
        meta=meta,
    )
    diagnostics = {
        "postprocess_version": POSTPROCESS_VERSION,
        "run_id": str(row["run_id"]),
        "dataset": dataset_name,
        "model_name": config.get("model_name"),
        "seed": int(config.get("seed")),
        "training_performed": False,
        "checkpoint_file": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_compatibility": checkpoint_compatibility,
        "config_file": str(config_path.relative_to(PROJECT_ROOT)),
        "base_prediction_file": str(base_path.relative_to(PROJECT_ROOT)),
        "base_prediction_sha256": _sha256(base_path),
        "split_protocol": {
            "window_origin_split": "raw-timestamp 80/10/10; all 24 targets remain within one segment",
            "num_train_origins": bundle.num_train_origins,
            "num_validation_origins": bundle.num_validation_origins,
            "num_test_origins": bundle.num_test_origins,
            "covariance_boundary_policy": (
                "all validation origins: target-timestamp split already prevents "
                "train/test target overlap"
            ),
        },
        "checkpoint_replay": {
            "n_origins_checked": int(replay_count),
            "prediction_allclose_rtol": replay_rtol,
            "prediction_allclose_atol": replay_atol,
            "prediction_tolerance_policy": replay_tolerance_policy,
            "prediction_allclose": replay_prediction_ok,
            "prediction_max_abs_difference": float(prediction_delta.max()),
            "prediction_mean_abs_difference": float(prediction_delta.mean()),
            "prediction_max_relative_difference": float(
                (
                    prediction_delta
                    / np.maximum(np.abs(saved_base[:replay_count]), 1.0)
                ).max()
            ),
            "true_allclose": replay_true_ok,
            "true_max_abs_difference": float(true_delta.max()),
        },
        "covariance_estimation": {
            "estimator": "sklearn.covariance.LedoitWolf",
            "assume_centered": False,
            "one_covariance_per_horizon": True,
            "test_targets_accessed": False,
            "horizons": covariance_diagnostics,
        },
        "reconciliation": reconciliation_diagnostic,
        "outputs": {
            key: str(path.relative_to(run_dir))
            for key, path in output_paths.items()
        },
    }

    _atomic_covariances(
        covariances, covariance_diagnostics, output_paths["covariances"]
    )
    _atomic_table(prediction_frame, output_paths["prediction"], index=True)
    _atomic_table(metrics, output_paths["metrics"], index=False)
    _atomic_json(diagnostics, output_paths["diagnostics"])
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "run_id": str(row["run_id"]),
        "dataset": dataset_name,
        "family": row["family"],
        "configuration": row["configuration"],
        "model_name": config.get("model_name"),
        "seed": int(config.get("seed")),
        "status": "ok",
        "replay_max_relative_difference": diagnostics["checkpoint_replay"][
            "prediction_max_relative_difference"
        ],
        "covariance_mean_shrinkage": float(
            np.mean([item["shrinkage"] for item in covariance_diagnostics])
        ),
        "covariance_max_condition_number": float(
            max(
                item["covariance_condition_number"]
                for item in covariance_diagnostics
            )
        ),
        "nnls_fraction": float(
            reconciliation_diagnostic["n_nnls_solves"]
            / reconciliation_diagnostic["n_columns"]
        ),
        "n_failures": reconciliation_diagnostic["n_failures"],
        "min_prediction": reconciliation_diagnostic["min_value"],
        "coherence_residual_max_abs": reconciliation_diagnostic[
            "coherence_residual_max_abs"
        ],
        "runtime_sec": reconciliation_diagnostic["runtime_sec"],
        **{
            f"{key}_file": str(path.relative_to(PROJECT_ROOT))
            for key, path in output_paths.items()
        },
    }


def write_preview_summary(records: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(records)
    _atomic_table(manifest, output_dir / "manifest.csv", index=False)
    metric_frames = []
    for record in records:
        if record.get("status") not in {"ok", "skipped_existing"}:
            continue
        metric_path = PROJECT_ROOT / record["metrics_file"]
        if metric_path.is_file():
            metric_frames.append(pd.read_csv(metric_path))
    if not metric_frames:
        return
    metrics = pd.concat(metric_frames, ignore_index=True)
    primary = metrics[
        (metrics["boundary_variant"] == "none")
        & (metrics["level"] == "All")
        & (metrics["horizon_label"] == "all")
    ].copy()
    _atomic_table(
        primary, output_dir / "mint_shrink_primary_run_metrics.csv", index=False
    )
    existing_rows = []
    for record in records:
        if record.get("status") not in {"ok", "skipped_existing"}:
            continue
        if "metrics_file" not in record:
            continue
        run_dir = (PROJECT_ROOT / record["metrics_file"]).parent
        existing_path = run_dir / "reconciliation_metrics.csv"
        if not existing_path.is_file():
            continue
        existing = pd.read_csv(existing_path, low_memory=False)
        existing = existing[
            (existing["boundary_variant"] == "none")
            & (existing["level"] == "All")
            & (existing["horizon_label"] == "all")
            & (existing["method"].isin(["base", "bu", "td_fp", "mint_ols"]))
        ].copy()
        existing_rows.append(existing)
    combined = pd.concat(existing_rows + [primary], ignore_index=True)
    combined["metric_source"] = np.where(
        combined["method"] == "mint_shrink",
        POSTPROCESS_VERSION,
        "existing_phase1",
    )
    _atomic_table(
        combined, output_dir / "all_methods_primary_run_metrics.csv", index=False
    )
    gain_rows = []
    for run_id, group in combined.groupby("run_id"):
        base = group[group["method"] == "base"]
        if len(base) != 1:
            continue
        base_row = base.iloc[0]
        for method in ("bu", "td_fp", "mint_ols", "mint_shrink"):
            reconciled = group[group["method"] == method]
            if len(reconciled) != 1:
                continue
            rec_row = reconciled.iloc[0]
            for metric in ("wape", "mase", "rmse"):
                base_value = float(base_row[metric])
                reconciled_value = float(rec_row[metric])
                gain_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": base_row["dataset"],
                        "model_name": base_row["model_name"],
                        "seed": int(base_row["seed"]),
                        "method": method,
                        "metric": metric,
                        "base_value": base_value,
                        "reconciled_value": reconciled_value,
                        "gain_pct": (
                            100.0
                            * (base_value - reconciled_value)
                            / base_value
                        ),
                    }
                )
    gains = pd.DataFrame(gain_rows)
    _atomic_table(gains, output_dir / "paired_gains.csv", index=False)
    summary = (
        gains.groupby(["method", "metric"], as_index=False)
        .agg(
            n=("gain_pct", "count"),
            mean_gain_pct=("gain_pct", "mean"),
            median_gain_pct=("gain_pct", "median"),
            sd_gain_pct=("gain_pct", "std"),
            improved_count=("gain_pct", lambda values: int((values > 0).sum())),
            improved_fraction=(
                "gain_pct",
                lambda values: float((values > 0).mean()),
            ),
        )
    )
    _atomic_table(summary, output_dir / "method_summary.csv", index=False)
    dataset_summary = (
        gains.groupby(["dataset", "method", "metric"], as_index=False)
        .agg(
            n=("gain_pct", "count"),
            mean_gain_pct=("gain_pct", "mean"),
            median_gain_pct=("gain_pct", "median"),
            improved_count=("gain_pct", lambda values: int((values > 0).sum())),
            improved_fraction=(
                "gain_pct",
                lambda values: float((values > 0).mean()),
            ),
        )
    )
    _atomic_table(
        dataset_summary, output_dir / "dataset_method_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT
        / "results/source_data"
        / "formal_run_manifest.csv",
    )
    parser.add_argument("--datasets", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--nnls-workers", type=int, default=4)
    parser.add_argument("--replay-batches", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=PROJECT_ROOT
        / "results/mint_shrink_preview",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.batch_size < 1 or args.nnls_workers < 1 or args.replay_batches < 1:
        raise ValueError("Batch size, NNLS workers, and replay batches must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {device}.")

    datasets = _parse_csv_set(args.datasets)
    models = _parse_csv_set(args.models, lambda value: value.upper())
    seeds = _parse_csv_set(args.seeds, int)
    rows = _formal_manifest_rows(
        args.manifest,
        datasets=datasets,
        models=models,
        seeds=seeds,
        limit=args.limit,
    )
    if not rows:
        raise RuntimeError("No formal runs matched the requested filters.")

    bundle_cache: dict[str, DatasetBundle] = {}
    records = []
    for index, row in enumerate(rows, start=1):
        dataset_name = row["dataset"]
        run_dir = PROJECT_ROOT / row["run_dir"]
        config, _ = _load_config(run_dir)
        if dataset_name not in bundle_cache:
            bundle_cache[dataset_name] = _dataset_bundle(
                dataset_name, config, device=device
            )
        print(
            f"[{index}/{len(rows)}] {row['run_id']} "
            f"on {device}",
            flush=True,
        )
        try:
            record = process_run(
                row,
                bundle_cache[dataset_name],
                device=device,
                batch_size=args.batch_size,
                nnls_workers=args.nnls_workers,
                replay_batches=args.replay_batches,
                force=args.force,
            )
        except Exception as exc:
            record = {
                "run_id": row["run_id"],
                "dataset": dataset_name,
                "family": row["family"],
                "configuration": row["configuration"],
                "seed": int(row["seed"]),
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(
                f"ERROR {row['run_id']}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        records.append(record)
        write_preview_summary(records, args.preview_dir)
    errors = [record for record in records if record["status"] == "error"]
    print(
        json.dumps(
            {
                "postprocess_version": POSTPROCESS_VERSION,
                "selected_runs": len(rows),
                "completed_or_existing": len(rows) - len(errors),
                "errors": len(errors),
                "preview_dir": str(args.preview_dir),
            },
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
