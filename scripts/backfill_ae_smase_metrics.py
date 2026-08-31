#!/usr/bin/env python3
"""Backfill formal sMASE metrics from saved AE prediction trajectories.

This utility never loads a checkpoint or retrains a model. It replaces legacy
MASE values in completed formal runs with the frozen training-period lag-24
sMASE protocol and synchronizes config.json, metrics.json,
validation_metrics.json, level_metrics.csv, and model_info.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT / "code", ROOT / "scripts"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

from ae_protocol import is_formal_ae_stage
import mase
from metrics import calculate_level_metrics
import postprocess_ae_phase1 as phase1
from train_eval import _compute_metrics


VERSION = "ae_smase24_saved_prediction_backfill_v1"


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_pair(run_dir: Path, pred_name: str, true_name: str, node_order: list[str]):
    prediction, pred_nodes, pred_index = phase1.parse_prediction_csv(run_dir / pred_name)
    truth, true_nodes, true_index = phase1.parse_prediction_csv(run_dir / true_name)
    if prediction.shape != truth.shape:
        raise ValueError(f"{run_dir}: {pred_name}/{true_name} shape mismatch.")
    if pred_nodes != true_nodes or pred_nodes != node_order:
        raise ValueError(f"{run_dir}: {pred_name}/{true_name} node order mismatch.")
    if not pred_index.equals(true_index):
        raise ValueError(f"{run_dir}: {pred_name}/{true_name} index mismatch.")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise FloatingPointError(f"{run_dir}: {pred_name}/{true_name} contains NaN or Inf.")
    return prediction, truth


def _merged_metrics(path: Path, fresh: dict) -> tuple[dict, str | None]:
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    prior_version = existing.get("MASE_version")
    existing.update(fresh)
    return existing, prior_version


def _already_current(run_dir: Path, config: dict) -> bool:
    metadata = config.get("smase_scale_metadata") or {}
    if metadata.get("mase_version") != mase.MASE_VERSION:
        return False
    for name in ("metrics.json", "validation_metrics.json"):
        path = run_dir / name
        if not path.is_file():
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("MASE_version") != mase.MASE_VERSION:
            return False
    level_path = run_dir / "level_metrics.csv"
    if not level_path.is_file():
        return False
    level_versions = set(pd.read_csv(level_path, usecols=["MASE_version"])["MASE_version"])
    if level_versions != {mase.MASE_VERSION}:
        return False
    model_info_path = run_dir / "model_info.json"
    if model_info_path.is_file():
        model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
        model_versions = {
            (model_info.get("metrics") or {}).get("MASE_version"),
            ((model_info.get("config") or {}).get("smase_scale_metadata") or {}).get(
                "mase_version"
            ),
        }
        model_versions.update(
            row.get("MASE_version") for row in (model_info.get("level_metrics") or [])
        )
        if model_versions != {mase.MASE_VERSION}:
            return False
    return True


def repair_run(run_dir: Path, data_root: Path, *, apply: bool) -> dict:
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stage = config.get("experiment_stage")
    if not is_formal_ae_stage(stage):
        raise ValueError(f"{run_dir}: stage {stage!r} is not a formal AE stage.")

    if _already_current(run_dir, config):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        return {
            "run_dir": str(run_dir),
            "dataset": str(config.get("dataset")),
            "stage": stage,
            "prior_mase_version": mase.MASE_VERSION,
            "new_mase_version": mase.MASE_VERSION,
            "test_smase": metrics["MASE"],
            "status": "already_current",
        }

    dataset = str(config.get("dataset") or Path(str(config.get("raw_data_dir", ""))).name)
    meta = phase1.load_dataset_meta(data_root / dataset)
    config_nodes = config.get("node_order") or config.get("node_names") or []
    if [str(value) for value in config_nodes] != meta.node_order:
        raise ValueError(f"{run_dir}: config node order differs from dataset metadata.")

    scale = np.asarray(meta.naive_scale, dtype=np.float64)
    scale_metadata = mase.naive_scale_metadata(
        scale,
        train_length=meta.train_length,
        node_names=meta.node_order,
        seasonal_period=mase.MASE_SEASONAL_PERIOD,
    )
    base, test_truth = _read_pair(run_dir, "base_pred.csv", "true.csv", meta.node_order)
    validation, validation_truth = _read_pair(
        run_dir, "validation_pred.csv", "validation_true.csv", meta.node_order
    )
    test_fresh = _compute_metrics(
        test_truth,
        base,
        num_timesteps_in=int(config.get("num_timesteps_in", 168)),
        mase_scale=scale,
        require_mase_scale=True,
    )
    validation_fresh = _compute_metrics(
        validation_truth,
        validation,
        num_timesteps_in=int(config.get("num_timesteps_in", 168)),
        mase_scale=scale,
        require_mase_scale=True,
    )
    test_metrics, prior_test_version = _merged_metrics(run_dir / "metrics.json", test_fresh)
    validation_metrics, prior_validation_version = _merged_metrics(
        run_dir / "validation_metrics.json", validation_fresh
    )

    provenance = {
        "version": VERSION,
        "source": "saved_base_and_true_predictions",
        "retraining": False,
        "prior_test_mase_version": prior_test_version,
        "prior_validation_mase_version": prior_validation_version,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    config["smase_scale_metadata"] = scale_metadata
    config["smase_metrics_backfill"] = provenance

    level_config = dict(config)
    level_config["_mase_scale"] = scale
    level_config["output_dir"] = str(run_dir)
    if apply:
        _atomic_json(config_path, config)
        _atomic_json(run_dir / "metrics.json", test_metrics)
        _atomic_json(run_dir / "validation_metrics.json", validation_metrics)
        level_metrics = calculate_level_metrics(base, test_truth, level_config)

        model_info_path = run_dir / "model_info.json"
        if model_info_path.is_file():
            model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
            model_config = dict(model_info.get("config") or {})
            model_config.update(config)
            model_config["_mase_scale"] = scale.tolist()
            model_info["config"] = model_config
            model_info["metrics"] = test_metrics
            model_info["level_metrics"] = level_metrics.to_dict(orient="records")
            _atomic_json(model_info_path, model_info)

    return {
        "run_dir": str(run_dir),
        "dataset": dataset,
        "stage": stage,
        "prior_mase_version": prior_test_version,
        "new_mase_version": test_metrics["MASE_version"],
        "test_smase": test_metrics["MASE"],
        "status": "repaired" if apply else "would_repair",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", action="append", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "Data")
    parser.add_argument("--experiment-id")
    parser.add_argument("--apply", action="store_true", help="Write changes; default is audit-only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = sorted({path for root in args.runs_root for path in root.rglob("config.json")})
    selected = []
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        if args.experiment_id and config.get("experiment_id") != args.experiment_id:
            continue
        if is_formal_ae_stage(config.get("experiment_stage")):
            selected.append(path.parent)
    if not selected:
        raise RuntimeError("No matching formal Applied Energy runs found.")

    rows = [repair_run(run_dir, args.data_root, apply=args.apply) for run_dir in selected]
    print(json.dumps({
        "version": VERSION,
        "apply": bool(args.apply),
        "num_runs": len(rows),
        "runs": rows,
    }, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
