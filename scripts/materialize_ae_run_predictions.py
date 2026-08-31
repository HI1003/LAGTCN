"""Materialize Applied Energy base and reconciled forecasts inside each run.

For every completed run referenced by an existing Phase 1 post-processing
manifest, this script produces the stable files:

    base_pred.csv
    bu_recon_pred.csv
    td_recon_pred.csv
    mint_recon_pred.csv
    reconciliation_metrics.csv
    reconciliation_diagnostics.json

The base artifact is renamed (not copied) to base_pred.csv. Reconciled arrays
are read from the validated central NPZ archive and written in the same wide
CSV layout as the base forecast. Writes are atomic and existing completed
targets are skipped, so interrupted executions can be resumed safely.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import postprocess_ae_phase1 as phase1

MATERIALIZE_VERSION = "ae_materialized_run_predictions_v2_smase24"

METHOD_FILES = {
    "base": ("base", "none", "base_pred.csv"),
    "bu": ("reconciled", "bu", "bu_recon_pred.csv"),
    "td_fp": ("reconciled", "td_fp", "td_recon_pred.csv"),
    "mint_ols": ("reconciled", "mint_ols", "mint_recon_pred.csv"),
}
ARCHIVE_KEYS = {
    "bu": "bu_recon_pred.csv",
    "td_fp": "td_recon_pred.csv",
    "mint_ols": "mint_recon_pred.csv",
}


def _atomic_csv(frame: pd.DataFrame, target: Path) -> None:
    temp = target.with_name(f".{target.name}.materialize.tmp")
    try:
        frame.to_csv(temp)
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _atomic_table(frame: pd.DataFrame, target: Path) -> None:
    temp = target.with_name(f".{target.name}.materialize.tmp")
    try:
        frame.to_csv(temp, index=False)
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _atomic_json(payload: dict, target: Path) -> None:
    temp = target.with_name(f".{target.name}.materialize.tmp")
    try:
        temp.write_text(json.dumps(payload, indent=2, allow_nan=False))
        os.replace(temp, target)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _prediction_columns(node_order: list[str], num_horizons: int) -> list[str]:
    if num_horizons == 1:
        return node_order
    return [
        f"{node}_t+{horizon}"
        for horizon in range(1, num_horizons + 1)
        for node in node_order
    ]


def _prediction_frame(
    values: np.ndarray,
    *,
    time_index: np.ndarray,
    node_order: list[str],
    index_name: str | None,
) -> pd.DataFrame:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"Expected [samples,nodes,horizons], got {array.shape}.")
    samples, nodes, horizons = array.shape
    if nodes != len(node_order) or samples != len(time_index):
        raise ValueError(
            f"Archive metadata mismatch: array={array.shape}, "
            f"time={len(time_index)}, nodes={len(node_order)}."
        )
    if horizons == 1:
        matrix = array[:, :, 0]
    else:
        matrix = array.transpose(0, 2, 1).reshape(samples, horizons * nodes)
    frame = pd.DataFrame(
        matrix,
        index=pd.Index(time_index, name=index_name),
        columns=_prediction_columns(node_order, horizons),
    )
    return frame


def _find_base_source(run_dir: Path) -> Path:
    source = phase1._find_artifact(
        run_dir,
        phase1.PREDICTION_CANDIDATES,
        phase1.LEGACY_PREDICTION_PATTERNS,
    )
    if source is None:
        raise FileNotFoundError(f"No base prediction artifact in {run_dir}.")
    return source


def _materialize_one(task: tuple[str, str, str, str]) -> dict:
    run_id, run_dir_text, archive_text, original_source_name = task
    run_dir = Path(run_dir_text)
    archive_path = Path(archive_text)
    base_target = run_dir / "base_pred.csv"
    source = _find_base_source(run_dir)
    base_renamed = source != base_target

    base_header = pd.read_csv(source, index_col=0, nrows=1)
    index_name = base_header.index.name
    base_columns = [str(column) for column in base_header.columns]

    written = []
    skipped = []
    with np.load(archive_path, allow_pickle=False) as archive:
        archive_run_id = str(archive["source_run_id"].item())
        if archive_run_id != run_id:
            raise RuntimeError(
                f"{run_id}: archive source_run_id is {archive_run_id!r}."
            )
        missing_keys = sorted(set(ARCHIVE_KEYS) - set(archive.files))
        if missing_keys:
            raise RuntimeError(
                f"{run_id}: archive is missing methods {missing_keys}."
            )
        time_index = np.asarray(archive["time_index"], dtype=str)
        node_order = [str(value) for value in archive["node_order"]]
        if not node_order or len(base_columns) % len(node_order):
            raise RuntimeError(
                f"{run_id}: base columns cannot be aligned with archive nodes."
            )
        num_horizons = len(base_columns) // len(node_order)
        expected_columns = _prediction_columns(node_order, num_horizons)
        if base_columns != expected_columns:
            raise RuntimeError(
                f"{run_id}: base column order differs from archive node order."
            )

        if base_renamed:
            if base_target.exists():
                raise RuntimeError(
                    f"{run_id}: both {base_target.name} and {source.name} exist."
                )
            os.replace(source, base_target)

        for archive_key, filename in ARCHIVE_KEYS.items():
            target = run_dir / filename
            if target.exists():
                skipped.append(filename)
                continue
            values = archive[archive_key]
            frame = _prediction_frame(
                values,
                time_index=time_index,
                node_order=node_order,
                index_name=index_name,
            )
            if [str(column) for column in frame.columns] != base_columns:
                raise RuntimeError(
                    f"{run_id}: {filename} columns differ from base_pred.csv."
                )
            _atomic_csv(frame, target)
            written.append(filename)

    return {
        "run_id": run_id,
        "source_prediction_file": original_source_name,
        "base_renamed": base_renamed,
        "written": written,
        "skipped": skipped,
    }


def _enrich_metric_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "source_prediction_file" not in result:
        result["source_prediction_file"] = result["prediction_file"]
    roles = result["method"].map(
        {method: values[0] for method, values in METHOD_FILES.items()}
    )
    reconciliations = result["method"].map(
        {method: values[1] for method, values in METHOD_FILES.items()}
    )
    files = result["method"].map(
        {method: values[2] for method, values in METHOD_FILES.items()}
    )
    if roles.isna().any() or reconciliations.isna().any() or files.isna().any():
        unknown = sorted(result.loc[files.isna(), "method"].astype(str).unique())
        raise RuntimeError(f"Unknown methods in phase1_metrics_long.csv: {unknown}")
    result["prediction_role"] = roles
    result["reconciliation_method"] = reconciliations
    result["prediction_file"] = files
    return result


def _enrich_base_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "source_prediction_file" not in result:
        result["source_prediction_file"] = result["prediction_file"]
    result["prediction_role"] = "base"
    result["reconciliation_method"] = "none"
    result["prediction_file"] = "base_pred.csv"
    return result


def materialize_dataset(
    dataset: str,
    *,
    data_root: Path,
    workers: int,
    limit: int | None = None,
) -> dict:
    ae_root = data_root / dataset / "output" / "ae"
    output_dir = ae_root / "_postprocess_ae_phase1"
    archive_manifest_path = output_dir / "reconciled_predictions_manifest.csv"
    archive_manifest = pd.read_csv(archive_manifest_path, low_memory=False)
    if limit is not None:
        archive_manifest = archive_manifest.iloc[:limit].copy()

    tasks = []
    for row in archive_manifest.itertuples(index=False):
        run_dir = ae_root / row.run_id
        archive_path = output_dir / row.archive_file
        if not run_dir.is_dir() or not archive_path.is_file():
            raise FileNotFoundError(
                f"Missing run/archive for {row.run_id}: {run_dir}, {archive_path}"
            )
        tasks.append((
            row.run_id,
            str(run_dir),
            str(archive_path),
            str(row.source_prediction_file),
        ))

    effective_workers = max(1, min(int(workers), len(tasks)))
    if effective_workers == 1:
        outcomes = [_materialize_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            outcomes = list(pool.map(_materialize_one, tasks, chunksize=1))
    outcome_by_run = {item["run_id"]: item for item in outcomes}

    metrics_path = output_dir / "phase1_metrics_long.csv"
    metrics = _enrich_metric_table(pd.read_csv(metrics_path, low_memory=False))
    negative_path = output_dir / "negative_prediction_rates.csv"
    negative = _enrich_base_table(pd.read_csv(negative_path, low_memory=False))
    clamp_path = output_dir / "clamp_trigger_rates.csv"
    clamp = _enrich_base_table(pd.read_csv(clamp_path, low_memory=False))

    selected_run_ids = set(archive_manifest["run_id"])
    metrics_selected = metrics[metrics["run_id"].isin(selected_run_ids)].copy()
    diagnostics_all = json.loads(
        (output_dir / "reconciliation_diagnostics.json").read_text()
    )

    file_manifest_rows = []
    grouped_metrics = {
        run_id: frame.copy()
        for run_id, frame in metrics_selected.groupby("run_id", sort=False)
    }
    for row in archive_manifest.to_dict(orient="records"):
        run_id = row["run_id"]
        run_dir = ae_root / run_id
        source_name = outcome_by_run[run_id]["source_prediction_file"]
        local_metrics = grouped_metrics[run_id]
        _atomic_table(local_metrics, run_dir / "reconciliation_metrics.csv")
        diagnostic_payload = {
            "materialize_version": MATERIALIZE_VERSION,
            "postprocess_version": phase1.POSTPROCESS_VERSION,
            "run_id": run_id,
            "prediction_files": {
                method: {
                    "prediction_role": role,
                    "reconciliation_method": reconciliation,
                    "prediction_file": filename,
                }
                for method, (role, reconciliation, filename) in METHOD_FILES.items()
            },
            "diagnostics": diagnostics_all[run_id],
        }
        _atomic_json(
            diagnostic_payload,
            run_dir / "reconciliation_diagnostics.json",
        )

        common = {
            key: value
            for key, value in row.items()
            if key not in {"archive_file", "methods", "shape", "dtype", "size_bytes"}
        }
        for method, (role, reconciliation, filename) in METHOD_FILES.items():
            file_manifest_rows.append({
                **common,
                "prediction_role": role,
                "reconciliation_method": reconciliation,
                "prediction_file": filename,
                "relative_file": str(Path(run_id) / filename),
                "source_prediction_file": source_name,
                "central_archive_file": row["archive_file"] if method != "base" else None,
                "materialize_version": MATERIALIZE_VERSION,
            })

    if limit is None:
        _atomic_table(metrics, metrics_path)
        _atomic_table(negative, negative_path)
        _atomic_table(clamp, clamp_path)
        file_manifest = pd.DataFrame(file_manifest_rows)
        _atomic_table(
            file_manifest,
            output_dir / "prediction_files_manifest.csv",
        )
        manifest_path = output_dir / "postprocess_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["materialized_run_predictions"] = {
            "materialize_version": MATERIALIZE_VERSION,
            "materialized_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "n_runs": len(tasks),
            "n_prediction_files": len(tasks) * len(METHOD_FILES),
            "filenames": {
                method: filename
                for method, (_, _, filename) in METHOD_FILES.items()
            },
            "per_run_metrics_file": "reconciliation_metrics.csv",
            "per_run_diagnostics_file": "reconciliation_diagnostics.json",
        }
        _atomic_json(manifest, manifest_path)

    return {
        "dataset": dataset,
        "n_runs": len(tasks),
        "workers": effective_workers,
        "n_written": sum(len(item["written"]) for item in outcomes),
        "n_skipped": sum(len(item["skipped"]) for item in outcomes),
        "n_base_renamed": sum(item["base_renamed"] for item in outcomes),
        "limit": limit,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "Data"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    summary = materialize_dataset(
        args.dataset,
        data_root=Path(args.data_root).resolve(),
        workers=args.workers,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
