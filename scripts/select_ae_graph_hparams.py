#!/usr/bin/env python3
"""Freeze S/A/D graph controls using validation-only predictive accuracy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT / "code", ROOT / "scripts"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

from graph_sparsity import (
    adaptive_topk_candidates,
    DYNAMIC_THRESHOLD_CANDIDATES,
    FINAL_GRAPH_SOURCE_POLICY,
    GRAPH_DESIGN_PROTOCOL_VERSION,
    STATIC_THRESHOLD_CANDIDATES,
)
from build_ae_final_manifest import DATASETS, GRAPH_SELECTION_PROTOCOL_VERSION, parse_csv
from build_ae_graph_tuning_manifest import NODE_COUNTS, STAGE
from data_loader import TARGET_TIMESTAMP_SPLIT_VERSION
from postprocess_ae_phase1 import parse_prediction_csv

SELECTION_VERSION = GRAPH_SELECTION_PROTOCOL_VERSION
SELECTION_RULE = "minimum_validation_WAPE_then_smase"


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return value


def expected_values(dataset: str, source: str) -> tuple[float | int, ...]:
    if source == "S":
        return STATIC_THRESHOLD_CANDIDATES
    if source == "D":
        return DYNAMIC_THRESHOLD_CANDIDATES
    return adaptive_topk_candidates(NODE_COUNTS[dataset])


def graph_value(config: dict, source: str) -> float | int:
    if source == "S":
        return float(config["static_threshold"])
    if source == "D":
        return float(config["dynamic_threshold"])
    return int(config["adaptive_top_k"])


def wape_components(pred: np.ndarray, true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if pred.shape != true.shape or pred.ndim != 3:
        raise ValueError(f"Prediction/true shapes are incompatible: {pred.shape}/{true.shape}.")
    return (
        np.abs(pred - true).sum(axis=(1, 2), dtype=np.float64),
        np.abs(true).sum(axis=(1, 2), dtype=np.float64),
    )


def validation_smase(pred: np.ndarray, true: np.ndarray, config: dict) -> float:
    metadata = config.get("smase_scale_metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Graph tuning config lacks frozen sMASE scale metadata.")
    scale = np.asarray(metadata.get("scale_per_node"), dtype=np.float64)
    if scale.shape != (pred.shape[1],):
        raise ValueError(
            f"sMASE scale shape {scale.shape} does not match {pred.shape[1]} nodes."
        )
    valid = np.isfinite(scale) & (scale > 1e-8)
    if not valid.any():
        raise ValueError("No valid nodewise sMASE denominator is available.")
    node_mae = np.abs(pred - true).mean(axis=(0, 2), dtype=np.float64)
    return float(np.mean(node_mae[valid] / scale[valid]))


def invariant_signature(config: dict) -> dict:
    """Return fields that must remain fixed throughout one graph search batch."""
    keys = (
        "model_name", "num_timesteps_in", "num_timesteps_out", "feature_set",
        "sim_type", "seed", "lr", "hidden_dim", "num_layers", "batch_size",
        "epochs", "patience", "lagtcn_graph_source_version",
        "graph_design_protocol_version", "split_protocol_version",
        "selection_source_experiment_id", "selection_protocol_version",
        "training_loss_space", "lagtcn_decoder_mode",
        "lagtcn_residual_scale_mode", "lagtcn_residual_scale_init",
    )
    return {key: config.get(key) for key in keys}


def source_revision_provenance(config: dict, config_path: Path) -> tuple[str, str]:
    """Return source provenance without enforcing a runtime Git binding."""
    commit = str(config.get("source_git_commit") or "").strip()
    branch = str(config.get("source_git_branch") or "").strip()
    if not commit or not branch:
        raise ValueError(f"{config_path}: source provenance is absent.")
    return commit, branch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("Data"))
    parser.add_argument("--experiment-id")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASETS),
        help=(
            "Comma-separated formal datasets to audit. Every S/A/D candidate for "
            "each requested dataset must be complete; omitted datasets are not "
            "represented by placeholders."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/selected_graph_hparams.generated.json"),
    )
    args = parser.parse_args()
    datasets = parse_csv(args.datasets)
    if not datasets:
        raise ValueError("At least one dataset must be requested for graph selection.")
    if len(datasets) != len(set(datasets)):
        raise ValueError(f"Duplicate datasets requested: {datasets}.")
    invalid = sorted(set(datasets) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unsupported formal datasets: {invalid}.")
    discovered = []
    for config_path in sorted(args.runs_root.rglob("config.json")):
        try:
            config = read_json(config_path)
        except Exception:
            continue
        if (
            config.get("experiment_stage") == STAGE
            and str(config.get("dataset")) in datasets
        ):
            discovered.append((config_path, config))
    experiment_ids = sorted({
        str(config.get("experiment_id"))
        for _, config in discovered if config.get("experiment_id")
    })
    experiment_id = args.experiment_id
    if experiment_id is None:
        if len(experiment_ids) != 1:
            raise RuntimeError(
                f"Graph selection requires exactly one experiment_id; found {experiment_ids}."
            )
        experiment_id = experiment_ids[0]
    discovered = [
        row for row in discovered if str(row[1].get("experiment_id")) == experiment_id
    ]
    if not discovered:
        raise RuntimeError(f"No {STAGE} runs found for experiment_id={experiment_id!r}.")

    grouped = {
        (dataset, source): []
        for dataset in datasets
        for source in ("S", "A", "D")
    }
    source_revisions = set()
    model_selection_provenance = set()
    invariant_signatures: dict[str, dict] = {}
    for config_path, config in discovered:
        if not bool(config.get("validation_only")):
            raise ValueError(f"{config_path}: graph tuning run is not validation-only.")
        if config.get("graph_sparsity_policy") != FINAL_GRAPH_SOURCE_POLICY:
            raise ValueError(f"{config_path}: wrong graph policy.")
        if config.get("graph_design_protocol_version") != GRAPH_DESIGN_PROTOCOL_VERSION:
            raise ValueError(f"{config_path}: wrong graph-design protocol version.")
        if config.get("split_protocol_version") != TARGET_TIMESTAMP_SPLIT_VERSION:
            raise ValueError(f"{config_path}: graph tuning uses a legacy split protocol.")
        dataset = str(config.get("dataset"))
        source = str(config.get("graph_mode"))
        key = (dataset, source)
        if key not in grouped:
            continue
        if str(config.get("model_name", "")).upper() != "LAGTCN":
            raise ValueError(f"{config_path}: graph tuning must use LAGTCN.")
        if (
            config.get("training_loss_space") != "original"
            or config.get("lagtcn_decoder_mode") != "persistence_residual"
            or config.get("lagtcn_residual_scale_mode") != "unit"
            or not np.isclose(
                float(config.get("lagtcn_residual_scale_init", np.nan)),
                1.0,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"{config_path}: graph tuning must use original-space loss and "
                "persistence residual with eta=1."
            )
        signature = invariant_signature(config)
        previous_signature = invariant_signatures.setdefault(dataset, signature)
        if signature != previous_signature:
            changed = sorted(
                key for key in signature if signature[key] != previous_signature[key]
            )
            raise ValueError(
                f"{config_path}: graph tuning changed non-graph controls {changed}."
            )
        commit, branch = source_revision_provenance(config, config_path)
        source_revisions.add((str(commit), str(branch)))
        model_selection_provenance.add((
            str(config.get("selection_source_experiment_id")),
            str(config.get("selection_protocol_version")),
        ))
        run_dir = config_path.parent
        pred_path = run_dir / "validation_pred.csv"
        true_path = run_dir / "validation_true.csv"
        row = {
            "value": graph_value(config, source),
            "run_dir": str(run_dir),
            "status": "incomplete",
        }
        if (run_dir / "failure.json").is_file():
            row["status"] = "failed"
        elif pred_path.is_file() and true_path.is_file():
            pred, nodes, index = parse_prediction_csv(pred_path)
            true, true_nodes, true_index = parse_prediction_csv(true_path)
            if nodes != true_nodes or not index.equals(true_index):
                raise ValueError(f"{run_dir}: validation prediction metadata differs from truth.")
            if not np.isfinite(pred).all() or not np.isfinite(true).all():
                row["status"] = "nonfinite"
            else:
                numerator, denominator = wape_components(pred, true)
                objective = 100.0 * float(numerator.sum()) / max(float(denominator.sum()), 1e-12)
                row.update({
                    "status": "finite",
                    "validation_objective": objective,
                    "validation_smase": validation_smase(pred, true, config),
                    "numerator_by_origin": numerator,
                    "denominator_by_origin": denominator,
                    "true_values": true,
                    "node_names": nodes,
                    "time_index": [str(value) for value in index],
                })
        grouped[key].append(row)

    if len(model_selection_provenance) != 1:
        raise RuntimeError("Graph tuning batch spans multiple model-selection decisions.")
    selected = {dataset: {} for dataset in datasets}
    audit = []
    for (dataset, source), rows in sorted(grouped.items()):
        values = [row["value"] for row in rows]
        if len(values) != len(set(values)) or set(values) != set(expected_values(dataset, source)):
            raise RuntimeError(
                f"{dataset}/{source}: incomplete or duplicate candidates {sorted(values)}."
            )
        finite = [row for row in rows if row["status"] == "finite"]
        if len(finite) != len(rows):
            raise RuntimeError(f"{dataset}/{source}: all candidates must finish with finite validation output.")
        reference_nodes = finite[0]["node_names"]
        reference_index = finite[0]["time_index"]
        reference_truth = finite[0]["true_values"]
        for row in finite[1:]:
            if row["node_names"] != reference_nodes or row["time_index"] != reference_index:
                raise RuntimeError(f"{dataset}/{source}: validation samples differ across candidates.")
            if not np.array_equal(row["true_values"], reference_truth):
                raise RuntimeError(f"{dataset}/{source}: validation truth differs across candidates.")
        winner = min(
            finite,
            key=lambda row: (
                row["validation_objective"],
                row["validation_smase"],
                str(row["run_dir"]),
            ),
        )
        for row in finite:
            row["difference_from_selected_WAPE"] = float(
                row["validation_objective"] - winner["validation_objective"]
            )
        output_key = {"S": "static_threshold", "A": "adaptive_top_k", "D": "dynamic_threshold"}[source]
        selected[dataset][output_key] = winner["value"]
        clean_rows = []
        for row in rows:
            clean_rows.append({key: value for key, value in row.items() if key not in {
                "numerator_by_origin", "denominator_by_origin", "true_values",
                "node_names", "time_index"
            }})
        audit.append({
            "dataset": dataset,
            "graph_source": source,
            "selection_metric": "all_level_mean_1_24_validation_WAPE_pct",
            "secondary_tie_break_metric": "all_level_mean_1_24_validation_smase",
            "selection_rule": SELECTION_RULE,
            "selected_validation_WAPE": winner["validation_objective"],
            "selected_value": winner["value"],
            "candidate_count": len(rows),
            "candidates": clean_rows,
        })

    model_experiment, model_protocol = next(iter(model_selection_provenance))
    payload = {
        "selection_protocol_version": SELECTION_VERSION,
        "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
        "source_experiment_id": experiment_id,
        "source_git_revisions": [
            {"commit": commit, "branch": branch}
            for commit, branch in sorted(source_revisions)
        ],
        "runtime_git_binding": "provenance_only",
        "model_selection_source_experiment_id": model_experiment,
        "model_selection_protocol_version": model_protocol,
        "selection_scope_datasets": list(datasets),
        "test_results_accessed": False,
        "non_graph_invariants_by_dataset": invariant_signatures,
        "selected": selected,
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "experiment_id": experiment_id,
        "selected": selected,
    }, indent=2))


if __name__ == "__main__":
    main()
