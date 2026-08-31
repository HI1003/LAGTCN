#!/usr/bin/env python3
"""Freeze model-dataset hyperparameters from one validation-only tuning batch."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from data_loader import TARGET_TIMESTAMP_SPLIT_VERSION
from build_ae_final_manifest import (
    MODEL_SELECTION_CANDIDATES_PER_GROUP,
    MODEL_SELECTION_MIN_FINITE_CANDIDATES,
    MODEL_SELECTION_PROTOCOL_VERSION,
)

MODELS = (
    "LAGTCN", "DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER",
    "DCRNN", "MTGNN",
)
DATASETS = (
    "GEFCom2012_2level",
    "GEFCom2017QualifyingMatch_3level",
    "GEFCom2017FinalMatch_4level",
)
STAGE = "ae_final_tuning_v1"
CANDIDATES_PER_GROUP = MODEL_SELECTION_CANDIDATES_PER_GROUP
TERMINAL_STATUS_VERSION = "ae_j0_terminal_status_v1"


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    return value


def _hyperparameters(config: dict) -> dict:
    hp = {
        "lr": float(config["lr"]),
        "hidden_dim": int(config["hidden_dim"]),
    }
    if str(config["model_name"]).upper() == "DEEPHGNN_SPECTGNN":
        hp.update({
            "deephgnn_hierarchical_loss_weight": float(
                config["deephgnn_hierarchical_loss_weight"]
            ),
            "spectgnn_alpha": float(config.get("spectgnn_alpha", 1.2)),
            "spectgnn_degree": int(config.get("spectgnn_degree", 4)),
            "spectgnn_modes": int(config.get("spectgnn_modes", 5)),
            "spectgnn_trend_window": int(config.get("spectgnn_trend_window", 24)),
        })
    return hp


def _candidate_signature(hyperparameters: dict) -> str:
    return json.dumps(hyperparameters, sort_keys=True, separators=(",", ":"))


def _read_manifest_entries(path: Path) -> list[dict]:
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object.")
            entries.append(value)
    return entries


def _discover(runs_root: Path) -> list[tuple[Path, dict]]:
    rows = []
    for config_path in sorted(runs_root.rglob("config.json")):
        try:
            config = _read_json(config_path)
        except Exception:
            continue
        if config.get("experiment_stage") == STAGE:
            rows.append((config_path, config))
    return rows


def _resolve_experiment(rows, requested: str | None) -> tuple[str, list]:
    ids = sorted({
        str(config.get("experiment_id"))
        for _, config in rows if config.get("experiment_id")
    })
    if requested is None:
        if len(ids) != 1:
            raise RuntimeError(
                "Hyperparameter selection requires exactly one tuning experiment_id; "
                f"found {ids}. Pass --experiment-id explicitly."
            )
        requested = ids[0]
    selected = [
        (path, config) for path, config in rows
        if str(config.get("experiment_id")) == requested
    ]
    if not selected:
        raise RuntimeError(f"No {STAGE} configs found for experiment_id={requested!r}.")
    return requested, selected


def _resolve_terminal_status_path(
    requested: Path | None,
    experiment_id: str,
) -> Path | None:
    if requested is not None:
        path = requested if requested.is_absolute() else ROOT / requested
        if not path.is_file():
            raise FileNotFoundError(f"Terminal-status audit not found: {path}.")
        return path
    directory = ROOT / "results/raw_manifests"
    matches = sorted(directory.glob(f"tuning_{experiment_id}_status_*.json"))
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple terminal-status audits found for {experiment_id}: {matches}."
        )
    return matches[0] if matches else None


def _load_terminal_registry(
    status_path: Path | None,
    experiment_id: str,
) -> dict[tuple[str, str, str], dict]:
    if status_path is None:
        return {}
    payload = _read_json(status_path)
    if payload.get("status_protocol_version") != TERMINAL_STATUS_VERSION:
        raise ValueError(f"{status_path}: unsupported terminal-status protocol.")
    if str(payload.get("experiment_id")) != experiment_id:
        raise ValueError(f"{status_path}: experiment_id does not match selection batch.")

    source_value = payload.get("source_manifest")
    if not source_value:
        raise ValueError(f"{status_path}: missing source_manifest.")
    source_manifest = Path(str(source_value))
    if not source_manifest.is_absolute():
        source_manifest = ROOT / source_manifest
    if not source_manifest.is_file():
        raise FileNotFoundError(f"{status_path}: source manifest not found: {source_manifest}.")
    source_entries = _read_manifest_entries(source_manifest)

    registry = {}
    for record in payload.get("terminal_invalid_configurations", []):
        if not isinstance(record, dict):
            raise ValueError(f"{status_path}: malformed terminal-invalid record.")
        if (
            record.get("status") != "terminal_invalid_configuration"
            or record.get("failure_type") != "numeric_divergence"
            or record.get("selection_action")
            != "exclude_from_validation_selection_without_retry"
        ):
            raise ValueError(f"{status_path}: unsupported terminal-invalid decision.")
        index = int(record.get("source_index_zero_based", -1))
        if not 0 <= index < len(source_entries):
            raise ValueError(f"{status_path}: source index {index} is out of range.")
        source = source_entries[index]
        if str(source.get("experiment_id")) != experiment_id:
            raise ValueError(f"{status_path}: source entry belongs to another experiment.")
        model = str(source.get("model", source.get("model_name", ""))).upper()
        source_hp = dict(source.get("hyperparameters") or {})
        source_config = {
            **source_hp,
            "model_name": model,
        }
        hyperparameters = _hyperparameters(source_config)
        declared = record.get("hyperparameters")
        if not isinstance(declared, dict):
            raise ValueError(f"{status_path}: terminal record lacks hyperparameters.")
        for name, value in hyperparameters.items():
            if name not in declared or declared[name] != value:
                raise ValueError(
                    f"{status_path}: terminal record differs from source manifest for {name}."
                )
        if str(record.get("model", "")).upper() != model:
            raise ValueError(f"{status_path}: terminal model differs from source manifest.")
        key = (
            str(source.get("dataset")),
            model,
            _candidate_signature(hyperparameters),
        )
        if key in registry:
            raise ValueError(f"{status_path}: duplicate terminal decision for {key}.")
        registry[key] = {
            **record,
            "status_file": str(status_path),
            "source_manifest": str(source_manifest),
        }
    return registry


def _canonicalize_candidates(
    grouped: dict,
    terminal_registry: dict[tuple[str, str, str], dict],
) -> tuple[dict, set[tuple[str, str, str]]]:
    canonical = {}
    used_terminal = set()
    for (dataset, model), attempts in grouped.items():
        by_signature = {}
        for attempt in attempts:
            signature = _candidate_signature(attempt["hyperparameters"])
            by_signature.setdefault(signature, []).append(attempt)
        rows = []
        for signature, candidate_attempts in sorted(by_signature.items()):
            finite = [row for row in candidate_attempts if row["status"] == "finite"]
            terminal_key = (dataset, model, signature)
            terminal = terminal_registry.get(terminal_key)
            if len(finite) > 1:
                raise RuntimeError(
                    f"{dataset}/{model}: candidate {signature} has multiple finite attempts."
                )
            if finite and terminal is not None:
                raise RuntimeError(
                    f"{dataset}/{model}: candidate {signature} is both finite and terminal-invalid."
                )
            if finite:
                row = dict(finite[0])
                row["attempts"] = candidate_attempts
            elif terminal is not None:
                used_terminal.add(terminal_key)
                row = {
                    "hyperparameters": candidate_attempts[0]["hyperparameters"],
                    "run_dir": candidate_attempts[0]["run_dir"],
                    "config_file": candidate_attempts[0]["config_file"],
                    "status": "terminal_invalid",
                    "validation_objective": None,
                    "validation_smase": None,
                    "terminal_decision": terminal,
                    "attempts": candidate_attempts,
                }
            else:
                row = dict(candidate_attempts[-1])
                row["attempts"] = candidate_attempts
            rows.append(row)
        canonical[(dataset, model)] = rows
    return canonical, used_terminal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("Data"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--terminal-status", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/selected_hparams.generated.json"),
    )
    args = parser.parse_args()

    experiment_id, discovered = _resolve_experiment(
        _discover(args.runs_root), args.experiment_id
    )
    terminal_status_path = _resolve_terminal_status_path(
        args.terminal_status, experiment_id
    )
    terminal_registry = _load_terminal_registry(terminal_status_path, experiment_id)
    grouped = {(dataset, model): [] for dataset in DATASETS for model in MODELS}
    source_revisions = set()
    for config_path, config in discovered:
        if not bool(config.get("validation_only")):
            raise ValueError(f"{config_path}: tuning run is not validation-only.")
        if int(config.get("num_timesteps_out", 0)) != 24:
            raise ValueError(f"{config_path}: tuning output length is not 24.")
        if config.get("training_loss_space") != "original":
            raise ValueError(f"{config_path}: tuning loss is not in original load units.")
        key = (str(config.get("dataset")), str(config.get("model_name", "")).upper())
        if key[1] == "LAGTCN" and (
            config.get("lagtcn_decoder_mode") != "persistence_residual"
            or config.get("lagtcn_residual_scale_mode") != "unit"
            or not math.isclose(
                float(config.get("lagtcn_residual_scale_init", math.nan)),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                f"{config_path}: LAGTCN tuning does not use persistence residual with eta=1."
            )
        if config.get("split_protocol_version") != TARGET_TIMESTAMP_SPLIT_VERSION:
            raise ValueError(f"{config_path}: tuning run uses a legacy split protocol.")
        if key not in grouped:
            continue
        commit = config.get("source_git_commit")
        branch = config.get("source_git_branch")
        if not commit or not branch:
            raise ValueError(f"{config_path}: missing source git provenance.")
        source_revisions.add((str(commit), str(branch)))
        hp = _hyperparameters(config)
        run_dir = config_path.parent
        metrics_path = run_dir / "validation_metrics.json"
        failure_path = run_dir / "failure.json"
        row = {
            "hyperparameters": hp,
            "run_dir": str(run_dir),
            "config_file": str(config_path),
            "status": "incomplete",
            "validation_objective": None,
            "validation_smase": None,
        }
        if failure_path.is_file():
            row["status"] = "failed"
            try:
                row["failure"] = _read_json(failure_path)
            except Exception as exc:
                row["failure"] = {"unreadable_failure_json": repr(exc)}
        elif metrics_path.is_file():
            metrics = _read_json(metrics_path)
            objective = metrics.get("WAPE")
            smase = metrics.get("MASE")
            if (
                objective is not None
                and smase is not None
                and math.isfinite(float(objective))
                and math.isfinite(float(smase))
            ):
                row["status"] = "finite"
                row["validation_objective"] = float(objective)
                row["validation_smase"] = float(smase)
            else:
                row["status"] = "nonfinite_metric"
        grouped[key].append(row)

    grouped, used_terminal = _canonicalize_candidates(grouped, terminal_registry)
    unused_terminal = sorted(set(terminal_registry) - used_terminal)
    if unused_terminal:
        raise RuntimeError(
            "Terminal-status audit contains candidates absent from the discovered batch: "
            f"{unused_terminal}."
        )

    malformed = []
    for key, rows in grouped.items():
        if len(rows) != CANDIDATES_PER_GROUP:
            malformed.append(f"{key[0]}/{key[1]}={len(rows)}")
            continue
        signatures = [json.dumps(row["hyperparameters"], sort_keys=True) for row in rows]
        if len(signatures) != len(set(signatures)):
            malformed.append(f"{key[0]}/{key[1]}=duplicate-hyperparameters")
    if malformed:
        raise RuntimeError(
            "Tuning matrix is incomplete or duplicated; expected four unique candidates per group: "
            + ", ".join(malformed)
        )

    selected = {dataset: {} for dataset in DATASETS}
    audit = []
    unresolved_groups = []
    for (dataset, model), rows in sorted(grouped.items()):
        finite = [row for row in rows if row["status"] == "finite"]
        finite.sort(key=lambda row: (
            row["validation_objective"],
            row["validation_smase"],
            json.dumps(row["hyperparameters"], sort_keys=True),
            row["run_dir"],
        ))
        terminal = all(
            row["status"] in {"finite", "terminal_invalid"} for row in rows
        )
        if not terminal or len(finite) < MODEL_SELECTION_MIN_FINITE_CANDIDATES:
            statuses = {status: 0 for status in (
                "finite", "terminal_invalid", "failed", "incomplete",
                "nonfinite_metric",
            )}
            for row in rows:
                statuses[row["status"]] = statuses.get(row["status"], 0) + 1
            unresolved_groups.append(f"{dataset}/{model}={statuses}")
            continue
        winner = finite[0]
        selected[dataset][model] = winner["hyperparameters"]
        audit.append({
            "dataset": dataset,
            "model": model,
            "selection_metric": "all_level_mean_1_24_validation_WAPE_pct",
            "secondary_tie_break_metric": "all_level_mean_1_24_validation_smase",
            "selection_rule": "minimum_WAPE_then_smase_on_exact_tie",
            "winner": winner,
            "candidate_count": len(rows),
            "finite_candidate_count": len(finite),
            "ineligible_candidate_count": len(rows) - len(finite),
            "all_candidates_terminal": True,
            "candidates": rows,
        })
    if unresolved_groups:
        raise RuntimeError(
            "Cannot freeze the formal matrix until all four preregistered candidates "
            "have auditable terminal outcomes and at least "
            f"{MODEL_SELECTION_MIN_FINITE_CANDIDATES} finite validation WAPE/sMASE "
            "candidate(s) per group: "
            + ", ".join(unresolved_groups)
        )

    payload = {
        "selection_protocol_version": MODEL_SELECTION_PROTOCOL_VERSION,
        "source_experiment_id": experiment_id,
        "source_git_revisions": [
            {"commit": commit, "branch": branch}
            for commit, branch in sorted(source_revisions)
        ],
        "candidate_terminal_policy": (
            "all_preregistered_candidates_accounted; terminal numeric-invalid "
            "candidates are ineligible; select among finite candidates"
        ),
        "terminal_status_file": str(terminal_status_path) if terminal_status_path else None,
        "runtime_git_binding": "provenance_only",
        "test_results_accessed": False,
        "includes_deephgnn": False,
        "optional_phase2_models": ["DEEPHGNN_SPECTGNN"],
        "selected": selected,
        "audit": audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "audit": str(args.output),
        "experiment_id": experiment_id,
        "selected_groups": len(audit),
    }, indent=2))


if __name__ == "__main__":
    main()
