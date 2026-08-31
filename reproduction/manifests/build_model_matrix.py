#!/usr/bin/env python3
"""Build the validation-tuning and final paper model manifests.

The full main manifest emits 144 runs: 90 LAGTCN graph runs and 54
external-baseline runs.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from lagtcn.core.graphs import (
    adaptive_topk_candidates,
    DYNAMIC_THRESHOLD_CANDIDATES,
    FINAL_GRAPH_SOURCE_POLICY,
    GRAPH_DESIGN_PROTOCOL_VERSION,
    STATIC_THRESHOLD_CANDIDATES,
)
from lagtcn.core.naming import (
    graph_alias,
    graph_components,
    model_alias,
    shorten_existing_run_label,
)

DATASETS = (
    "GEFCom2012_2level",
    "GEFCom2017QualifyingMatch_3level",
    "GEFCom2017FinalMatch_4level",
)
SEEDS = (42, 43, 44)
MODEL_SELECTION_PROTOCOL_VERSION = (
    "ae_phase1_validation_wape_v6_one_finite_terminal_outcomes"
)
MODEL_SELECTION_CANDIDATES_PER_GROUP = 4
MODEL_SELECTION_MIN_FINITE_CANDIDATES = 1
GRAPH_SELECTION_PROTOCOL_VERSION = (
    "ae_graph_validation_accuracy_v4_original_persistence_unit"
)
LAGTCN_GRAPHS = ("H", "HG", "S", "A", "D", "S+A+D", "H+S", "H+A", "H+D", "H+S+A+D")
TEMPORAL_MODELS = ("DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER")
CURRENT_MODEL_CONFIGS = (
    *(dict(model=model, graph="H", gnn="none", temporal="gru", source="project", role="base")
      for model in TEMPORAL_MODELS),
    dict(model="DCRNN", graph="H", gnn="gcn", temporal="gru", source="project", role="base"),
    dict(model="MTGNN", graph="H", gnn="gcn", temporal="gru", source="native", role="base"),
)
DEFAULT_HIDDEN = {
    "GEFCom2012_2level": 64,
    "GEFCom2017QualifyingMatch_3level": 64,
    "GEFCom2017FinalMatch_4level": 128,
}
DATASET_NODE_COUNTS = {
    "GEFCom2012_2level": 21,
    "GEFCom2017QualifyingMatch_3level": 15,
    "GEFCom2017FinalMatch_4level": 158,
}
DATASET_ALIASES = {
    "GEFCom2012_2level": "2l",
    "GEFCom2017QualifyingMatch_3level": "3l",
    "GEFCom2017FinalMatch_4level": "4l",
}

FORMAL_EFFECTIVE_BATCH_SIZE = 128
FOUR_LEVEL_STGNN_PHYSICAL_BATCH = {
    ("DCRNN", 128): 64,
    ("DCRNN", 256): 32,
    ("MTGNN", 128): 32,
    ("MTGNN", 256): 16,
}


def parse_csv(value: str, cast=str):
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def formal_batch_protocol(
    dataset: str,
    model: str,
    hidden_dim: int,
    effective_batch_size: int = FORMAL_EFFECTIVE_BATCH_SIZE,
) -> dict[str, int]:
    """Return the frozen physical-batch/accumulation protocol for one run."""
    effective = int(effective_batch_size)
    if effective < 1:
        raise ValueError("effective_batch_size must be positive.")
    model = str(model).upper()
    physical = effective
    if dataset == "GEFCom2017FinalMatch_4level" and model in {"DCRNN", "MTGNN"}:
        key = (model, int(hidden_dim))
        if key not in FOUR_LEVEL_STGNN_PHYSICAL_BATCH:
            raise ValueError(
                "No audited 4-level physical-batch protocol for "
                f"{model} hidden_dim={hidden_dim}."
            )
        physical = min(effective, FOUR_LEVEL_STGNN_PHYSICAL_BATCH[key])
    if effective % physical:
        raise ValueError(
            f"Effective batch {effective} is not divisible by physical batch {physical}."
        )
    accumulation = effective // physical
    return {
        "physical_batch_size": int(physical),
        "gradient_accumulation_steps": int(accumulation),
        "effective_batch_size": int(effective),
    }


def git_source() -> tuple[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
    try:
        return run("rev-parse", "HEAD"), run("branch", "--show-current") or "DETACHED"
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot freeze manifest source revision: {exc}") from exc


def load_selected(
    path: Path | None,
    *,
    model_configs=CURRENT_MODEL_CONFIGS,
) -> tuple[dict, dict]:
    if path is None:
        raise FileNotFoundError(
            "Main/ablation manifests require the audited selected_hparams.json."
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen validation selection not found: {path}. Build tuning manifest first, "
            "then freeze selected_hparams.json; do not select from test results."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("selected"), dict):
        raise ValueError(
            "--selected-hparams must be the full audited selected_hparams.json, "
            "not the provenance-free nested manifest input."
        )
    if payload.get("selection_protocol_version") != MODEL_SELECTION_PROTOCOL_VERSION:
        raise ValueError("Unsupported or legacy hyperparameter-selection protocol.")
    if payload.get("test_results_accessed") is not False:
        raise ValueError("Hyperparameter selection is not certified test-blind.")
    expected_groups = {
        (dataset, model)
        for dataset in DATASETS
        for model in ("LAGTCN", *(item["model"] for item in model_configs))
    }
    audit = payload.get("audit")
    if not isinstance(audit, list):
        raise ValueError("Audited selection lacks the per-group candidate audit.")
    audit_keys = [
        (str(row.get("dataset")), str(row.get("model")).upper())
        for row in audit if isinstance(row, dict)
    ]
    if len(audit_keys) != len(set(audit_keys)) or set(audit_keys) != expected_groups:
        raise ValueError(
            "Audited selection must contain exactly one record for every formal "
            f"model/dataset group; found {len(audit_keys)}, expected {len(expected_groups)}."
        )
    for row in audit:
        key = (str(row["dataset"]), str(row["model"]).upper())
        candidates = row.get("candidates")
        winner = row.get("winner")
        finite_candidate_count = int(row.get("finite_candidate_count", 0))
        ineligible_candidate_count = int(row.get("ineligible_candidate_count", -1))
        if (
            row.get("selection_metric")
            != "all_level_mean_1_24_validation_WAPE_pct"
            or row.get("secondary_tie_break_metric")
            != "all_level_mean_1_24_validation_smase"
            or row.get("selection_rule")
            != "minimum_WAPE_then_smase_on_exact_tie"
            or int(row.get("candidate_count", -1))
            != MODEL_SELECTION_CANDIDATES_PER_GROUP
            or not isinstance(candidates, list)
            or len(candidates) != MODEL_SELECTION_CANDIDATES_PER_GROUP
            or not MODEL_SELECTION_MIN_FINITE_CANDIDATES
            <= finite_candidate_count
            <= MODEL_SELECTION_CANDIDATES_PER_GROUP
            or ineligible_candidate_count
            != MODEL_SELECTION_CANDIDATES_PER_GROUP - finite_candidate_count
            or row.get("all_candidates_terminal") is not True
            or not isinstance(winner, dict)
        ):
            raise ValueError(f"Malformed tuning audit for {key[0]}/{key[1]}.")
        signatures = [
            json.dumps(candidate.get("hyperparameters"), sort_keys=True)
            for candidate in candidates if isinstance(candidate, dict)
        ]
        finite = [
            candidate for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("status") == "finite"
            and candidate.get("validation_objective") is not None
            and math.isfinite(float(candidate["validation_objective"]))
            and candidate.get("validation_smase") is not None
            and math.isfinite(float(candidate["validation_smase"]))
        ]
        terminal_invalid = [
            candidate for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("status") == "terminal_invalid"
        ]
        terminal_evidence_valid = all(
            isinstance(candidate.get("terminal_decision"), dict)
            and candidate["terminal_decision"].get("status")
            == "terminal_invalid_configuration"
            and candidate["terminal_decision"].get("failure_type")
            == "numeric_divergence"
            and candidate["terminal_decision"].get("selection_action")
            == "exclude_from_validation_selection_without_retry"
            and bool(candidate["terminal_decision"].get("status_file"))
            and bool(candidate["terminal_decision"].get("source_manifest"))
            and isinstance(candidate.get("hyperparameters"), dict)
            and isinstance(candidate["terminal_decision"].get("hyperparameters"), dict)
            and all(
                candidate["terminal_decision"]["hyperparameters"].get(name) == value
                for name, value in candidate["hyperparameters"].items()
            )
            for candidate in terminal_invalid
        )
        if (
            len(signatures) != MODEL_SELECTION_CANDIDATES_PER_GROUP
            or len(set(signatures)) != MODEL_SELECTION_CANDIDATES_PER_GROUP
            or len(finite) != finite_candidate_count
            or len(terminal_invalid) != ineligible_candidate_count
            or not terminal_evidence_valid
            or any(
                not isinstance(candidate.get("hyperparameters"), dict)
                for candidate in candidates
            )
            or any(
                candidate.get("status") not in {"finite", "terminal_invalid"}
                for candidate in candidates
                if isinstance(candidate, dict)
            )
        ):
            raise ValueError(f"Invalid candidate records for {key[0]}/{key[1]}.")
        selected_hp_value = payload["selected"].get(key[0], {}).get(key[1])
        expected_winner = min(
            finite,
            key=lambda candidate: (
                float(candidate["validation_objective"]),
                float(candidate["validation_smase"]),
                json.dumps(candidate["hyperparameters"], sort_keys=True),
                str(candidate.get("run_dir", "")),
            ),
        )
        if (
            winner.get("hyperparameters") != selected_hp_value
            or winner.get("status") != "finite"
            or winner != expected_winner
        ):
            raise ValueError(
                f"Selected hyperparameters and audited winner differ for {key[0]}/{key[1]}."
            )

    metadata = {
        "selection_source_experiment_id": payload.get("source_experiment_id"),
        "selection_protocol_version": payload.get("selection_protocol_version"),
    }
    if not metadata["selection_source_experiment_id"]:
        raise ValueError("Audited selection lacks source_experiment_id.")
    return payload["selected"], metadata


def selected_hp(selected: dict, dataset: str, model: str) -> dict:
    value = selected.get(dataset, {}).get(model)
    if not isinstance(value, dict):
        raise KeyError(f"Missing frozen validation hyperparameters for {dataset}/{model}.")
    required = {"lr", "hidden_dim"}
    missing = sorted(required - set(value))
    if missing:
        raise KeyError(f"Missing {missing} for {dataset}/{model} in selected_hparams.")
    return value


def load_selected_graphs(
    path: Path | None,
    *,
    datasets: tuple[str, ...] = DATASETS,
) -> tuple[dict, dict]:
    if path is None or not path.is_file():
        raise FileNotFoundError(
            "Main/ablation manifests require audited selected_graph_hparams.json."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selection_protocol_version") != GRAPH_SELECTION_PROTOCOL_VERSION:
        raise ValueError("Unsupported or legacy graph-selection protocol.")
    if payload.get("graph_design_protocol_version") != GRAPH_DESIGN_PROTOCOL_VERSION:
        raise ValueError("Graph selection uses a different graph-design protocol.")

    if payload.get("test_results_accessed") is not False:
        raise ValueError("Graph selection is not certified test-blind.")
    selected = payload.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("Graph selection lacks the selected mapping.")
    requested = tuple(datasets)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError(f"Invalid requested graph-selection dataset scope: {requested}.")
    invalid_requested = sorted(set(requested) - set(DATASETS))
    if invalid_requested:
        raise ValueError(f"Unsupported graph-selection datasets: {invalid_requested}.")
    declared_scope = payload.get("selection_scope_datasets")
    if declared_scope is None:
        # Backward compatibility for a complete pre-scope artifact.
        scope = tuple(dataset for dataset in DATASETS if dataset in selected)
    elif isinstance(declared_scope, list):
        scope = tuple(str(dataset) for dataset in declared_scope)
    else:
        raise ValueError("Graph selection has a malformed dataset scope.")
    if (
        not scope
        or len(scope) != len(set(scope))
        or set(scope) - set(DATASETS)
        or set(selected) != set(scope)
    ):
        raise ValueError("Graph selection dataset scope and selected mapping differ.")
    missing_requested = sorted(set(requested) - set(scope))
    if missing_requested:
        raise ValueError(
            "Graph selection does not audit the requested datasets: "
            f"{missing_requested}."
        )
    expected_keys = {"static_threshold", "adaptive_top_k", "dynamic_threshold"}
    for dataset in scope:
        value = selected.get(dataset)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise ValueError(f"Graph selection for {dataset} must contain {sorted(expected_keys)}.")
        if not 0.0 <= float(value["static_threshold"]) <= 1.0:
            raise ValueError(f"Invalid static_threshold for {dataset}.")
        if int(value["adaptive_top_k"]) < 0:
            raise ValueError(f"Invalid adaptive_top_k for {dataset}.")
        if not 0.0 <= float(value["dynamic_threshold"]) <= 1.0:
            raise ValueError(f"Invalid dynamic_threshold for {dataset}.")
    audit = payload.get("audit")
    expected_audit = {
        (dataset, source) for dataset in scope for source in ("S", "A", "D")
    }
    actual_audit = {
        (str(row.get("dataset")), str(row.get("graph_source")))
        for row in audit or [] if isinstance(row, dict)
    }
    if actual_audit != expected_audit or len(audit or []) != len(expected_audit):
        raise ValueError("Graph selection lacks one auditable S/A/D decision per dataset.")
    audit_by_key = {
        (str(row["dataset"]), str(row["graph_source"])): row for row in audit
    }
    output_key = {"S": "static_threshold", "A": "adaptive_top_k", "D": "dynamic_threshold"}
    for dataset, source in sorted(expected_audit):
        row = audit_by_key[(dataset, source)]
        candidates = row.get("candidates")
        if source == "S":
            expected_values = set(STATIC_THRESHOLD_CANDIDATES)
        elif source == "D":
            expected_values = set(DYNAMIC_THRESHOLD_CANDIDATES)
        else:
            expected_values = set(adaptive_topk_candidates(DATASET_NODE_COUNTS[dataset]))
        if (
            row.get("selection_metric") != "all_level_mean_1_24_validation_WAPE_pct"
            or row.get("secondary_tie_break_metric")
            != "all_level_mean_1_24_validation_smase"
            or row.get("selection_rule") != "minimum_validation_WAPE_then_smase"
            or not isinstance(candidates, list)
            or int(row.get("candidate_count", -1)) != len(expected_values)
            or len(candidates) != len(expected_values)
            or {candidate.get("value") for candidate in candidates} != expected_values
            or any(candidate.get("status") != "finite" for candidate in candidates)
        ):
            raise ValueError(f"Malformed graph-selection audit for {dataset}/{source}.")
        expected_winner = min(
            candidates,
            key=lambda candidate: (
                candidate["validation_objective"],
                candidate["validation_smase"],
                str(candidate["run_dir"]),
            ),
        )["value"]
        selected_value = selected[dataset][output_key[source]]
        if row.get("selected_value") != selected_value or selected_value != expected_winner:
            raise ValueError(f"Frozen graph winner differs from its audit for {dataset}/{source}.")
    experiment_id = payload.get("source_experiment_id")
    if not experiment_id:
        raise ValueError("Graph selection lacks source_experiment_id.")
    return {dataset: selected[dataset] for dataset in requested}, {
        "graph_selection_source_experiment_id": experiment_id,
        "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
        "graph_selection_protocol_version": payload["selection_protocol_version"],
        "graph_selection_scope_datasets": list(scope),
    }


def dataset_manifest_suffix(datasets: tuple[str, ...]) -> str:
    """Keep partial-dataset manifests distinct while preserving full names."""
    if tuple(datasets) == DATASETS:
        return ""
    return "_" + "".join(DATASET_ALIASES[dataset] for dataset in datasets)


def graph_flags(graph: str, graph_hp: dict | None) -> list[str]:
    tokens = graph_components(graph)
    if not tokens.intersection({"S", "A", "D"}):
        return []
    if not isinstance(graph_hp, dict):
        raise ValueError(f"Graph mode {graph} requires frozen graph hyperparameters.")
    flags: list[str] = []
    if "S" in tokens:
        flags += ["--static-threshold", str(graph_hp["static_threshold"])]
    if "A" in tokens:
        flags += ["--adaptive-top-k", str(graph_hp["adaptive_top_k"])]
    if "D" in tokens:
        flags += ["--dynamic-threshold", str(graph_hp["dynamic_threshold"])]
    return flags


def _compact_scientific(value: float) -> str:
    number = float(value)
    if number == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(number)))
    mantissa = number / (10 ** exponent)
    return f"{mantissa:.6g}e{exponent}"


def _compact_decimal(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def formal_run_label(stage: str, item: dict, hp: dict, seed: int, graph_hp=None) -> str:
    """Build a compact, human-readable label for one formal-paper run."""
    model = str(item["model"]).upper()
    graph = str(item["graph"])
    parts = [model_alias(model)]

    # Temporal-only models have no operational graph. Native identifies the
    # MTGNN learned directed graph; other graph models retain their source.
    if str(item["gnn"]).lower() != "none":
        if str(item["source"]).lower() == "native":
            parts.append("nat")
        else:
            parts.append(graph_alias(graph))
    parts.extend(("h24", f"s{int(seed)}"))

    if stage == "ae_final_tuning_v1":
        parts.append(f"lr{_compact_scientific(hp['lr'])}")
        if model != "DLINEAR":
            parts.append(f"d{int(hp['hidden_dim'])}")
    elif stage == "ae_final_graph_tuning_v3":
        graph_hp = graph_hp or {}
        if graph == "A":
            parts.append(f"k{int(graph_hp['adaptive_top_k'])}")
        elif graph == "S":
            parts.append("t" + _compact_decimal(graph_hp["static_threshold"]))
        elif graph == "D":
            parts.append("t" + _compact_decimal(graph_hp["dynamic_threshold"]))
    return shorten_existing_run_label("_".join(parts))


def common_command(
    args, dataset, seed, item, hp, experiment_id, tuning=False,
    graph_hp=None, stage_override=None, namespace_override=None,
):
    stage = (
        stage_override if stage_override
        else "ae_final_tuning_v1" if tuning
        else "ae_final_main_v1"
    )
    graph = item["graph"]
    model = item["model"]
    batch_protocol = formal_batch_protocol(
        dataset,
        model,
        int(hp["hidden_dim"]),
        int(args.batch_size),
    )
    label = formal_run_label(stage, item, hp, seed, graph_hp=graph_hp)
    namespace_base = namespace_override or (
        "ae/j0" if tuning
        else "ae/main"
    )
    output_namespace = f"{namespace_base.rstrip('/')}/{experiment_id}"
    cmd = [
        sys.executable, "-m", "lagtcn.train",
        "--dataset", dataset,
        "--feature-set", "target",
        "--num-timesteps-in", "168",
        "--num-timesteps-out", "24",
        "--model-name", model,
        "--graph-mode", graph,
        "--gnn-type", item["gnn"],
        "--temporal-type", item["temporal"],
        "--stgnn-graph-source", item["source"],
        "--native-top-k", "5",
        "--coherency-lambda", "0",
        "--training-loss-space", "original",
        "--graph-sparsity-policy", FINAL_GRAPH_SOURCE_POLICY,
        *graph_flags(graph, graph_hp),
        "--hidden-dim", str(hp["hidden_dim"]),
        "--num-layers", str(args.num_layers),
        "--lr", str(hp["lr"]),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(batch_protocol["physical_batch_size"]),
        "--gradient-accumulation-steps",
        str(batch_protocol["gradient_accumulation_steps"]),
        "--seed", str(seed),
        "--device", args.device,
        "--resume", "auto",
        *([
            "--selection-source-experiment-id", args.selection_source_experiment_id,
            "--selection-protocol-version", args.selection_protocol_version,
        ] if args.selection_source_experiment_id else []),
        *([
            "--graph-selection-source-experiment-id", args.graph_selection_source_experiment_id,
            "--graph-selection-protocol-version", args.graph_selection_protocol_version,
        ] if getattr(args, "graph_selection_source_experiment_id", None) else []),
        "--paper-scope", "journal_applied_energy",
        "--experiment-stage", stage,
        "--experiment-id", experiment_id,
        "--output-namespace", output_namespace,
        "--run-label", label,
        "--checkpoint-every-epochs", "1",
        "--no-plots",
    ]
    if model == "LAGTCN":
        cmd += [
            "--lagtcn-decoder-mode", "persistence_residual",
            "--lagtcn-residual-scale-mode", "unit",
            "--lagtcn-residual-scale-init", "1",
        ]
    if model == "LAGTCN" and item.get("ablation"):
        cmd += ["--lagtcn-ablation", item["ablation"]]
    if tuning:
        cmd.append("--validation-only")
    return label, cmd


def validate_unique_run_labels(runs: list[dict]) -> None:
    """Reject labels that could resolve to the same formal output directory."""
    seen: dict[tuple[str, str, str], int] = {}
    for index, run in enumerate(runs, start=1):
        key = (
            str(run["dataset"]),
            str(run["graph"]),
            str(run["run_label"]),
        )
        previous = seen.get(key)
        if previous is not None:
            raise ValueError(
                "Formal manifest contains a duplicate effective run label in "
                f"one dataset/graph output directory: lines {previous} and "
                f"{index}, key={key}."
            )
        seen[key] = index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix", choices=("tuning", "main"), required=True
    )
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument("--selected-hparams", type=Path)
    parser.add_argument("--selected-graph-hparams", type=Path)
    parser.add_argument("--manifest-dir", type=Path, default=Path("results/raw_manifests"))
    parser.add_argument("--experiment-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size != FORMAL_EFFECTIVE_BATCH_SIZE:
        parser.error(
            "Formal AE manifests require effective batch size "
            f"{FORMAL_EFFECTIVE_BATCH_SIZE}; physical batches are assigned per run."
        )
    args.source_git_commit, args.source_git_branch = git_source()
    model_configs = CURRENT_MODEL_CONFIGS

    datasets = parse_csv(args.datasets)
    invalid = sorted(set(datasets) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unsupported formal datasets: {invalid}")
    seeds = parse_csv(args.seeds, int)
    experiment_id = args.experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    selection_metadata = {}
    if args.matrix == "main":
        selected, selection_metadata = load_selected(
            args.selected_hparams,
            model_configs=model_configs,
        )
        selected_graphs, graph_selection_metadata = load_selected_graphs(
            args.selected_graph_hparams,
            datasets=datasets,
        )
    else:
        selected = {}
        selected_graphs = {}
        graph_selection_metadata = {}
    args.selection_source_experiment_id = selection_metadata.get(
        "selection_source_experiment_id"
    )
    args.selection_protocol_version = selection_metadata.get(
        "selection_protocol_version"
    )
    args.graph_selection_source_experiment_id = graph_selection_metadata.get(
        "graph_selection_source_experiment_id"
    )
    args.graph_selection_protocol_version = graph_selection_metadata.get(
        "graph_selection_protocol_version"
    )

    runs = []
    if args.matrix == "main":
        for dataset in datasets:
            for seed in seeds:
                for graph in LAGTCN_GRAPHS:
                    item = dict(
                        model="LAGTCN", graph=graph, gnn="gcn",
                        temporal="patch_transformer", source="project", role="base",
                    )
                    hp = selected_hp(selected, dataset, "LAGTCN")
                    label, cmd = common_command(
                        args, dataset, seed, item, hp, experiment_id,
                        graph_hp=selected_graphs[dataset],
                    )
                    runs.append({**item, "dataset": dataset, "seed": seed, "horizon": 24,
                                 "run_label": label, "hyperparameters": hp, "cmd": cmd})
                for item in model_configs:
                    hp = selected_hp(selected, dataset, item["model"])
                    label, cmd = common_command(args, dataset, seed, item, hp, experiment_id)
                    runs.append({**item, "dataset": dataset, "seed": seed, "horizon": 24,
                                 "run_label": label, "hyperparameters": hp, "cmd": cmd})
        expected = len(datasets) * len(seeds) * (len(LAGTCN_GRAPHS) + len(model_configs))
        if len(runs) != expected:
            raise AssertionError(f"Built {len(runs)} main runs, expected {expected}.")
        full_expected = 144
        if datasets == DATASETS and seeds == SEEDS and len(runs) != full_expected:
            raise AssertionError(
                f"Full main matrix must contain {full_expected} runs, got {len(runs)}."
            )
    elif args.matrix == "tuning":
        tuning_items = [
            dict(model="LAGTCN", graph="H", gnn="gcn", temporal="patch_transformer", source="project", role="base"),
            *model_configs,
        ]
        for dataset in datasets:
            for item in tuning_items:
                if item["model"] == "DLINEAR":
                    # DLinear has no hidden representation; varying hidden_dim would
                    # create duplicate effective candidates. Tune four learning rates.
                    candidates = [
                        dict(
                            lr=lr,
                            hidden_dim=DEFAULT_HIDDEN[dataset],
                            variant=f"lr{lr}",
                        )
                        for lr in (2.5e-4, 5e-4, 1e-3, 2e-3)
                    ]
                else:
                    candidates = [
                        dict(lr=lr, hidden_dim=hidden, variant=f"lr{lr}_hidden{hidden}")
                        for lr in (5e-4, 1e-3)
                        for hidden in (DEFAULT_HIDDEN[dataset], DEFAULT_HIDDEN[dataset] * 2)
                    ]
                for hp in candidates:
                    label, cmd = common_command(
                        args, dataset, args.tuning_seed, item, hp, experiment_id, tuning=True
                    )
                    runs.append({**item, "dataset": dataset, "seed": args.tuning_seed,
                                 "horizon": 24, "run_label": label,
                                 "hyperparameters": hp, "selection_partition": "validation", "cmd": cmd})
        expected = len(datasets) * len(tuning_items) * 4
        if len(runs) != expected:
            raise AssertionError(f"Built {len(runs)} tuning pilots, expected {expected}.")
        full_expected = 84
        if datasets == DATASETS and len(runs) != full_expected:
            raise AssertionError(
                f"Full tuning matrix must contain {full_expected} runs, got {len(runs)}."
            )
    for run in runs:
        resource = formal_batch_protocol(
            str(run["dataset"]),
            str(run["model"]),
            int(run["hyperparameters"]["hidden_dim"]),
            args.batch_size,
        )
        run.update(resource)
        run["experiment_id"] = experiment_id
        run.update(selection_metadata)
        run.update(graph_selection_metadata)
        run["source_git_commit"] = args.source_git_commit
        run["source_git_branch"] = args.source_git_branch
        run["runtime_git_binding"] = "provenance_only"

    validate_unique_run_labels(runs)

    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    scope_suffix = dataset_manifest_suffix(datasets)
    path = args.manifest_dir / f"{args.matrix}_{experiment_id}{scope_suffix}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "manifest": str(path),
        "matrix": args.matrix,
        "run_count": len(runs),
        "datasets": datasets,
        "seeds": [args.tuning_seed] if args.matrix == "tuning" else seeds,
        "protocol": "ae_phase1_original_persistence_unit_v1",
        "graph_policy": FINAL_GRAPH_SOURCE_POLICY,
        "source_git_commit": args.source_git_commit,
        "source_git_branch": args.source_git_branch,
        "runtime_git_binding": "provenance_only",
        **selection_metadata,
        **graph_selection_metadata,
    }, indent=2))


if __name__ == "__main__":
    main()
