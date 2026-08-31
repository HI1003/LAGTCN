#!/usr/bin/env python3
"""Build validation-only S/A/D graph-hyperparameter tuning runs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

from lagtcn.core.graphs import (
    adaptive_topk_candidates,
    DYNAMIC_THRESHOLD_CANDIDATES,
    FINAL_GRAPH_SOURCE_POLICY,
    GRAPH_DESIGN_PROTOCOL_VERSION,
    STATIC_THRESHOLD_CANDIDATES,
)
from reproduction.manifests.build_model_matrix import (
    CURRENT_MODEL_CONFIGS,
    FORMAL_EFFECTIVE_BATCH_SIZE,
    DATASET_NODE_COUNTS,
    DATASETS,
    common_command,
    git_source,
    formal_batch_protocol,
    load_selected,
    selected_hp,
)

STAGE = "ae_final_graph_tuning_v3"
NAMESPACE = "ae/j1"
NODE_COUNTS = DATASET_NODE_COUNTS


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def candidates(dataset: str):
    for value in STATIC_THRESHOLD_CANDIDATES:
        yield "S", {"static_threshold": value}, f"tauS{value:g}"
    clipped_topk = adaptive_topk_candidates(NODE_COUNTS[dataset])
    for value in clipped_topk:
        yield "A", {"adaptive_top_k": value}, f"kA{value}"
    for value in DYNAMIC_THRESHOLD_CANDIDATES:
        yield "D", {"dynamic_threshold": value}, f"tauD{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-hparams", type=Path, required=True)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--tuning-seed", type=int, default=42)
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("results/raw_manifests"),
    )
    parser.add_argument("--experiment-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size != FORMAL_EFFECTIVE_BATCH_SIZE:
        parser.error(
            "Formal AE graph tuning requires effective batch size "
            f"{FORMAL_EFFECTIVE_BATCH_SIZE}."
        )

    datasets = parse_csv(args.datasets)
    invalid = sorted(set(datasets) - set(DATASETS))
    if invalid:
        raise ValueError(f"Unsupported formal datasets: {invalid}")
    source_commit, source_branch = git_source()
    selected, selection_meta = load_selected(
        args.selected_hparams,
        model_configs=CURRENT_MODEL_CONFIGS,
    )
    command_args = SimpleNamespace(
        source_git_commit=source_commit,
        source_git_branch=source_branch,
        selection_source_experiment_id=selection_meta["selection_source_experiment_id"],
        selection_protocol_version=selection_meta["selection_protocol_version"],
        graph_selection_source_experiment_id=None,
        graph_selection_protocol_version=None,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        num_layers=args.num_layers,
    )
    experiment_id = args.experiment_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    item_base = dict(
        model="LAGTCN", gnn="gcn", temporal="patch_transformer",
        source="project", role="graph_tuning",
    )
    runs = []
    for dataset in datasets:
        model_hp = selected_hp(selected, dataset, "LAGTCN")
        for graph, graph_hp, variant in candidates(dataset):
            item = {**item_base, "graph": graph, "graph_source": graph}
            hp = {**model_hp, "variant": variant}
            label, cmd = common_command(
                command_args,
                dataset,
                args.tuning_seed,
                item,
                hp,
                experiment_id,
                tuning=True,
                graph_hp=graph_hp,
                stage_override=STAGE,
                namespace_override=NAMESPACE,
            )
            runs.append({
                **item,
                "dataset": dataset,
                "seed": args.tuning_seed,
                "horizon": 24,
                "run_label": label,
                "model_hyperparameters": model_hp,
                "graph_hyperparameters": graph_hp,
                "selection_partition": "validation",
                "cmd": cmd,
                "experiment_id": experiment_id,
                "source_git_commit": source_commit,
                "source_git_branch": source_branch,
                "graph_policy": FINAL_GRAPH_SOURCE_POLICY,
                "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
                "runtime_git_binding": "provenance_only",
                **selection_meta,
                **formal_batch_protocol(
                    dataset,
                    "LAGTCN",
                    int(model_hp["hidden_dim"]),
                    args.batch_size,
                ),
            })

    expected = sum(sum(1 for _ in candidates(dataset)) for dataset in datasets)
    if len(runs) != expected:
        raise AssertionError(f"Built {len(runs)} graph tuning runs, expected {expected}.")
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    path = args.manifest_dir / f"graph_tuning_{experiment_id}.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for run in runs:
            handle.write(json.dumps(run, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({
        "manifest": str(path),
        "run_count": len(runs),
        "datasets": datasets,
        "tuning_seed": args.tuning_seed,
        "stage": STAGE,
        "graph_policy": FINAL_GRAPH_SOURCE_POLICY,
        "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
        "source_git_commit": source_commit,
        "source_git_branch": source_branch,
        **selection_meta,
        "runtime_git_binding": "provenance_only",
    }, indent=2))


if __name__ == "__main__":
    main()
