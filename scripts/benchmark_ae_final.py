#!/usr/bin/env python3
"""Batch-1 end-to-end deployment benchmark for the final AE model set."""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for search in (ROOT / "code", ROOT / "scripts"):
    if str(search) not in sys.path:
        sys.path.insert(0, str(search))

from graph_sparsity import GRAPH_DESIGN_PROTOCOL_VERSION
from data_loader import TARGET_TIMESTAMP_SPLIT_VERSION
import reconcile_ae
from postprocess_ae_mint_shrink import _build_model, _dataset_bundle
from train_eval import load_best_model_strict
from build_ae_final_manifest import DATASETS, SEEDS

VERSION = "ae_batch1_end_to_end_v2"
MODELS = {
    "DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER",
    "DCRNN", "MTGNN", "LAGTCN",
}



def current_git_provenance() -> dict:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run_git("rev-parse", "HEAD")
        branch = run_git("branch", "--show-current") or "DETACHED"
        tracked_status = run_git("status", "--porcelain", "--untracked-files=no")
        untracked_source = run_git(
            "ls-files", "--others", "--exclude-standard", "--", "code", "scripts"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot audit benchmark source provenance: {exc}") from exc
    return {
        "commit": commit,
        "branch": branch,
        "tracked_dirty": bool(tracked_status),
        "tracked_status": tracked_status.splitlines(),
        "untracked_source": bool(untracked_source),
        "untracked_source_files": untracked_source.splitlines(),
    }


def discover(
    root: Path,
    experiment_id: str | None = None,
    *,
    require_complete: bool = False,
):
    matched = []
    for path in sorted(root.rglob("config.json")):
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if config.get("experiment_stage") != "ae_final_main_v1":
            continue
        model = str(config.get("model_name", "")).upper()
        if model not in MODELS:
            continue
        if model == "LAGTCN" and config.get("graph_mode") != "H":
            continue
        matched.append((path.parent, config))
    experiment_ids = sorted({
        str(config.get("experiment_id"))
        for _, config in matched if config.get("experiment_id")
    })
    if experiment_id is None:
        if len(experiment_ids) != 1:
            raise RuntimeError(
                "Formal benchmark requires exactly one experiment_id; "
                f"found {experiment_ids}. Pass --experiment-id explicitly."
            )
        experiment_id = experiment_ids[0]
    selected = [
        (run_dir, config) for run_dir, config in matched
        if str(config.get("experiment_id")) == experiment_id
    ]
    if not selected:
        raise RuntimeError(f"No benchmark runs found for experiment_id={experiment_id!r}.")
    keys = [
        (str(c.get("dataset")), str(c.get("model_name")), int(c.get("seed")))
        for _, c in selected
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            f"Duplicate benchmark keys detected within experiment_id={experiment_id!r}."
        )
    expected = {
        (dataset, model, seed)
        for dataset in DATASETS for model in MODELS for seed in SEEDS
    }
    actual_keys = set(keys)
    unexpected = sorted(actual_keys - expected)
    if unexpected:
        raise RuntimeError(f"Unexpected benchmark keys: {unexpected}.")
    if require_complete and actual_keys != expected:
        raise RuntimeError(
            "Benchmark batch is not the complete 63-run phase-1 matrix."
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
            f"Benchmark batch spans multiple or missing selections {sorted(selections)}."
        )
    if any(
        c.get("split_protocol_version") != TARGET_TIMESTAMP_SPLIT_VERSION
        for _, c in selected
    ):
        raise RuntimeError("Benchmark batch uses a legacy or missing split protocol.")

    if any(
        str(c.get("model_name", "")).upper() == "LAGTCN"
        and c.get("graph_design_protocol_version") != GRAPH_DESIGN_PROTOCOL_VERSION
        for _, c in selected
    ):
        raise RuntimeError("Benchmark batch uses a legacy graph-design protocol.")

    benchmark_source = current_git_provenance()
    for _, config in selected:
        config["_benchmark_git_commit"] = benchmark_source["commit"]
        config["_benchmark_git_branch"] = benchmark_source["branch"]
        config["_benchmark_source_git_commit"] = config.get("source_git_commit")
        config["_benchmark_source_git_branch"] = config.get("source_git_branch")
    return experiment_id, selected


def benchmark_run(run_dir, config, bundle, device, warmup, repeats):
    model = _build_model(config, bundle, device=device)
    checkpoint = run_dir / "best_model.pth"
    checkpoint_meta = load_best_model_strict(model, str(checkpoint), config, device)
    model.eval()
    features = bundle.test.features
    if not features:
        raise RuntimeError(f"{run_dir}: empty test partition.")
    S = bundle.loader.sum_matrix
    bottom_start = int(bundle.loader.bottom_start_idx)

    def request(request_index):
        feature = features[request_index % len(features)]
        x = torch.as_tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        prediction = model(x, bundle.edge_index)
        prediction_np = prediction.detach().cpu().numpy()
        reconciled, _ = reconcile_ae.apply_reconciliation_ae(
            "bu", prediction_np, S, bottom_start_idx=bottom_start
        )
        if not np.isfinite(reconciled).all():
            raise FloatingPointError("Non-finite operational BU output.")

    with torch.no_grad():
        for index in range(warmup):
            request(index)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        latencies = []
        for index in range(repeats):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            request(index + warmup)
            torch.cuda.synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)

    values = np.asarray(latencies, dtype=np.float64)
    q25, median, q75, p95 = np.percentile(values, [25, 50, 75, 95])
    result = {
        "benchmark_version": VERSION,
        "dataset": config["dataset"],
        "experiment_id": config.get("experiment_id"),
        "model_name": config["model_name"],
        "graph_mode": config.get("graph_mode"),
        "seed": int(config["seed"]),
        "input_length": int(config["num_timesteps_in"]),
        "output_length": int(config["num_timesteps_out"]),
        "batch_size": 1,
        "warmup_requests": int(warmup),
        "measured_requests": int(repeats),
        "timed_scope": "host_input_to_device+model_forward+inverse_transform+host_BU",
        "excluded_scope": "checkpoint_load+dataset_load+disk_IO",
        "latency_ms_median": float(median),
        "latency_ms_iqr": float(q75 - q25),
        "latency_ms_q25": float(q25),
        "latency_ms_q75": float(q75),
        "latency_ms_p95": float(p95),
        "peak_gpu_memory_mb": float(
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        ),
        "gpu_name": torch.cuda.get_device_name(device),
        "checkpoint_sha256": checkpoint_meta["checkpoint_sha256"],
        "benchmark_git_commit": config["_benchmark_git_commit"],
        "benchmark_git_branch": config["_benchmark_git_branch"],
        "source_git_commit": config["_benchmark_source_git_commit"],
        "source_git_branch": config["_benchmark_source_git_branch"],
    }
    output = run_dir / "batch1_benchmark.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("Data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--experiment-id")
    parser.add_argument("--allow-partial-matrix", action="store_true")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/final_checkpoint_benchmark.csv"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Formal deployment benchmark requires one CUDA GPU.")
    experiment_id, runs = discover(
        args.runs_root,
        args.experiment_id,
        require_complete=args.limit is None and not args.allow_partial_matrix,
    )
    if (
        args.limit is None
        and not args.allow_partial_matrix
        and len(runs) != len(DATASETS) * len(SEEDS) * len(MODELS)
    ):
        raise RuntimeError(
            f"Benchmark matrix {experiment_id!r} contains {len(runs)} configs; "
            "expected 63 phase-1 checkpoint replays. "
            "Use --allow-partial-matrix only for diagnostics."
        )
    if args.limit is not None:
        runs = runs[:max(0, args.limit)]

    bundle_cache = {}
    rows = []
    errors = []
    for run_dir, config in runs:
        try:
            dataset = str(config["dataset"])
            if dataset not in bundle_cache:
                bundle_cache[dataset] = _dataset_bundle(dataset, config, device=device)
            rows.append(
                benchmark_run(
                    run_dir, config, bundle_cache[dataset], device,
                    warmup=args.warmup, repeats=args.repeats,
                )
            )
        except Exception as exc:
            errors.append({"run_dir": str(run_dir), "error": repr(exc)})
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.summary, index=False)
    error_path = args.summary.with_suffix(".errors.json")
    error_path.write_text(
        json.dumps(errors, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "benchmark_version": VERSION,
        "experiment_id": experiment_id,
        "discovered": len(runs),
        "includes_deephgnn": False,
        "completed": len(rows),
        "errors": len(errors),
        "summary": str(args.summary),
        "error_log": str(error_path),
    }, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
