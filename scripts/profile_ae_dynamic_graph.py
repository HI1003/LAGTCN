#!/usr/bin/env python3
"""GPU admission profile for 158-node sample-wise LAGTCN dynamic graphs."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from data_loader import LoadDatasetLoader
from graph_sparsity import (
    FINAL_GRAPH_SOURCE_POLICY,
    GRAPH_DESIGN_PROTOCOL_VERSION,
    build_threshold_similarity_adj,
    compute_similarity_numpy,
    graph_edge_diagnostics,
)
from models_additional_baselines import LAGTCNBaseline

DATASET = "GEFCom2017FinalMatch_4level"


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def median_ms(samples):
    return float(np.median(np.asarray(samples, dtype=np.float64)) * 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--static-threshold", type=float, default=0.9)
    parser.add_argument("--adaptive-top-k", type=int, default=8)
    parser.add_argument("--dynamic-threshold", type=float, default=0.9)
    parser.add_argument(
        "--skip-backward-check",
        action="store_true",
        help="Skip the one-batch finite-gradient training-path admission check.",
    )
    parser.add_argument(
        "--timing-context",
        choices=("formal_idle", "shared_gpu_background_load_provisional"),
        default="formal_idle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dynamic_graph_admission.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Admission profile requires an idle CUDA GPU.")

    dataset_dir = args.data_root / DATASET
    loader = LoadDatasetLoader(
        str(dataset_dir),
        input_dim=1,
        adj_file="adj_hierarchy.npy",
        value_file="node_values.npy",
        raw_csv_file="load_final_filled.csv",
    )
    signal = loader.get_dataset(num_timesteps_in=168, num_timesteps_out=24)
    train, _, _, _, _ = loader.split_dataset_by_target_timestamp(signal)
    batch_size = min(args.batch_size, len(train.features))
    x = torch.as_tensor(
        np.stack(train.features[:batch_size]), dtype=torch.float32, device=device
    )
    hierarchy = np.load(dataset_dir / "adj_hierarchy.npy").astype(np.float32)
    train_end = int(loader.X.shape[2] * 0.8)
    static_sim = compute_similarity_numpy(
        loader.X[:, 0, :train_end].detach().cpu().numpy().T,
        sim_type="cosine",
        use_abs=True,
    )

    profiles = []
    for graph_mode in ("H+D", "H+S+A+D"):
        config = {
            "graph_mode": graph_mode,
            "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
            "sim_type": "cosine",
            "adaptive_sim_type": "cosine",
            "dynamic_sim_type": "cosine",
            "static_threshold": float(args.static_threshold),
            "adaptive_top_k": int(args.adaptive_top_k),
            "dynamic_threshold": float(args.dynamic_threshold),
            "adaptive_emb_dim": 16,
            "include_self_loops": True,
            "native_top_k": 5,
        }
        static = None
        if "S" in graph_mode.split("+"):
            static = build_threshold_similarity_adj(
                static_sim,
                threshold=config["static_threshold"],
            )
        model = LAGTCNBaseline(
            node_num=loader.num_total_nodes,
            input_dim=1,
            hidden_dim=args.hidden_dim,
            output_dim=24,
            num_layers=args.num_layers,
            global_min=loader.global_min,
            global_max=loader.global_max,
            num_timesteps_in=168,
        ).to(device)
        model.set_norm_params(loader.norm_params)
        model.set_graph_config(config, base_adj=hierarchy)
        model.set_static_graph_sources(hierarchy_adj=hierarchy, similarity_adj=static)
        model.set_hierarchy_metadata(
            loader.sum_matrix,
            middle_levels=(loader.hierarchy_info or {}).get("middle_levels", []),
            bottom_start_idx=loader.bottom_start_idx,
        )
        edge_index = torch.as_tensor(loader.edges, dtype=torch.long, device=device)
        model.eval()

        with torch.no_grad():
            for _ in range(args.warmup):
                model._compute_samplewise_dynamic_adj(x)
                model(x, edge_index)
            synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            dynamic_times, forward_times = [], []
            first_dynamic = None
            for _ in range(args.repeats):
                synchronize(device)
                started = time.perf_counter()
                dynamic = model._compute_samplewise_dynamic_adj(x)
                synchronize(device)
                dynamic_times.append(time.perf_counter() - started)
                if first_dynamic is None:
                    first_dynamic = dynamic.detach().cpu().numpy()

                synchronize(device)
                started = time.perf_counter()
                prediction = model(x, edge_index)
                synchronize(device)
                forward_times.append(time.perf_counter() - started)
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError(f"{graph_mode}: non-finite forward output.")
            forward_peak_memory_mb = float(
                torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            )

        backward_diagnostic = {"enabled": not args.skip_backward_check}
        if not args.skip_backward_check:
            model.train()
            model.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            backward_started = time.perf_counter()
            training_prediction = model(x, edge_index)
            if not bool(torch.isfinite(training_prediction).all()):
                raise FloatingPointError(
                    f"{graph_mode}: non-finite training-path forward output."
                )
            training_loss = training_prediction.abs().mean()
            training_loss.backward()
            synchronize(device)
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ]
            nonfinite_gradients = sum(
                not bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            if not gradients or nonfinite_gradients:
                raise FloatingPointError(
                    f"{graph_mode}: gradients present={len(gradients)}, "
                    f"nonfinite={nonfinite_gradients}."
                )
            backward_diagnostic.update({
                "loss": float(training_loss.detach().cpu()),
                "parameters_with_gradients": len(gradients),
                "nonfinite_gradient_tensors": int(nonfinite_gradients),
                "forward_backward_ms_per_batch": float(
                    (time.perf_counter() - backward_started) * 1000.0
                ),
                "peak_gpu_memory_mb": float(
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                ),
            })
            model.zero_grad(set_to_none=True)

        dynamic_diags = [
            graph_edge_diagnostics(matrix, hierarchy_adj=hierarchy)
            for matrix in first_dynamic
        ]
        overlaps = [diag["hierarchy_overlap_edge_count"] for diag in dynamic_diags]
        profiles.append({
            "graph_mode": graph_mode,
            "batch_size": batch_size,
            "static_threshold": config["static_threshold"] if static is not None else None,
            "adaptive_top_k": config["adaptive_top_k"] if "A" in graph_mode else None,
            "dynamic_threshold": config["dynamic_threshold"],
            "dynamic_graph_diagnostics_first_sample": dynamic_diags[0],
            "dynamic_graph_diagnostics_batch": {
                "samples_checked": len(dynamic_diags),
                "undirected_edge_count_min": min(
                    diag["undirected_edge_count"] for diag in dynamic_diags
                ),
                "undirected_edge_count_max": max(
                    diag["undirected_edge_count"] for diag in dynamic_diags
                ),
                "degree_min_across_samples": min(
                    diag["degree_min"] for diag in dynamic_diags
                ),
                "degree_mean_across_samples": float(np.mean([
                    diag["degree_mean"] for diag in dynamic_diags
                ])),
                "degree_max_across_samples": max(
                    diag["degree_max"] for diag in dynamic_diags
                ),
                "isolated_node_count_max": max(
                    diag["isolated_node_count"] for diag in dynamic_diags
                ),
                "hierarchy_overlap_edge_count_min": min(overlaps),
                "hierarchy_overlap_edge_count_max": max(overlaps),
            },
            "dynamic_build_median_ms_per_batch": median_ms(dynamic_times),
            "dynamic_build_median_ms_per_sample": median_ms(dynamic_times) / batch_size,
            "full_forward_median_ms_per_batch": median_ms(forward_times),
            "full_forward_median_ms_per_sample": median_ms(forward_times) / batch_size,
            "forward_peak_gpu_memory_mb": forward_peak_memory_mb,
            "backward_check": backward_diagnostic,
        })

    payload = {
        "profile_version": "ae_dynamic_graph_admission_v4_training_path",
        "dataset": DATASET,
        "num_nodes": int(loader.num_total_nodes),
        "input_length": 168,
        "output_length": 24,
        "graph_policy": FINAL_GRAPH_SOURCE_POLICY,
        "graph_design_protocol_version": GRAPH_DESIGN_PROTOCOL_VERSION,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "timing_context": args.timing_context,
        "timing_is_formal_idle_measurement": args.timing_context == "formal_idle",
        "backward_check_enabled": not args.skip_backward_check,
        "profiles": profiles,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
