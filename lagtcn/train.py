"""Train, validate, and evaluate LAGTCN."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import HierarchicalLoadDataset, WindowDataset
from .graphs import (
    FINAL_GRAPH_SOURCE_POLICY,
    build_threshold_similarity_adj,
    compute_similarity_numpy,
    graph_sources,
    normalize_graph_mode,
)
from .metrics import coherence_metrics, forecast_metrics
from .model import LAGTCN


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LAGTCN model")
    parser.add_argument("--data-root", type=Path, default=Path("Data"))
    parser.add_argument("--dataset", default="GEFCom2012_2level")
    parser.add_argument("--value-file", default="node_values.npy")
    parser.add_argument("--raw-csv-file", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)

    parser.add_argument("--num-timesteps-in", type=int, default=168)
    parser.add_argument("--num-timesteps-out", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--patch-len", type=int, default=8)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--hop-order", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--graph-mode", default="H")
    parser.add_argument("--sim-type", choices=("cosine", "pearson"), default="cosine")
    parser.add_argument("--static-threshold", type=float, default=None)
    parser.add_argument("--adaptive-top-k", type=int, default=None)
    parser.add_argument("--dynamic-threshold", type=float, default=None)

    parser.add_argument(
        "--decoder-mode",
        choices=("persistence_residual", "seasonal_residual", "direct"),
        default="persistence_residual",
    )
    parser.add_argument(
        "--residual-scale-mode",
        choices=("unit", "fixed", "learnable"),
        default="unit",
    )
    parser.add_argument("--residual-scale-init", type=float, default=1.0)
    parser.add_argument("--seasonal-lag", type=int, default=24)
    parser.add_argument("--no-level-awareness", action="store_true")
    parser.add_argument("--no-coevolution", action="store_true")
    parser.add_argument("--uniform-source-fusion", action="store_true")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    for name in (
        "num_timesteps_in",
        "num_timesteps_out",
        "hidden_dim",
        "num_layers",
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "patience",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must lie in [0, 1)")
    return args


def _resolve_device(value: str) -> torch.device:
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_graph_args(args: argparse.Namespace) -> tuple[str, tuple[str, ...]]:
    mode = normalize_graph_mode(args.graph_mode)
    sources = graph_sources(mode)
    required = {
        "similarity": ("--static-threshold", args.static_threshold),
        "adaptive": ("--adaptive-top-k", args.adaptive_top_k),
        "dynamic": ("--dynamic-threshold", args.dynamic_threshold),
    }
    missing = [flag for source, (flag, value) in required.items() if source in sources and value is None]
    if missing:
        raise ValueError(f"graph mode {mode} requires {' '.join(missing)}")
    for value, name in (
        (args.static_threshold, "static_threshold"),
        (args.dynamic_threshold, "dynamic_threshold"),
    ):
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if args.adaptive_top_k is not None and args.adaptive_top_k < 0:
        raise ValueError("adaptive_top_k must be nonnegative")
    return mode, sources


def _make_loader(dataset: WindowDataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _evaluate_loss(
    model: LAGTCN,
    dataset: WindowDataset,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for features, targets in _make_loader(dataset, batch_size, shuffle=False):
            features = features.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            predictions = model(features)
            targets_original = model.transform_target(targets)
            loss = criterion(predictions, targets_original)
            total_loss += float(loss.item()) * features.shape[0]
            total_count += features.shape[0]
    return total_loss / total_count


def _train(
    model: LAGTCN,
    train_dataset: WindowDataset,
    validation_dataset: WindowDataset,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
) -> dict:
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )
    criterion = nn.SmoothL1Loss()
    train_loader = _make_loader(train_dataset, args.batch_size, shuffle=True)
    checkpoint_path = output_dir / "best_model.pt"
    best_validation = float("inf")
    stale_epochs = 0
    history = {"train_loss": [], "validation_loss": []}
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        sample_count = 0
        for batch_index, (features, targets) in enumerate(train_loader, start=1):
            features = features.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            predictions = model(features)
            targets_original = model.transform_target(targets)
            loss = criterion(predictions, targets_original)
            (loss / args.gradient_accumulation_steps).backward()
            if (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item()) * features.shape[0]
            sample_count += features.shape[0]

        train_loss = running_loss / sample_count
        validation_loss = _evaluate_loss(
            model,
            validation_dataset,
            criterion,
            args.batch_size,
            device,
        )
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        logging.info(
            "epoch %d/%d | train %.6f | validation %.6f",
            epoch,
            args.epochs,
            train_loss,
            validation_loss,
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                logging.info("early stopping after epoch %d", epoch)
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    history.update(
        {
            "best_epoch": int(checkpoint["epoch"]),
            "best_validation_loss": float(checkpoint["validation_loss"]),
            "training_seconds": float(time.perf_counter() - start_time),
        }
    )
    return history


def _predict(
    model: LAGTCN,
    dataset: WindowDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for features, targets in _make_loader(dataset, batch_size, shuffle=False):
            features = features.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)
            predictions.append(model(features).cpu().numpy())
            true_values.append(model.transform_target(targets).cpu().numpy())
    return np.concatenate(predictions), np.concatenate(true_values)


def _save_prediction_archive(
    path: Path,
    predictions: np.ndarray,
    true_values: np.ndarray,
    dataset: WindowDataset,
    node_names: list[str],
) -> None:
    np.savez_compressed(
        path,
        predictions=predictions.astype(np.float32),
        true_values=true_values.astype(np.float32),
        target_timestamps=np.asarray(dataset.target_timestamps),
        node_names=np.asarray(node_names),
    )


def _json_dump(payload: dict, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _set_seed(args.seed)
    device = _resolve_device(args.device)
    graph_mode, active_sources = _validate_graph_args(args)

    dataset_dir = args.data_root / args.dataset
    adjacency_file = "adj_HGNN.npy" if graph_mode == "HG" else "adj_hierarchy.npy"
    data = HierarchicalLoadDataset(
        dataset_dir,
        value_file=args.value_file,
        adjacency_file=adjacency_file,
        raw_csv_file=args.raw_csv_file,
    )
    train_data, validation_data, test_data, split_info = data.make_splits(
        args.num_timesteps_in,
        args.num_timesteps_out,
    )

    hierarchy_adjacency = data.adjacency
    similarity_adjacency = None
    if "similarity" in active_sources:
        similarity = compute_similarity_numpy(
            data.training_target_values(),
            similarity_type=args.sim_type,
            use_abs=True,
        )
        similarity_adjacency = build_threshold_similarity_adj(
            similarity,
            threshold=args.static_threshold,
            include_self_loops=True,
        )

    model = LAGTCN(
        node_num=data.num_nodes,
        input_dim=data.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_timesteps_out,
        num_layers=args.num_layers,
        global_min=data.global_min,
        global_max=data.global_max,
        num_timesteps_in=args.num_timesteps_in,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        dropout=args.dropout,
        hop_order=args.hop_order,
        use_level_awareness=not args.no_level_awareness,
        use_coevolution=not args.no_coevolution,
        learn_source_fusion=not args.uniform_source_fusion,
        decoder_mode=args.decoder_mode,
        residual_scale_mode=args.residual_scale_mode,
        residual_scale_init=args.residual_scale_init,
        seasonal_lag=args.seasonal_lag,
    ).to(device)
    model.set_norm_params(data.norm_params)
    model.set_graph_config(
        {
            "graph_mode": graph_mode,
            "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
            "sim_type": args.sim_type,
            "static_threshold": args.static_threshold,
            "adaptive_top_k": args.adaptive_top_k,
            "dynamic_threshold": args.dynamic_threshold,
            "include_self_loops": True,
        }
    )
    model.set_static_graph_sources(
        hierarchy_adj=hierarchy_adjacency,
        similarity_adj=similarity_adjacency,
    )
    model.set_hierarchy_metadata(
        data.sum_matrix,
        middle_levels=data.middle_levels,
        bottom_start_idx=data.bottom_start_idx,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or dataset_dir / "output" / f"lagtcn_{graph_mode}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "model": "LAGTCN",
        "architecture_version": model.architecture_version,
        "dataset": args.dataset,
        "value_file": args.value_file,
        "raw_csv_file": args.raw_csv_file,
        "input_length": args.num_timesteps_in,
        "output_length": args.num_timesteps_out,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "patch_len": args.patch_len,
        "patch_stride": args.patch_stride,
        "hop_order": args.hop_order,
        "dropout": args.dropout,
        "graph_mode": graph_mode,
        "active_graph_sources": list(active_sources),
        "graph_sparsity_policy": FINAL_GRAPH_SOURCE_POLICY,
        "sim_type": args.sim_type,
        "static_threshold": args.static_threshold,
        "adaptive_top_k": args.adaptive_top_k,
        "dynamic_threshold": args.dynamic_threshold,
        "decoder_mode": args.decoder_mode,
        "residual_scale_mode": args.residual_scale_mode,
        "residual_scale_init": args.residual_scale_init,
        "seasonal_lag": args.seasonal_lag,
        "use_level_awareness": not args.no_level_awareness,
        "use_coevolution": not args.no_coevolution,
        "learn_source_fusion": not args.uniform_source_fusion,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "split": split_info,
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    _json_dump(config, output_dir / "config.json")

    logging.info("device=%s output=%s", device, output_dir)
    training = _train(
        model,
        train_data,
        validation_data,
        args,
        output_dir,
        device,
    )
    _json_dump(training, output_dir / "training.json")

    validation_predictions, validation_true = _predict(
        model, validation_data, args.batch_size, device
    )
    _save_prediction_archive(
        output_dir / "validation_predictions.npz",
        validation_predictions,
        validation_true,
        validation_data,
        data.node_names,
    )
    metrics = {
        "validation": {
            **forecast_metrics(validation_true, validation_predictions),
            **coherence_metrics(
                validation_predictions, data.sum_matrix, data.bottom_start_idx
            ),
        }
    }

    if not args.validation_only:
        test_predictions, test_true = _predict(model, test_data, args.batch_size, device)
        _save_prediction_archive(
            output_dir / "base_predictions.npz",
            test_predictions,
            test_true,
            test_data,
            data.node_names,
        )
        metrics["test"] = {
            **forecast_metrics(test_true, test_predictions),
            **coherence_metrics(test_predictions, data.sum_matrix, data.bottom_start_idx),
        }
    _json_dump(metrics, output_dir / "metrics.json")
    _json_dump(
        {"graph_source_gates": model.get_graph_source_gates()},
        output_dir / "model_metadata.json",
    )
    logging.info("completed: %s", output_dir)


if __name__ == "__main__":
    main()
