import hashlib
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from lagtcn.core.protocol import is_formal_ae_stage
from lagtcn.core import scaled_error as ae_mase
from lagtcn.core.metrics import compute_mase
from lagtcn.core.naming import PRED_FILENAME, TRUE_FILENAME, artifact_filename


def compute_coherency_loss(
    y_pred: torch.Tensor,
    sum_matrix: torch.Tensor,
    bottom_start_idx: int,
) -> torch.Tensor:
    """Differentiable coherency violation loss.

    Measures how much the predictions violate the hierarchical constraint:
        y_hat = S @ y_hat_bottom

    Args:
        y_pred: [S, N, H] predictions in original scale
        sum_matrix: [N, B] aggregation matrix
        bottom_start_idx: index where bottom nodes start

    Returns:
        Scalar loss: mean absolute coherency violation
    """
    B = sum_matrix.shape[1]
    bottom_preds = y_pred[:, bottom_start_idx:bottom_start_idx + B, :]  # [S, B, H]
    coherent_preds = torch.matmul(sum_matrix, bottom_preds)  # [S, N, H]
    violation = y_pred - coherent_preds
    return violation.abs().mean()


class _TemporalSignalDataset(Dataset):
    def __init__(self, signal):
        self.features = signal.features
        self.targets = signal.targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.as_tensor(self.features[idx], dtype=torch.float32)
        y = torch.as_tensor(self.targets[idx], dtype=torch.float32)
        return x, y


def _make_loader(signal, batch_size, shuffle):
    dataset = _TemporalSignalDataset(signal)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _align_target(y_true, y_pred):
    if y_true.dim() == 2:
        y_true = y_true.unsqueeze(-1)
    if tuple(y_true.shape) != tuple(y_pred.shape):
        raise ValueError(
            f"Target/prediction shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}"
        )
    return y_true


def _get_alpha_value(model):
    if not hasattr(model, "alpha"):
        return None
    alpha = model.alpha
    if isinstance(alpha, torch.nn.Parameter):
        return torch.sigmoid(alpha).detach().cpu().item()
    return float(alpha)



CONFIG_FINGERPRINT_PROTOCOL_VERSION = "training_protocol_v2_git_provenance_excluded"
LEGACY_CONFIG_FINGERPRINT_PROTOCOL_VERSION = "training_protocol_v1_git_bound"

_CONFIG_FINGERPRINT_KEYS = (
        "dataset", "feature_set", "model_name", "graph_mode", "sim_type",
        "experiment_stage", "experiment_id", "paper_scope", "run_label",
        "selection_source_experiment_id", "selection_protocol_version",
        "graph_selection_source_experiment_id", "graph_selection_protocol_version",
        "graph_sparsity_policy", "graph_protocol_version",
        "graph_design_protocol_version",
        "static_threshold", "adaptive_top_k", "dynamic_threshold",
        "stgnn_graph_source", "native_top_k", "gnn_type", "temporal_type",
        "model_architecture_version", "dcrnn_filter_type",
        "dcrnn_max_diffusion_step", "dcrnn_cl_decay_steps",
        "dcrnn_scheduled_sampling", "mtgnn_training_horizon_scope", "training_loss_space",
        "st_mode", "num_timesteps_in", "num_timesteps_out", "node_num",
        "input_dim", "hidden_dim", "output_dim", "num_layers", "seed",
        "lr", "batch_size", "gradient_accumulation_steps", "effective_batch_size",
        "patience", "coherency_lambda", "include_self_loops",
        "adaptive_sim_type", "dynamic_sim_type",
        "deephgnn_hierarchical_loss_weight", "spectgnn_alpha",
        "spectgnn_degree", "spectgnn_modes", "spectgnn_trend_window",
        "lagtcn_ablation", "lagtcn_decoder_mode", "lagtcn_residual_scale_mode",
        "lagtcn_residual_scale_init", "lagtcn_seasonal_lag",
        "split_protocol_version", "validation_only",
)


def _fingerprint_config_keys(config: dict, keys: tuple[str, ...]) -> str:
    payload = {key: config.get(key) for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _config_fingerprint(config: dict) -> str:
    """Fingerprint fields that can change training, excluding Git provenance."""
    return _fingerprint_config_keys(config, _CONFIG_FINGERPRINT_KEYS)


def _legacy_config_fingerprint(config: dict) -> str:
    """Reproduce the pre-v2 fingerprint for controlled compatibility checks."""
    return _fingerprint_config_keys(
        config,
        _CONFIG_FINGERPRINT_KEYS + ("source_git_commit", "source_git_branch"),
    )


def _match_config_fingerprint(config: dict, checkpoint_fingerprint: str | None) -> str | None:
    """Return the matching protocol without weakening training-field checks."""
    if checkpoint_fingerprint == _config_fingerprint(config):
        return CONFIG_FINGERPRINT_PROTOCOL_VERSION
    if checkpoint_fingerprint == _legacy_config_fingerprint(config):
        return LEGACY_CONFIG_FINGERPRINT_PROTOCOL_VERSION
    return None



def _sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_dump(payload: dict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
    os.replace(tmp_path, path)


def _finite_summary(value) -> dict:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": [int(v) for v in array.shape],
        "dtype": str(array.dtype),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(array.size - finite.sum()),
        "finite_min": float(finite_values.min()) if finite_values.size else None,
        "finite_max": float(finite_values.max()) if finite_values.size else None,
    }


def assert_finite(value, name: str, stage: str, config: dict | None = None) -> None:
    """Fail a run immediately and persist a structured non-finite diagnostic."""
    is_finite = bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else bool(
        np.isfinite(np.asarray(value)).all()
    )
    if is_finite:
        return
    summary = _finite_summary(value)
    payload = {
        "failure_protocol_version": "finite_failfast_v1",
        "failure_type": "nonfinite",
        "stage": str(stage),
        "value_name": str(name),
        "summary": summary,
    }
    if config:
        payload["model_name"] = config.get("model_name")
        payload["dataset"] = config.get("dataset")
        payload["seed"] = config.get("seed")
        output_dir = config.get("output_dir")
        if output_dir:
            _atomic_json_dump(payload, Path(output_dir) / "failure.json")
    raise FloatingPointError(
        f"Non-finite {name} detected during {stage}: {summary}"
    )


def _compute_model_loss(model, y_pred, y_true, criterion):
    if hasattr(model, "compute_training_loss"):
        return model.compute_training_loss(y_pred, y_true, criterion)
    return criterion(y_pred, y_true)


def _uses_normalized_log_loss(config: dict | None) -> bool:
    return bool(config) and config.get("training_loss_space", "original") == "normalized_log"


def _target_for_loss(model, normalized_target, prediction, config):
    if _uses_normalized_log_loss(config):
        return _align_target(normalized_target, prediction)
    return _align_target(model.transform_target(normalized_target), prediction)


def _compute_loss_for_space(model, prediction, target, criterion, config):
    if _uses_normalized_log_loss(config):
        normalized_hook = getattr(model, "compute_normalized_training_loss", None)
        if normalized_hook is not None:
            return normalized_hook(prediction, target, criterion)
        return criterion(prediction, target)
    return _compute_model_loss(model, prediction, target, criterion)


def _forward_normalized(model, x, edge_index):
    method = getattr(model, "forward_normalized", None)
    if method is None:
        raise TypeError(
            f"{type(model).__name__} does not implement normalized-log training output."
        )
    return method(x, edge_index)


def _forward_training_model(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    normalized_target: torch.Tensor,
    batches_seen: int,
    config: dict | None = None,
) -> torch.Tensor:
    """Use paper-specific training hooks without changing evaluation forward."""
    if _uses_normalized_log_loss(config):
        return _forward_normalized(model, x, edge_index)
    if hasattr(model, "set_training_progress"):
        model.set_training_progress(int(batches_seen))
    if hasattr(model, "forward_train"):
        return model.forward_train(
            x,
            edge_index,
            teacher_targets=normalized_target,
            batches_seen=int(batches_seen),
        )
    return model(x, edge_index)


def load_best_model_strict(model, checkpoint_path: str, config: dict, device) -> dict:
    """Load a best state only after hash and config-fingerprint validation."""
    path = Path(checkpoint_path)
    metadata_path = Path(f"{checkpoint_path}.metadata.json")
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Best checkpoint metadata was not created: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    checkpoint_fingerprint = metadata.get("config_fingerprint")
    matched_protocol = _match_config_fingerprint(config, checkpoint_fingerprint)
    if matched_protocol is None:
        raise ValueError(
            "Best checkpoint/config mismatch: "
            f"checkpoint={checkpoint_fingerprint}, "
            f"expected={_config_fingerprint(config)}, "
            f"legacy_expected={_legacy_config_fingerprint(config)}"
        )
    if matched_protocol == LEGACY_CONFIG_FINGERPRINT_PROTOCOL_VERSION:
        logging.warning(
            "Loading a legacy Git-bound best-checkpoint fingerprint; all training-protocol "
            "fields and the recorded Git provenance match."
        )
    actual_hash = _sha256_file(path)
    if metadata.get("checkpoint_sha256") != actual_hash:
        raise ValueError(
            f"Best checkpoint hash mismatch: metadata={metadata.get('checkpoint_sha256')}, "
            f"actual={actual_hash}"
        )
    state_dict = _torch_load(path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return metadata




def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _capture_rng_state(use_cuda: bool) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if use_cuda:
        state["cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None, use_cuda: bool) -> None:
    if not isinstance(state, dict):
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch_state = state["torch"]
            if isinstance(torch_state, torch.Tensor):
                torch_state = torch_state.cpu()
            torch.set_rng_state(torch_state)
        if use_cuda and "cuda_all" in state:
            cuda_states = [
                rng_state.cpu() if isinstance(rng_state, torch.Tensor) else rng_state
                for rng_state in state["cuda_all"]
            ]
            torch.cuda.set_rng_state_all(cuda_states)
    except Exception as exc:
        logging.warning("Could not restore checkpoint RNG state: %s", exc)


def _atomic_torch_save(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _save_training_checkpoint(
    path: str,
    model,
    optimizer,
    epoch: int,
    best_val_loss: float,
    patience_counter: int,
    train_losses: list,
    val_losses: list,
    alpha_values: list,
    elapsed_train_time_sec: float,
    train_peak_gpu_mem_mb: float | None,
    batches_seen: int,
    config: dict,
    use_cuda: bool,
) -> None:
    payload = {
        "checkpoint_version": 2,
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": float(best_val_loss),
        "patience_counter": int(patience_counter),
        "train_losses": [float(v) for v in train_losses],
        "val_losses": [float(v) for v in val_losses],
        "alpha_values": [float(v) for v in alpha_values],
        "elapsed_train_time_sec": float(elapsed_train_time_sec),
        "train_peak_gpu_mem_mb": (
            float(train_peak_gpu_mem_mb) if train_peak_gpu_mem_mb is not None else None
        ),
        "batches_seen": int(batches_seen),
        "model_name": config.get("model_name"),
        "lagtcn_graph_source_version": config.get("lagtcn_graph_source_version"),
        "timestamp": config.get("timestamp"),
        "run_label": config.get("run_label"),
        "output_dir": config.get("output_dir"),
        "config_fingerprint": _config_fingerprint(config),
        "config_fingerprint_protocol_version": CONFIG_FINGERPRINT_PROTOCOL_VERSION,
        "best_epoch": config.get("_best_epoch"),
        "stop_reason": config.get("_stop_reason"),
        "rng_state": _capture_rng_state(use_cuda),
    }
    _atomic_torch_save(payload, path)


def _compute_metrics(
    y_true,
    y_pred,
    num_timesteps_in: int = 7,
    epsilon=1e-3,
    mase_scale=None,
    require_mase_scale=False,
):
    """Compute overall/per-lead metrics with optional frozen sMASE-24 scale."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")
    assert_finite(y_true, "metric_target", "metrics")
    assert_finite(y_pred, "metric_prediction", "metrics")

    if y_true.ndim == 2:
        y_true = y_true[..., None]
        y_pred = y_pred[..., None]
    frozen_scale = None if mase_scale is None else np.asarray(mase_scale, dtype=np.float64)
    if require_mase_scale and frozen_scale is None:
        raise ValueError(
            "Formal Applied Energy evaluation requires the frozen training-period "
            "lag-24 sMASE scale; refusing legacy MASE fallback."
        )
    if frozen_scale is not None and frozen_scale.shape != (y_true.shape[1],):
        raise ValueError(
            f"mase_scale shape {frozen_scale.shape} != ({y_true.shape[1]},)"
        )

    def _metric_block(y_t, y_p):
        y_t_flat = y_t.reshape(-1)
        y_p_flat = y_p.reshape(-1)
        mae = np.mean(np.abs(y_t_flat - y_p_flat))
        rmse = np.sqrt(np.mean((y_t_flat - y_p_flat) ** 2))
        denom = np.maximum(np.abs(y_t_flat), epsilon)
        mape = np.mean(np.abs((y_t_flat - y_p_flat) / denom)) * 100
        wape = np.sum(np.abs(y_t_flat - y_p_flat)) / np.maximum(np.sum(np.abs(y_t_flat)), epsilon) * 100
        if frozen_scale is None:
            mase_value = compute_mase(
                y_t, y_p, num_timesteps_in=num_timesteps_in
            )
            n_excluded = 0
        else:
            per_node = ae_mase.compute_mase_per_node(y_t, y_p, frozen_scale)
            summary = ae_mase.macro_average_mase(
                per_node, list(range(y_t.shape[1]))
            )
            mase_value = summary["mase"]
            n_excluded = summary["n_excluded"]
        return {
            "MAE": float(mae),
            "RMSE": float(rmse),
            "MAPE": float(mape),
            "WAPE": float(wape),
            "MASE": float(mase_value) if mase_value is not None else np.nan,
            "MASE_n_excluded": int(n_excluded),
        }

    metrics = _metric_block(y_true, y_pred)
    horizon_count = y_true.shape[-1]
    if horizon_count > 1:
        for h in range(horizon_count):
            per_h = _metric_block(y_true[:, :, h:h + 1], y_pred[:, :, h:h + 1])
            for key, value in per_h.items():
                metrics[f"h{h + 1}_{key}"] = value
    if frozen_scale is not None:
        metrics.update({
            "MASE_version": ae_mase.MASE_VERSION,
            "MASE_label": ae_mase.MASE_LABEL,
            "MASE_seasonal_period": ae_mase.MASE_SEASONAL_PERIOD,
            "MASE_scale_reference": "training_period_seasonal_naive_mae",
        })
    else:
        metrics["MASE_version"] = "legacy_test_window_one_step_diagnostic"
    return metrics


def train_model(model, train_dataset, val_dataset, static_edge_index, optimizer, criterion, config, device):
    epochs = int(config.get("epochs", 1))
    patience = int(config.get("patience", 10))
    batch_size = int(config.get("batch_size", 32))
    gradient_accumulation_steps = int(config.get("gradient_accumulation_steps", 1))
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    effective_batch_size = batch_size * gradient_accumulation_steps
    config["effective_batch_size"] = effective_batch_size
    output_dir = config.get("output_dir", ".")
    model_name = config.get("model_name", "model")
    timestamp = config.get("timestamp", "run")
    checkpoint_every = int(config.get("checkpoint_every_epochs", 1) or 0)
    checkpoint_path = os.path.join(output_dir, artifact_filename("last_checkpoint"))
    resume_checkpoint = config.get("resume_checkpoint")

    # Coherency loss settings
    coherency_lambda = float(config.get("coherency_lambda", 0.0))
    sum_matrix_tensor = None
    bottom_start_idx = None
    if coherency_lambda > 0:
        sum_matrix_np = config.get("_sum_matrix")
        if sum_matrix_np is not None:
            sum_matrix_tensor = torch.tensor(sum_matrix_np, dtype=torch.float32).to(device)
            bottom_start_idx = int(config.get("bottom_start_idx", 0))
            logging.info("Coherency loss enabled: lambda=%.4f", coherency_lambda)
        else:
            logging.warning("coherency_lambda > 0 but no sum_matrix provided; disabling coherency loss.")
            coherency_lambda = 0.0

    train_loader = _make_loader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = _make_loader(val_dataset, batch_size=batch_size, shuffle=False)
    optimizer_steps_per_epoch = (
        len(train_loader) + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    edge_index = static_edge_index.to(device)
    logging.info(
        "Training batch protocol: physical_batch_size=%d, gradient_accumulation_steps=%d, "
        "nominal_effective_batch_size=%d.",
        batch_size,
        gradient_accumulation_steps,
        effective_batch_size,
    )

    best_val_loss = float("inf")
    best_epoch = None
    patience_counter = 0
    train_losses = []
    val_losses = []
    alpha_values = []
    previous_train_time = 0.0
    start_epoch = 1
    use_cuda = bool(torch.cuda.is_available() and str(device).startswith("cuda"))
    train_peak_gpu_mem_mb = None
    batches_seen = 0

    if resume_checkpoint:
        checkpoint = _torch_load(resume_checkpoint, map_location=device)
        checkpoint_fingerprint = checkpoint.get("config_fingerprint")
        matched_protocol = _match_config_fingerprint(config, checkpoint_fingerprint)
        if matched_protocol is None:
            raise ValueError(
                "Cannot resume checkpoint with a different protocol configuration: "
                f"checkpoint={checkpoint_fingerprint!r}, "
                f"expected={_config_fingerprint(config)!r}, "
                f"legacy_expected={_legacy_config_fingerprint(config)!r}."
            )
        if matched_protocol == LEGACY_CONFIG_FINGERPRINT_PROTOCOL_VERSION:
            logging.warning(
                "Resuming a legacy Git-bound checkpoint fingerprint; all training-protocol "
                "fields and the recorded Git provenance match."
            )
        expected_lagtcn_version = config.get("lagtcn_graph_source_version")
        checkpoint_lagtcn_version = checkpoint.get("lagtcn_graph_source_version")
        if expected_lagtcn_version and checkpoint_lagtcn_version != expected_lagtcn_version:
            raise ValueError(
                "Cannot resume LAGTCN from a checkpoint with a different graph-source layout: "
                f"checkpoint={checkpoint_lagtcn_version!r}, expected={expected_lagtcn_version!r}. "
                "Start a fresh run for the independent hierarchy/similarity/adaptive/dynamic model."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        if checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        patience_counter = int(checkpoint.get("patience_counter", 0))
        best_epoch_value = checkpoint.get("best_epoch")
        best_epoch = int(best_epoch_value) if best_epoch_value is not None else None
        train_losses = [float(v) for v in checkpoint.get("train_losses", [])]
        val_losses = [float(v) for v in checkpoint.get("val_losses", [])]
        alpha_values = [float(v) for v in checkpoint.get("alpha_values", [])]
        previous_train_time = float(checkpoint.get("elapsed_train_time_sec", 0.0))
        train_peak_gpu_mem_mb = checkpoint.get("train_peak_gpu_mem_mb")
        if train_peak_gpu_mem_mb is not None:
            train_peak_gpu_mem_mb = float(train_peak_gpu_mem_mb)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        batches_seen = int(
            checkpoint.get(
                "batches_seen", (start_epoch - 1) * optimizer_steps_per_epoch
            )
        )
        _restore_rng_state(checkpoint.get("rng_state"), use_cuda)
        logging.info(
            "Resumed training checkpoint %s at epoch %d/%d; best val %.6f; patience %d/%d.",
            resume_checkpoint,
            start_epoch - 1,
            epochs,
            best_val_loss,
            patience_counter,
            patience,
        )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.time()
    if start_epoch > epochs:
        logging.info(
            "Checkpoint already reached requested epoch budget (%d/%d); skipping training loop.",
            start_epoch - 1,
            epochs,
        )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        num_samples = 0

        optimizer.zero_grad()
        train_dataset_size = len(train_loader.dataset)
        train_batch_count = len(train_loader)
        for batch_index, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            assert_finite(x, "model_input", "train", config)
            assert_finite(y, "normalized_target", "train", config)

            y_pred = _forward_training_model(
                model, x, edge_index, y, batches_seen, config
            )
            loss_space = "normalized_log" if _uses_normalized_log_loss(config) else "original_scale"
            assert_finite(y_pred, f"{loss_space}_prediction", "train", config)
            y_true = _target_for_loss(model, y, y_pred, config)
            assert_finite(y_true, f"{loss_space}_target", "train", config)

            loss = _compute_loss_for_space(model, y_pred, y_true, criterion, config)
            if coherency_lambda > 0 and sum_matrix_tensor is not None:
                c_loss = compute_coherency_loss(y_pred, sum_matrix_tensor, bottom_start_idx)
                assert_finite(c_loss, "coherency_loss", "train", config)
                loss = loss + coherency_lambda * c_loss
            assert_finite(loss, "loss", "train", config)

            batch_samples = int(x.shape[0])
            running_loss += loss.item() * batch_samples
            num_samples += batch_samples

            # Weight every micro-batch by its sample share in the logical batch.
            # This also handles the final, possibly incomplete accumulation group.
            group_start = (
                batch_index // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            group_samples = min(
                effective_batch_size,
                train_dataset_size - group_start * batch_size,
            )
            (loss * (batch_samples / group_samples)).backward()

            accumulation_boundary = (
                (batch_index + 1) % gradient_accumulation_steps == 0
                or batch_index + 1 == train_batch_count
            )
            if accumulation_boundary:
                optimizer.step()
                optimizer.zero_grad()
                # DCRNN scheduled sampling is indexed by optimizer updates, as in
                # the original one-mini-batch-per-update training protocol.
                batches_seen += 1

        avg_train_loss = running_loss / max(1, num_samples)
        train_losses.append(avg_train_loss)

        val_loss = _evaluate_loss(model, val_loader, edge_index, criterion, device, config)
        val_losses.append(val_loss)

        alpha_val = _get_alpha_value(model)
        if alpha_val is not None:
            alpha_values.append(alpha_val)

        logging.info(
            "Epoch %d/%d - train loss: %.6f - val loss: %.6f",
            epoch,
            epochs,
            avg_train_loss,
            val_loss,
        )

        assert_finite(np.asarray(val_loss), "validation_loss", "validation", config)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            patience_counter = 0
            best_path = os.path.join(output_dir, artifact_filename("best_model"))
            _atomic_torch_save(model.state_dict(), best_path)
            checkpoint_hash = _sha256_file(best_path)
            _atomic_json_dump(
                {
                    "checkpoint_protocol_version": "strict_best_checkpoint_v1",
                    "best_epoch": best_epoch,
                    "best_val_metric": float(best_val_loss),
                    "best_val_metric_name": "full_output_validation_loss",
                    "config_fingerprint": _config_fingerprint(config),
                    "config_fingerprint_protocol_version": CONFIG_FINGERPRINT_PROTOCOL_VERSION,
                    "checkpoint_sha256": checkpoint_hash,
                },
                f"{best_path}.metadata.json",
            )
        else:
            patience_counter += 1

        if use_cuda:
            peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            train_peak_gpu_mem_mb = peak_mb if train_peak_gpu_mem_mb is None else max(train_peak_gpu_mem_mb, peak_mb)

        elapsed_train_time = previous_train_time + (time.time() - start_time)
        should_checkpoint = checkpoint_every > 0 and (
            epoch % checkpoint_every == 0 or epoch == epochs or patience_counter >= patience
        )
        config["_best_epoch"] = best_epoch
        config["_stop_reason"] = (
            "early_stopping" if patience_counter >= patience
            else (
                "epoch_budget_completed" if epoch == epochs
                else "in_progress"
            )
        )
        if should_checkpoint:
            _save_training_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val_loss,
                patience_counter,
                train_losses,
                val_losses,
                alpha_values,
                elapsed_train_time,
                train_peak_gpu_mem_mb,
                batches_seen,
                config,
                use_cuda,
            )
            logging.info("Training checkpoint saved to %s", checkpoint_path)

        if patience_counter >= patience:
            logging.info("Early stopping triggered at epoch %d.", epoch)
            break

    train_time = previous_train_time + (time.time() - start_time)
    best_path = os.path.join(output_dir, artifact_filename("best_model"))
    if not os.path.isfile(best_path) or not os.path.isfile(f"{best_path}.metadata.json"):
        raise RuntimeError("Training finished without a valid strict best checkpoint.")
    stop_reason = config.get("_stop_reason")
    if start_epoch > epochs:
        stop_reason = "checkpoint_already_completed_epoch_budget"
    train_efficiency = {
        "train_time_sec": float(train_time),
        "train_peak_gpu_mem_mb": float(train_peak_gpu_mem_mb) if train_peak_gpu_mem_mb is not None else None,
        "checkpoint_path": checkpoint_path if checkpoint_every > 0 else None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "start_epoch": int(start_epoch),
        "completed_epochs": int(len(train_losses)),
        "best_val_loss": float(best_val_loss) if np.isfinite(best_val_loss) else None,
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_checkpoint_sha256": _sha256_file(best_path),
        "config_fingerprint": _config_fingerprint(config),
        "config_fingerprint_protocol_version": CONFIG_FINGERPRINT_PROTOCOL_VERSION,
        "stop_reason": stop_reason,
        "physical_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "nominal_effective_batch_size": int(effective_batch_size),
        "optimizer_steps_per_epoch": int(optimizer_steps_per_epoch),
    }
    return train_losses, val_losses, train_time, alpha_values, train_efficiency

def _evaluate_loss(model, loader, edge_index, criterion, device, config=None):
    model.eval()
    running_loss = 0.0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            assert_finite(x, "model_input", "validation", config)
            assert_finite(y, "normalized_target", "validation", config)
            y_pred = _forward_normalized(model, x, edge_index) if _uses_normalized_log_loss(config) else model(x, edge_index)
            loss_space = "normalized_log" if _uses_normalized_log_loss(config) else "original_scale"
            assert_finite(y_pred, f"{loss_space}_prediction", "validation", config)
            y_true = _target_for_loss(model, y, y_pred, config)
            assert_finite(y_true, f"{loss_space}_target", "validation", config)
            loss = _compute_loss_for_space(model, y_pred, y_true, criterion, config)
            assert_finite(loss, "loss", "validation", config)
            batch_samples = int(x.shape[0])
            running_loss += loss.item() * batch_samples
            num_samples += batch_samples
    return running_loss / max(1, num_samples)


def evaluate_model(model, test_dataset, static_edge_index, criterion, device, config=None):
    batch_size = 64
    test_loader = _make_loader(test_dataset, batch_size=batch_size, shuffle=False)
    edge_index = static_edge_index.to(device)
    num_timesteps_in = 7
    if config is not None:
        num_timesteps_in = int(config.get("num_timesteps_in", num_timesteps_in))

    model.eval()
    preds = []
    trues = []
    test_loss = 0.0
    loss_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            assert_finite(x, "model_input", "test", config)
            assert_finite(y, "normalized_target", "test", config)
            if _uses_normalized_log_loss(config):
                normalized_pred = _forward_normalized(model, x, edge_index)
                assert_finite(normalized_pred, "normalized_log_prediction", "test", config)
                normalized_true = _align_target(y, normalized_pred)
                assert_finite(normalized_true, "normalized_log_target", "test", config)
                loss = _compute_loss_for_space(
                    model, normalized_pred, normalized_true, criterion, config
                ) if criterion is not None else None
                y_pred = model.prediction_from_normalized(normalized_pred)
            else:
                y_pred = model(x, edge_index)
                loss = None
            assert_finite(y_pred, "original_scale_prediction", "test", config)
            y_true = model.transform_target(y)
            y_true = _align_target(y_true, y_pred)
            assert_finite(y_true, "original_scale_target", "test", config)

            preds.append(y_pred.detach().cpu().numpy())
            trues.append(y_true.detach().cpu().numpy())
            if criterion is not None:
                if loss is None:
                    loss = _compute_model_loss(model, y_pred, y_true, criterion)
                assert_finite(loss, "loss", "test", config)
                batch_samples = int(x.shape[0])
                test_loss += loss.item() * batch_samples
                loss_samples += batch_samples

    predictions = np.concatenate(preds, axis=0)
    true_values = np.concatenate(trues, axis=0)
    metrics = _compute_metrics(
        true_values, predictions, num_timesteps_in=num_timesteps_in,
        mase_scale=(None if config is None else config.get("_mase_scale")),
        require_mase_scale=(
            config is not None
            and is_formal_ae_stage(config.get("experiment_stage"))
        ),
    )
    if loss_samples:
        metrics["loss"] = float(test_loss / loss_samples)

    logging.info(
        "Test metrics - MAE: %.4f, RMSE: %.4f, MAPE: %.4f%%, WAPE: %.4f%%, MASE: %.4f",
        metrics["MAE"],
        metrics["RMSE"],
        metrics["MAPE"],
        metrics["WAPE"],
        metrics["MASE"],
    )
    if "loss" in metrics:
        logging.info("Test loss: %.6f", metrics["loss"])
    return predictions, true_values, metrics


def evaluate_base_forecasts(model, test_dataset, static_edge_index, device):
    try:
        from lagtcn.models.backbones import BaseGCNGRUModel
    except Exception as exc:
        logging.warning("Skipping base-forecast export: %s", exc)
        return None

    if not isinstance(model, BaseGCNGRUModel):
        return None

    batch_size = 64
    test_loader = _make_loader(test_dataset, batch_size=batch_size, shuffle=False)
    edge_index = static_edge_index.to(device)
    model.eval()
    preds = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            _, base_norm = BaseGCNGRUModel.forward(model, x, edge_index)
            base_denorm = model.denormalize(base_norm)
            base_raw = model.inverse_log(base_denorm)
            preds.append(base_raw.detach().cpu().numpy())

    if not preds:
        return None
    return np.concatenate(preds, axis=0)


def benchmark_inference(
    model,
    dataset,
    static_edge_index,
    device,
    batch_size: int = 1,
    warmup_batches: int = 2,
    measure_batches: int = 20,
):
    loader = _make_loader(dataset, batch_size=batch_size, shuffle=False)
    edge_index = static_edge_index.to(device)
    use_cuda = bool(torch.cuda.is_available() and str(device).startswith("cuda"))

    model.eval()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= warmup_batches:
                break
            x = x.to(device)
            _ = model(x, edge_index)
        if use_cuda:
            torch.cuda.synchronize(device)

        total_time = 0.0
        total_batches = 0
        total_samples = 0
        for i, (x, _) in enumerate(loader):
            if i >= measure_batches:
                break
            x = x.to(device)
            if use_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model(x, edge_index)
            if use_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            total_time += elapsed
            total_batches += 1
            total_samples += int(x.shape[0])

    if total_batches == 0 or total_samples == 0 or total_time <= 0:
        return {
            "infer_latency_ms_per_batch": None,
            "infer_latency_ms_per_sample": None,
            "infer_throughput_samples_per_sec": None,
            "infer_peak_gpu_mem_mb": None,
        }

    return {
        "infer_latency_ms_per_batch": float((total_time / total_batches) * 1000.0),
        "infer_latency_ms_per_sample": float((total_time / total_samples) * 1000.0),
        "infer_throughput_samples_per_sec": float(total_samples / total_time),
        "infer_peak_gpu_mem_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)) if use_cuda else None,
    }


def save_predictions(
    predictions, true_values, config, filename=PRED_FILENAME, true_filename=TRUE_FILENAME
):
    output_dir = config.get("output_dir", ".")
    model_name = config.get("model_name", "model")
    timestamp = config.get("timestamp", "run")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    preds = np.asarray(predictions)
    trues = np.asarray(true_values)
    if preds.ndim == 2:
        preds = preds[..., None]
    if trues.ndim == 2:
        trues = trues[..., None]
    if preds.shape != trues.shape:
        raise ValueError(f"Prediction/true shape mismatch: {preds.shape} vs {trues.shape}")
    assert_finite(preds, "prediction", "save_predictions", config)
    assert_finite(trues, "target", "save_predictions", config)

    num_samples, num_nodes, num_horizons = preds.shape

    time_index = config.get("time_index")
    is_formal_ae = is_formal_ae_stage(config.get("experiment_stage"))
    if time_index is not None and len(time_index) == num_samples:
        index = pd.to_datetime(time_index)
    else:
        if is_formal_ae:
            raise ValueError(
                f"Formal prediction time_index length {None if time_index is None else len(time_index)} "
                f"does not match prediction length {num_samples}."
            )
        if time_index is not None:
            logging.warning(
                "time_index length %s does not match prediction length %s; using range index.",
                len(time_index), num_samples,
            )
        index = pd.RangeIndex(start=0, stop=num_samples, step=1)

    node_names = config.get("node_names")
    if node_names is None or len(node_names) != num_nodes:
        if is_formal_ae:
            raise ValueError(
                f"Formal node_names length {None if node_names is None else len(node_names)} "
                f"does not match node count {num_nodes}."
            )
        if node_names is not None:
            logging.warning(
                "node_names length %s does not match node count %s; using default names.",
                len(node_names), num_nodes,
            )
        node_names = [f"node_{i}" for i in range(num_nodes)]

    if num_horizons == 1:
        pred_df = pd.DataFrame(preds[:, :, 0], index=index, columns=node_names)
        true_df = pd.DataFrame(trues[:, :, 0], index=index, columns=node_names)
    else:
        flat_columns = [
            f"{node}_t+{h + 1}"
            for h in range(num_horizons)
            for node in node_names
        ]
        pred_df = pd.DataFrame(
            preds.transpose(0, 2, 1).reshape(num_samples, num_horizons * num_nodes),
            index=index,
            columns=flat_columns,
        )
        true_df = pd.DataFrame(
            trues.transpose(0, 2, 1).reshape(num_samples, num_horizons * num_nodes),
            index=index,
            columns=flat_columns,
        )

    pred_path = os.path.join(output_dir, filename)
    true_path = os.path.join(output_dir, true_filename)
    pred_df.to_csv(pred_path)
    true_df.to_csv(true_path)
    logging.info("Predictions saved to %s", pred_path)
    logging.info("True values saved to %s", true_path)


def plot_loss_curves(train_losses, val_losses, config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("Skipping loss-curve generation: %s", exc)
        return

    output_dir = config.get("output_dir", ".")
    model_name = config.get("model_name", "model")
    timestamp = config.get("timestamp", "run")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    epochs = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, label="Train", linewidth=1.6)
    if val_losses:
        ax.plot(epochs[:len(val_losses)], val_losses, label="Validation", linewidth=1.6)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve - {model_name}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(plot_dir, artifact_filename("loss_curve"))
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logging.info("Loss curve saved to %s", plot_path)


def plot_predictions(predictions, true_values, node_idx, config):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logging.warning("Skipping plot generation: %s", exc)
        return

    output_dir = config.get("output_dir", ".")
    model_name = config.get("model_name", "model")
    timestamp = config.get("timestamp", "run")
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    preds_all = np.asarray(predictions)
    trues_all = np.asarray(true_values)
    horizon_idx = int(config.get("plot_horizon_idx", 0))

    if preds_all.ndim == 3:
        horizon_idx = max(0, min(horizon_idx, preds_all.shape[-1] - 1))
        preds = preds_all[:, node_idx, horizon_idx]
    else:
        preds = preds_all[:, node_idx]
        horizon_idx = 0
    if trues_all.ndim == 3:
        trues = trues_all[:, node_idx, horizon_idx]
    else:
        trues = trues_all[:, node_idx]

    time_index = config.get("time_index")
    if time_index is not None and len(time_index) == len(preds):
        x_vals = pd.to_datetime(time_index)
    else:
        x_vals = np.arange(len(preds))

    node_names = config.get("node_names")
    node_name = node_names[node_idx] if node_names and node_idx < len(node_names) else f"node_{node_idx}"
    safe_node_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(node_name))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_vals, trues, label="True", linewidth=1.2)
    ax.plot(x_vals, preds, label="Pred", linewidth=1.2)
    ax.set_title(f"{node_name} - {model_name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(
        plot_dir,
        f"pred_{safe_node_name}_h{horizon_idx + 1}.png"
    )
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def _json_ready(obj):
    if isinstance(obj, dict):
        return {k: _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (np.datetime64,)):
        return pd.to_datetime(obj).isoformat()
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_model_info(model, config, metrics, level_metrics, training_results):
    output_dir = config.get("output_dir", ".")
    model_name = config.get("model_name", "model")
    timestamp = config.get("timestamp", "run")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "model_name": model_name,
        "timestamp": timestamp,
        "params": {
            "total": int(total_params),
            "trainable": int(trainable_params),
        },
        "config": config,
        "metrics": metrics,
        "level_metrics": level_metrics.to_dict(orient="records") if isinstance(level_metrics, pd.DataFrame) else level_metrics,
        "training_results": training_results,
    }

    info_path = os.path.join(output_dir, artifact_filename("model_info"))
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            _json_ready(info), f, indent=4, ensure_ascii=False, allow_nan=False
        )
    logging.info("Model info saved to %s", info_path)
