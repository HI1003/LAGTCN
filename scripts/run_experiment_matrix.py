#!/usr/bin/env python3
"""Run compatibility experiment grids with fixed structural graphs.

The frozen S/A/D protocol is owned by ``build_ae_graph_tuning_manifest.py``
and ``build_ae_final_manifest.py``; this runner does not implement data-driven
graph fusion.

  A. temporal       – Strong temporal-only forecasters without graph input
  B. stgnn          – Dedicated graph-temporal forecasters
  C. graph_temporal – Fixed hierarchy graph + temporal backbone combinations

Historical fixed-graph compatibility stages:
  0. tuning       – Per-model hyperparameter selection on each dataset
  1. graph        – Fixed structural topology comparison (H versus HG)
  2. architecture – Top-k graph × AE candidate backbone × ST-mode search
  3. reconcile    – Top-k architecture × reconciliation strategy search
  4. baseline     – External baselines (temporal-only + graph-temporal)
  5. ablation     – Framework innovations ablation
  6. sensitivity  – Input window length sensitivity (L_in={24,168,336}, supplementary)

Key design decisions:
  - Hyperparameters: {lr, hidden_dim} searched PER DATASET (validation loss, 2 seeds).
    Fixed: batch=128, layers=2, patience=20, epochs=150, L_in=168.
  - Horizons: h={1, 6, 24} for the three hourly GEFCom datasets.
    Multi-horizon degradation figure extracted from Stage 4 results (no extra runs).
  - Metrics: MAE + RMSE (main text); coherency NMAE% for reconcile tables.
    WAPE, MASE, per-level metrics → supplementary material.
  - External graph baselines use fixed H or their paper-defined native graph.
  - Only LAGTCN owns the current S/A/D independent-propagation fusion.

Workflow:
  # Journal priority lane, h=24, two GEF 2017 datasets
  grun -w python scripts/run_experiment_matrix.py --stage temporal \\
      --datasets GEFCom2017QualifyingMatch_3level,GEFCom2017FinalMatch_4level \\
      --horizons 24 --seeds 42,43 --batch-size 16 --device cuda:0

  grun -w python scripts/run_experiment_matrix.py --stage stgnn \\
      --datasets GEFCom2017QualifyingMatch_3level,GEFCom2017FinalMatch_4level \\
      --horizons 24 --seeds 42,43 --batch-size 16 --device cuda:0

  grun -w python scripts/run_experiment_matrix.py --stage graph_temporal \\
      --datasets GEFCom2017QualifyingMatch_3level,GEFCom2017FinalMatch_4level \\
      --horizons 24 --seeds 42,43 --batch-size 16 --device cuda:0

  # Step 0: Hyperparameter selection (on each dataset, h=1, 2 seeds)
  grun -w python scripts/run_experiment_matrix.py --stage tuning --device cuda:0

  # Step 1: Graph topology ablation
  grun -w python scripts/run_experiment_matrix.py --stage graph --device cuda:0

  # Step 2: Architecture search (auto-uses Stage 1 top-k graphs, plus H)
  grun -w python scripts/run_experiment_matrix.py --stage architecture \\
      --architecture-graph-topk 2 --device cuda:0

  # Step 3: Reconciliation (auto-selects Top-k graph/backbone/ST choices from Stage 2)
  grun -w python scripts/run_experiment_matrix.py --stage reconcile --device cuda:0

  # Step 4: External baselines (use the validation-frozen graph/reconcile choice)
  grun -w python scripts/run_experiment_matrix.py --stage baseline \\
      --best-graph "<selected_graph>" --best-reconcile "<selected_reconcile>" --device cuda:0

  # Step 5: Ablation (use the validation-frozen graph)
  grun -w python scripts/run_experiment_matrix.py --stage ablation \\
      --override-graph "<selected_graph>" --device cuda:0

  # Step 6: Input window sensitivity (supplementary material)
  grun -w python scripts/run_experiment_matrix.py --stage sensitivity \\
      --override-graph "<selected_graph>" --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from graph_sparsity import FINAL_GRAPH_SOURCE_POLICY

from output_naming import (
    LAGTCN_GRAPH_SOURCE_VERSION_CURRENT,
    compact_namespace,
    compact_output_namespace,
    normalize_graph_mode,
    paper_manifest_filename,
    selected_runs_filename,
    short_run_label,
)


# ===================================================================
# Model / graph / backbone constants
# ===================================================================
GRAPH_MODES_ALL = ["I", "H", "HG"]  # Compatibility runner; formal LAGTCN S/A/D use dedicated manifests.
RECONCILE_MODELS = [
    "GCN-GRU-LP-NO",
    "GCN-GRU-LP-BUD", "GCN-GRU-LP-BUL", "GCN-GRU-LP-BUN",
    "GCN-GRU-LP-TDD", "GCN-GRU-LP-TDL", "GCN-GRU-LP-TDN",
    "GCN-GRU-LP-HYBD", "GCN-GRU-LP-HYBL", "GCN-GRU-LP-HYBN",
]
GNN_TYPES = ["none", "gcn", "gatv2", "graphsage", "transformer"]
TEMPORAL_TYPES = ["gru", "transformer", "tcn", "timemixer", "patchtst", "dlinear", "itransformer"]
ARCH_ST_MODES = ["sequential", "alternating", "hier_fusion"]
ARCH_BACKBONES = [
    # Candidate backbones retained for the Applied Energy compatibility matrix.
    ("gcn", "gru"),
    ("gcn", "tcn"),
    ("gcn", "timemixer"),
    ("gcn", "patchtst"),
    ("gatv2", "tcn"),
    ("graphsage", "tcn"),
    # Temporal-only controls are kept in Stage 2 but are run only once under H.
    ("none", "tcn"),
    ("none", "timemixer"),
]
ARCH_BACKBONES_SUPPLEMENTARY = [
    ("gcn", "transformer"),
    ("gcn", "dlinear"),
    ("gcn", "itransformer"),
    ("gatv2", "gru"),
    ("graphsage", "gru"),
    ("transformer", "gru"),
    ("transformer", "tcn"),
]
STRONG_TEMPORAL_MODELS = ["DLINEAR", "PATCHTST", "NHITS", "ITRANSFORMER"]
EXTRA_TEMPORAL_MODELS = []
GRAPH_NATIVE_TEMPORAL_MODELS = ["GRAPH_DLINEAR", "GRAPH_PATCHTST", "GRAPH_ITRANSFORMER", "GRAPH_ADAPTER", "LAGTCN"]
GRAPH_ADAPTER_TEMPORAL_TYPES = ["patchtst", "itransformer"]
DEDICATED_GRAPH_MODELS = ["DCRNN", "STGCN", "GWNET", "MTGNN", "DEEPHGNN_SPECTGNN", "AGCRN"]
EXTRA_GRAPH_MODELS = [*GRAPH_NATIVE_TEMPORAL_MODELS, *DEDICATED_GRAPH_MODELS]
NATIVE_GRAPH_STGNN_MODELS = {"GWNET", "MTGNN", "AGCRN"}
FIXED_GRAPH_STGNN_MODELS = {"DCRNN", "STGCN", "DEEPHGNN_SPECTGNN"}
STGNN_GRAPH_SOURCES = ["project", "native", "hybrid"]

ALL_DATASETS = [
    "GEFCom2012_2level",
    "GEFCom2017QualifyingMatch_3level",
    "GEFCom2017FinalMatch_4level",
]

PAPER_RESULTS_ROOTS = {
    "journal_applied_energy": Path("results"),
}

STAGE_LABELS = {
    "tuning": "stage_0_tuning",
    "graph": "stage_1_graph",
    "architecture": "stage_2_architecture",
    "reconcile": "stage_3_reconcile",
    "baseline": "stage_4_baseline",
    "ablation": "stage_5_ablation",
    "sensitivity": "stage_6_sensitivity",
    "temporal": "stage_a_temporal",
    "stgnn": "stage_b_stgnn",
    "graph_temporal": "stage_c_graph_temporal",
}

# ===================================================================
# Stage 0: Per-model hyperparameter grid
# Searched PER DATASET on validation MAE (h=1, 2 seeds for speed)
# ===================================================================
HP_GRID = {
    "lr": [5e-4, 1e-3],
    "hidden_dim": [64, 96, 128],
}
TUNING_SEEDS = [42, 43]   # fewer seeds for speed

# Models that participate in tuning
TUNING_MODELS = [
    # (model_name, gnn_type, temporal_type, graph_mode)
    ("GCN-GRU-LP-BUD", "gcn", "gru", "H"),
    ("DLINEAR", "none", "gru", "H"),
    ("PATCHTST", "none", "gru", "H"),
    ("NHITS", "none", "gru", "H"),
    ("ITRANSFORMER", "none", "gru", "H"),
    ("GRAPH_DLINEAR", "gcn", "gru", "H"),
    ("GRAPH_PATCHTST", "gcn", "gru", "H"),
    ("GRAPH_ITRANSFORMER", "gcn", "gru", "H"),
    ("DCRNN", "gcn", "gru", "H"),
    ("STGCN", "gcn", "gru", "H"),
    ("GWNET", "gcn", "gru", "H"),
    ("MTGNN", "gcn", "gru", "H"),
    ("AGCRN", "gcn", "gru", "H"),
]


@dataclass(frozen=True)
class Preset:
    graph_modes: list[str]
    model_names: list[str]
    horizons: list[int]
    backbones: list[tuple[str, str]]
    extra_args: list[list[str]] = field(default_factory=lambda: [[]])


@dataclass(frozen=True)
class ArchitectureChoice:
    graph_mode: str
    gnn_type: str
    temporal_type: str
    st_mode: str


# ===================================================================
# Stage presets
# ===================================================================
STAGE_PRESETS: dict[str, Preset] = {
    "tuning": Preset(
        graph_modes=["H"],
        model_names=["GCN-GRU-LP-BUD"],
        horizons=[1],
        backbones=[("gcn", "gru")],
    ),
    "graph": Preset(
        graph_modes=["H", "HG"],
        model_names=["GCN-GRU-LP-NO"],
        horizons=[1, 6, 24],
        backbones=[("gcn", "gru")],
    ),
    "architecture": Preset(
        graph_modes=["H"],
        model_names=["GCN-GRU-LP-NO"],
        horizons=[1, 6, 24],
        backbones=ARCH_BACKBONES,
    ),
    "reconcile": Preset(
        graph_modes=["H"],
        model_names=RECONCILE_MODELS,
        horizons=[1, 6, 24],
        backbones=[("gcn", "gru")],
    ),
    "baseline": Preset(
        graph_modes=["H"],
        model_names=[
            "GCN-GRU-LP-NO",
            *STRONG_TEMPORAL_MODELS,
            *EXTRA_TEMPORAL_MODELS,
            *EXTRA_GRAPH_MODELS,
        ],
        horizons=[1, 6, 24],
        backbones=[("gcn", "gru")],
    ),
    "ablation": Preset(
        graph_modes=["H"],
        model_names=["GCN-GRU-LP-NO"],
        horizons=[1, 6, 24],
        backbones=[("gcn", "gru")],
        extra_args=[
            # A: full proposed (default: learnable weights, sequential ST)
            [],
            # B-D: coherency loss sweep
            ["--coherency-lambda", "0.01"],
            ["--coherency-lambda", "0.05"],
            ["--coherency-lambda", "0.1"],
            # E: alternating spatio-temporal mode
            ["--st-mode", "alternating"],
            # F: alternating + coherency
            ["--st-mode", "alternating", "--coherency-lambda", "0.05"],
        ],
    ),
    # Stage 6: Input window sensitivity (supplementary material)
    # Tests L_in={24, 168, 336} on proposed model + 1 baseline, h=24 only
    # NOTE: L_in is overridden via --num-timesteps-in; this stage runs 3 times
    #   with different --num-timesteps-in values (see workflow in docstring)
    "sensitivity": Preset(
        graph_modes=["H"],
        model_names=["GCN-GRU-LP-NO", "DLINEAR"],
        horizons=[24],              # only day-ahead (most sensitive to L_in)
        backbones=[("gcn", "gru")],
    ),
    "temporal": Preset(
        graph_modes=["H"],          # graph is ignored by temporal-only models
        model_names=[*STRONG_TEMPORAL_MODELS, *EXTRA_TEMPORAL_MODELS],
        horizons=[1, 6, 24],
        backbones=[("none", "gru")],
    ),
    "stgnn": Preset(
        graph_modes=["H"],
        model_names=DEDICATED_GRAPH_MODELS,
        horizons=[1, 6, 24],
        backbones=[("gcn", "gru")],
    ),
    "graph_temporal": Preset(
        graph_modes=["H"],
        model_names=GRAPH_NATIVE_TEMPORAL_MODELS,
        horizons=[1, 6, 24],
        backbones=[("gcn", "gru")],
    ),
}


# ===================================================================
# Parsing helpers
# ===================================================================
def _parse_csv_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _iter_model_info_files(root: Path):
    seen = set()
    for pattern in ("**/model_info.json", "**/model_info_*.json"):
        for path in sorted(root.glob(pattern)):
            if path not in seen:
                seen.add(path)
                yield path


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_backbone_list(value: str) -> list[tuple[str, str]]:
    items = []
    for token in _parse_csv_list(value):
        if "-" not in token:
            raise ValueError(f"Invalid backbone '{token}'. Expected 'gcn-gru' format.")
        gnn_type, temporal_type = token.split("-", 1)
        items.append((gnn_type.strip().lower(), temporal_type.strip().lower()))
    if not items:
        raise ValueError("No valid backbones parsed.")
    return items


def _parse_st_modes(value: str) -> list[str]:
    aliases = {
        "hierarchy_fusion": "hier_fusion",
        "hierarchical_fusion": "hier_fusion",
    }
    modes = []
    for token in _parse_csv_list(value):
        mode = aliases.get(token.strip().lower(), token.strip().lower())
        if mode not in {"sequential", "alternating", "hier_fusion"}:
            raise ValueError(
                f"Invalid ST mode '{token}'. Choose from sequential, alternating, hier_fusion."
            )
        modes.append(mode)
    if not modes:
        raise ValueError("No valid ST modes parsed.")
    return modes


def _safe_segment(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = text.strip("-._")
    return text or "default"


def _stage_label(stage: str) -> str:
    return STAGE_LABELS.get(stage, _safe_segment(stage))


def _paper_results_root(project_root: Path, paper_scope: str) -> Path:
    rel = PAPER_RESULTS_ROOTS.get(str(paper_scope))
    if rel is None:
        return project_root / "results" / _safe_segment(paper_scope)
    return project_root / rel


def _output_namespace(args: argparse.Namespace, stage_label: str, dataset: str) -> str:
    if args.output_namespace:
        return compact_namespace(args.output_namespace)
    feature_tag = _safe_segment(args.feature_set or "target")
    return compact_output_namespace(_safe_segment(args.paper_scope), stage_label, feature_tag)


def _feature_metadata(args: argparse.Namespace, dataset: str) -> dict:
    return {"feature_set": args.feature_set or "target"}


def _make_run_label(
    stage: str,
    model_name: str,
    gnn_type: str,
    temporal_type: str,
    graph_mode: str,
    horizon: int,
    seed: int,
    variant: str | None = None,
) -> str:
    stgnn_graph_source = None
    clean_variant = variant
    if variant:
        variant_text = str(variant)
        if variant_text.startswith(("graphsrc-", "graphsrc=")):
            stgnn_graph_source = re.split("[-=]", variant_text, maxsplit=1)[1]
            clean_variant = None
        elif variant_text.startswith(("st-", "st=")):
            clean_variant = re.split("[-=]", variant_text, maxsplit=1)[1]
    return short_run_label(
        stage=_stage_label(stage),
        model_name=model_name,
        gnn_type=gnn_type,
        temporal_type=temporal_type,
        graph_mode=graph_mode,
        horizon=horizon,
        seed=seed,
        st_mode=clean_variant if clean_variant in {"sequential", "alternating", "hier_fusion"} else None,
        stgnn_graph_source=stgnn_graph_source,
        variant=(None if clean_variant in {"sequential", "alternating", "hier_fusion"} else clean_variant),
    )


def _model_family(model_name: str) -> str:
    model = str(model_name).upper()
    if model in {*STRONG_TEMPORAL_MODELS, *EXTRA_TEMPORAL_MODELS}:
        return "temporal-only"
    if model in GRAPH_NATIVE_TEMPORAL_MODELS:
        return "graph-enhanced-temporal"
    if model in DEDICATED_GRAPH_MODELS:
        return "dedicated-STGNN"
    if model == "GCN-GRU-LP-NO":
        return "GCN-GRU reference"
    if model.startswith("GCN-GRU-LP-"):
        return "GCN-GRU reconciliation"
    return "other"


def _graph_sparsity_metadata(args: argparse.Namespace) -> dict:
    return {
        "graph_sparsity_policy": args.graph_sparsity_policy,
        "static_threshold": getattr(args, "static_threshold", None),
        "adaptive_top_k": getattr(args, "adaptive_top_k", None),
        "dynamic_threshold": getattr(args, "dynamic_threshold", None),
    }


def _stgnn_sources_for_model(model_name: str, requested_sources: list[str]) -> list[str]:
    model = str(model_name).upper()
    if model in FIXED_GRAPH_STGNN_MODELS:
        return ["project"]
    if model in NATIVE_GRAPH_STGNN_MODELS:
        return requested_sources
    return ["project"]


def _stgnn_graph_source_from_extra_args(extra_args: list[str] | None) -> str | None:
    if not extra_args:
        return None
    for idx, token in enumerate(extra_args):
        if token == "--stgnn-graph-source" and idx + 1 < len(extra_args):
            return str(extra_args[idx + 1]).lower()
    return None


# ===================================================================
# Per-model / per-stage resolution
# ===================================================================
def _resolve_stage_backbones(
    stage: str, model_name: str, default_backbones: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if model_name == "DEEPHGNN_SPECTGNN":
        return [("gegenconv", "tgc")]
    if stage == "temporal" and model_name in {*STRONG_TEMPORAL_MODELS, *EXTRA_TEMPORAL_MODELS}:
        return [("none", "gru")]
    if stage not in ("baseline", "sensitivity"):
        return default_backbones
    if model_name in {*STRONG_TEMPORAL_MODELS, *EXTRA_TEMPORAL_MODELS}:
        return [("none", "gru")]
    if model_name in {"GRAPH_ADAPTER", "LAGTCN"}:
        return [("gcn", temporal) for temporal in GRAPH_ADAPTER_TEMPORAL_TYPES]
    if model_name in EXTRA_GRAPH_MODELS:
        return [("gcn", "gru")]
    return default_backbones


def _uses_graph_encoder(gnn_type: str) -> bool:
    return str(gnn_type).strip().lower() != "none"


def _resolve_graph_for_model(
    stage: str, model_name: str, best_graph: str, default_graph: str,
) -> str:
    """Fair graph assignment per model type in baseline/horizon stages.

    - Temporal-only: graph irrelevant → "H"
    - Graph baselines: use the fixed hierarchy graph H
    - LAGTCN: this compatibility runner accepts only I/H/HG; formal S/A/D
      configurations are owned by build_ae_graph_tuning_manifest.py and
      build_ae_final_manifest.py
    """
    if stage not in ("baseline", "sensitivity"):
        return default_graph
    if model_name in {*STRONG_TEMPORAL_MODELS, *EXTRA_TEMPORAL_MODELS}:
        return "H"
    if model_name in EXTRA_GRAPH_MODELS:
        return "H"
    return best_graph


def _default_hidden_dim(dataset: str) -> int:
    return 128 if dataset == "GEFCom2017FinalMatch_4level" else 96


def _resolved_hidden_dim(args: argparse.Namespace, dataset: str, hp_hidden: int | None) -> int:
    if args.hidden_dim is not None:
        return int(args.hidden_dim)
    if hp_hidden is not None:
        return int(hp_hidden)
    return _default_hidden_dim(dataset)


# ===================================================================
# best_hparams.json loader + lookup
# ===================================================================
def _load_best_hparams(path: Path | None) -> dict:
    if path is None or not Path(path).exists():
        return {}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[WARN] could not parse {path}: {exc}")
        return {}


def _lookup_hp(
    best_hp: dict, dataset: str, model_name: str,
) -> tuple[float | None, int | None]:
    """Return (lr, hidden_dim) for (dataset, model_name).

    Fallback chain:
      1. Exact match (dataset, model_name).
      2. GCN-GRU-LP-* reconcile variants inherit from GCN-GRU-LP-BUD
         (they share the same base encoder, only the reconcile head differs).
      3. None, None  -> caller uses CLI args / defaults.
    """
    ds_map = best_hp.get(dataset, {})
    if model_name in ds_map:
        e = ds_map[model_name]
        return e.get("lr"), e.get("hidden_dim")
    if model_name.startswith("GCN-GRU-LP-") and "GCN-GRU-LP-BUD" in ds_map:
        e = ds_map["GCN-GRU-LP-BUD"]
        return e.get("lr"), e.get("hidden_dim")
    return None, None


# ===================================================================
# Stage 2 → Stage 3 backbone auto-selection
# ===================================================================
def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _best_val_loss(payload: dict) -> float | None:
    training = payload.get("training_results", {}) or {}
    val_losses = training.get("val_losses")
    if not val_losses:
        return None
    vals = [_to_float(v) for v in val_losses]
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None


def _metric_score(payload: dict, metric_name: str) -> float | None:
    metric_key = str(metric_name).strip()
    normalized = metric_key.lower().replace("-", "_")
    if normalized in {"val_loss", "best_val_loss", "validation_loss"}:
        return _best_val_loss(payload)
    metrics = payload.get("metrics", {}) or {}
    return _to_float(metrics.get(metric_key))


def _select_top_graphs_from_stage1(
    data_root: Path,
    dataset: str,
    horizon: int,
    metric_name: str = "MAE",
    topk: int = 2,
    paper_scope: str | None = None,
    experiment_stage: str | None = "stage_1_graph",
    experiment_id: str | None = None,
    include_graphs: list[str] | None = None,
    graph_sparsity_policy: str | None = None,
) -> list[str]:
    """Select Stage-1 graph candidates by validation-selected test records.

    Stage 1 is only a screening step. The returned list is intentionally small:
    top-k graphs by the requested metric, optionally plus fixed reference graphs
    such as H for interpretability.
    """
    output_root = data_root / dataset / "output"
    if not output_root.exists():
        return []

    scores = defaultdict(list)
    for path in _iter_model_info_files(output_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = payload.get("config", {})
        if paper_scope and str(cfg.get("paper_scope", "")) != str(paper_scope):
            continue
        if experiment_stage and str(cfg.get("experiment_stage", "")) != str(experiment_stage):
            continue
        if experiment_id and str(cfg.get("experiment_id", "")) != str(experiment_id):
            continue
        if graph_sparsity_policy and str(cfg.get("graph_sparsity_policy", "")) != str(graph_sparsity_policy):
            continue
        if str(cfg.get("model_name")) != "GCN-GRU-LP-NO":
            continue
        if int(cfg.get("num_timesteps_out", -1)) != int(horizon):
            continue
        graph_mode = normalize_graph_mode(cfg.get("graph_mode", ""))
        if graph_mode not in GRAPH_MODES_ALL:
            continue
        score = _metric_score(payload, metric_name)
        if score is not None:
            scores[graph_mode].append(score)

    ranked = [
        (sum(v) / len(v), -len(v), graph)
        for graph, v in scores.items() if v
    ]
    ranked.sort()

    selected = [item[2] for item in ranked[:max(1, int(topk))]]
    for graph in include_graphs or []:
        graph = graph.strip().upper()
        if graph and graph not in selected:
            selected.append(graph)
    return selected


def _select_top_architectures_from_stage2(
    data_root: Path,
    dataset: str,
    horizon: int,
    metric_name: str = "MAE",
    topk: int = 2,
    paper_scope: str | None = None,
    experiment_stage: str | None = "stage_2_architecture",
    experiment_id: str | None = None,
    graph_sparsity_policy: str | None = None,
) -> list[ArchitectureChoice]:
    """Select top graph/backbone/ST-mode choices from Stage 2."""
    output_root = data_root / dataset / "output"
    if not output_root.exists():
        return []

    scores = defaultdict(list)
    for path in _iter_model_info_files(output_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = payload.get("config", {})
        if paper_scope and str(cfg.get("paper_scope", "")) != str(paper_scope):
            continue
        if experiment_stage and str(cfg.get("experiment_stage", "")) != str(experiment_stage):
            continue
        if experiment_id and str(cfg.get("experiment_id", "")) != str(experiment_id):
            continue
        if graph_sparsity_policy and str(cfg.get("graph_sparsity_policy", "")) != str(graph_sparsity_policy):
            continue
        if str(cfg.get("model_name")) != "GCN-GRU-LP-NO":
            continue
        if int(cfg.get("num_timesteps_out", -1)) != int(horizon):
            continue
        graph_mode = normalize_graph_mode(cfg.get("graph_mode", ""))
        gnn_type = str(cfg.get("gnn_type", "gcn")).lower()
        temporal_type = str(cfg.get("temporal_type", "gru")).lower()
        st_mode = str(cfg.get("st_mode", "sequential")).lower()
        if st_mode in {"hierarchy_fusion", "hierarchical_fusion"}:
            st_mode = "hier_fusion"
        if graph_mode not in GRAPH_MODES_ALL:
            continue
        if gnn_type not in GNN_TYPES or temporal_type not in TEMPORAL_TYPES:
            continue
        if st_mode not in {"sequential", "alternating", "hier_fusion"}:
            continue
        score = _metric_score(payload, metric_name)
        if score is not None:
            scores[(graph_mode, gnn_type, temporal_type, st_mode)].append(score)

    ranked = [
        (sum(v) / len(v), -len(v), choice)
        for choice, v in scores.items() if v
    ]
    ranked.sort()
    return [
        ArchitectureChoice(
            graph_mode=item[2][0],
            gnn_type=item[2][1],
            temporal_type=item[2][2],
            st_mode=item[2][3],
        )
        for item in ranked[:max(1, int(topk))]
    ]


def _select_top_backbones_from_stage2(
    data_root: Path, dataset: str, horizon: int,
    graph_mode: str = "H", metric_name: str = "MAE", topk: int = 2,
    paper_scope: str | None = None,
    experiment_stage: str | None = "stage_2_architecture",
) -> list[tuple[str, str]]:
    output_root = data_root / dataset / "output"
    if not output_root.exists():
        return []

    scores = defaultdict(list)
    for path in _iter_model_info_files(output_root):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = payload.get("config", {})
        if paper_scope and str(cfg.get("paper_scope", "")) != str(paper_scope):
            continue
        if experiment_stage and str(cfg.get("experiment_stage", "")) != str(experiment_stage):
            continue
        if str(cfg.get("model_name")) != "GCN-GRU-LP-NO":
            continue
        if str(cfg.get("graph_mode", "")).upper() != graph_mode.upper():
            continue
        if int(cfg.get("num_timesteps_out", -1)) != int(horizon):
            continue
        gnn_type = str(cfg.get("gnn_type", "gcn")).lower()
        temporal_type = str(cfg.get("temporal_type", "gru")).lower()
        if gnn_type not in GNN_TYPES or temporal_type not in TEMPORAL_TYPES:
            continue
        score = _metric_score(payload, metric_name)
        if score is not None:
            scores[(gnn_type, temporal_type)].append(score)

    ranked = [
        (sum(v) / len(v), -len(v), bb)
        for bb, v in scores.items() if v
    ]
    ranked.sort()
    return [item[2] for item in ranked[:max(1, int(topk))]]


# ===================================================================
# Command builder
# ===================================================================
def _build_command(
    args: argparse.Namespace,
    dataset: str, seed: int, graph_mode: str, model_name: str,
    horizon: int, gnn_type: str, temporal_type: str,
    extra_args: list[str] | None = None,
    override_lr: float | None = None,
    override_hidden: int | None = None,
    run_label: str | None = None,
) -> list[str]:
    lr = override_lr if override_lr is not None else args.lr
    hidden = (
        args.hidden_dim
        if args.hidden_dim is not None
        else (override_hidden if override_hidden is not None else _default_hidden_dim(dataset))
    )
    stage_label = _stage_label(args.stage)
    cmd = [
        sys.executable, "code/main.py",
        "--dataset", dataset,
        "--batch-size", str(args.batch_size),
        "--num-timesteps-in", str(args.num_timesteps_in),
        "--num-timesteps-out", str(horizon),
        "--hidden-dim", str(hidden),
        "--num-layers", str(args.num_layers),
        "--lr", str(lr),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--resume", args.resume,
        "--checkpoint-every-epochs", str(args.checkpoint_every_epochs),
        "--graph-mode", graph_mode,
        "--sim-type", args.sim_type,
        "--graph-sparsity-policy", args.graph_sparsity_policy,
        "--native-top-k", str(getattr(args, "native_top_k", 5)),
        "--deephgnn-hierarchical-loss-weight", str(
            getattr(args, "deephgnn_hierarchical_loss_weight", 1.0)
        ),
        "--spectgnn-alpha", str(getattr(args, "spectgnn_alpha", 1.2)),
        "--spectgnn-degree", str(getattr(args, "spectgnn_degree", 4)),
        "--spectgnn-modes", str(getattr(args, "spectgnn_modes", 5)),
        "--spectgnn-trend-window", str(getattr(args, "spectgnn_trend_window", 24)),
        "--model-name", model_name,
        "--gnn-type", gnn_type,
        "--temporal-type", temporal_type,
        "--seed", str(seed),
        "--device", args.device,
        "--paper-scope", _safe_segment(args.paper_scope),
        "--experiment-stage", stage_label,
        "--experiment-id", str(args.experiment_id),
        "--output-namespace", _output_namespace(args, stage_label, dataset),
        "--run-label", run_label or _make_run_label(
            args.stage, model_name, gnn_type, temporal_type, graph_mode, horizon, seed
        ),
    ]
    if getattr(args, "static_threshold", None) is not None:
        cmd += ["--static-threshold", str(args.static_threshold)]
    if getattr(args, "adaptive_top_k", None) is not None:
        cmd += ["--adaptive-top-k", str(args.adaptive_top_k)]
    if getattr(args, "dynamic_threshold", None) is not None:
        cmd += ["--dynamic-threshold", str(args.dynamic_threshold)]
    if args.feature_set:
        cmd += ["--feature-set", args.feature_set]
    if args.no_plots:
        cmd.append("--no-plots")
    if args.plot_node_limit is not None:
        cmd += ["--plot-node-limit", str(args.plot_node_limit)]
    if model_name.startswith("GCN-GRU-LP-"):
        cmd += ["--mlp-hidden-dims", args.mlp_hidden_dims]
        cmd += ["--mlp-dropout", str(args.mlp_dropout)]
        cmd += ["--mlp-layer-norm", str(args.mlp_layer_norm).lower()]
        cmd += ["--mlp-activation", args.mlp_activation]
        cmd += ["--mixing-hidden-dims", args.mixing_hidden_dims]
        cmd += ["--mixing-dropout", str(args.mixing_dropout)]
    if extra_args:
        cmd.extend(extra_args)
    return cmd


def _describe_extra_args(extra_args: list[str]) -> str:
    if not extra_args:
        return "default"
    parts = []
    it = iter(extra_args)
    for token in it:
        if token == "--coherency-lambda":
            parts.append(f"coh={next(it)}")
        elif token == "--st-mode":
            parts.append(f"st={next(it)}")
        elif token == "--stgnn-graph-source":
            parts.append(f"graphsrc={next(it)}")
        else:
            parts.append(token)
    return "+".join(parts)


def _st_mode_from_extra_args(extra_args: list[str] | None) -> str:
    if not extra_args:
        return "sequential"
    for idx, token in enumerate(extra_args):
        if token == "--st-mode" and idx + 1 < len(extra_args):
            mode = str(extra_args[idx + 1]).lower()
            if mode in {"hierarchy_fusion", "hierarchical_fusion"}:
                return "hier_fusion"
            return mode
    return "sequential"


# ===================================================================
# Stage 0: Tuning runs (per dataset)
# ===================================================================
def _build_tuning_runs(args: argparse.Namespace, datasets: list[str]) -> list[dict]:
    runs = []
    seeds = _parse_int_list(args.tuning_seeds)
    for dataset in datasets:
        for model_name, gnn_type, temporal_type, graph_mode in TUNING_MODELS:
            for lr in HP_GRID["lr"]:
                for hidden in HP_GRID["hidden_dim"]:
                    for seed in seeds:
                        variant = f"lr{lr}_hd{hidden}"
                        run_label = _make_run_label(
                            args.stage,
                            model_name,
                            gnn_type,
                            temporal_type,
                            graph_mode,
                            1,
                            seed,
                            variant=variant,
                        )
                        cmd = _build_command(
                            args, dataset, seed, graph_mode, model_name,
                            1, gnn_type, temporal_type,
                            override_lr=lr, override_hidden=hidden,
                            run_label=run_label,
                        )
                        stage_label = _stage_label(args.stage)
                        runs.append({
                            "paper_scope": args.paper_scope,
                            "experiment_stage": stage_label,
                            "experiment_id": args.experiment_id,
                            "output_namespace": _output_namespace(args, stage_label, dataset),
                            "run_label": run_label,
                            "dataset": dataset,
                            **_feature_metadata(args, dataset),
                            "graph_mode": graph_mode,
                            "model_name": model_name,
                            "model_family": _model_family(model_name),
                            "gnn_type": gnn_type,
                            "temporal_type": temporal_type,
                            "seed": seed,
                            "horizon": 1,
                            "lr": lr,
                            "hidden_dim": hidden,
                            **_graph_sparsity_metadata(args),
                            "cmd": cmd,
                        })
    return runs


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run experiment matrix for AE-style evaluations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", type=str, default="graph",
                        choices=sorted(STAGE_PRESETS.keys()))
    parser.add_argument("--datasets", type=str, default=",".join(ALL_DATASETS))
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--data-root", type=str, default="Data")
    parser.add_argument(
        "--paper-scope",
        type=str,
        default="journal_applied_energy",
        choices=["journal_applied_energy"],
        help="Logical paper owner for outputs. Applied Energy is the only supported paper scope.",
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Batch id shared by manifest entries and code/main.py outputs. Defaults to current timestamp.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Directory for manifest JSONL files. Defaults to <paper>/results/raw_manifests.",
    )
    parser.add_argument(
        "--output-namespace",
        type=str,
        default=None,
        help=(
            "Override Data/<dataset>/output namespace for all runs. "
            "Known long segments are compacted. Default: ae/j*/target-style aliases."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help=(
            "Override hidden dimension for all runs. If omitted, use best_hparams "
            "when available, otherwise the runner's dataset-specific default."
        ),
    )
    parser.add_argument("--num-timesteps-in", type=int, default=168)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--resume",
        type=str,
        default="auto",
        help=(
            "Checkpoint recovery mode written into every training command. "
            "Default: auto (resume the latest checkpoint for the same run label, "
            "or start fresh when none exists). Use none to force a clean restart."
        ),
    )
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=1,
        help="Save a resumable checkpoint every N epochs; set 0 to disable.",
    )
    parser.add_argument("--sim-type", type=str, default="cosine")
    parser.add_argument(
        "--graph-sparsity-policy",
        type=str,
        default=FINAL_GRAPH_SOURCE_POLICY,
        choices=[FINAL_GRAPH_SOURCE_POLICY],
        help="Graph sparsity policy passed to code/main.py for every run.",
    )
    parser.add_argument("--native-top-k", type=int, default=5)
    parser.add_argument("--deephgnn-hierarchical-loss-weight", type=float, default=1.0)
    parser.add_argument("--spectgnn-alpha", type=float, default=1.2)
    parser.add_argument("--spectgnn-degree", type=int, default=4)
    parser.add_argument("--spectgnn-modes", type=int, default=5)
    parser.add_argument("--spectgnn-trend-window", type=int, default=24)
    parser.add_argument("--static-threshold", type=float, default=None)
    parser.add_argument("--adaptive-top-k", type=int, default=None)
    parser.add_argument("--dynamic-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--horizons", type=str, default=None,
                        help="Override horizons, e.g. 1,6,24.")
    parser.add_argument(
        "--graph-modes",
        type=str,
        default=None,
        help="Override fixed graph candidates for this compatibility stage, e.g. H,HG.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Override model list for this stage, e.g. GCN-GRU-LP-NO,GCN-GRU-LP-HYBN.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default=None,
        choices=[
            "target",
            "target_calendar",
            "target_calendar_weather",
        ],
        help=(
            "Input feature tensor passed through to code/main.py. "
            "Defaults to target."
        ),
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation in code/main.py.")
    parser.add_argument("--plot-node-limit", type=int, default=None, help="Cap per-node prediction plots per run.")
    parser.add_argument(
        "--mlp-hidden-dims",
        type=str,
        default="128,64",
        help="Reconciliation MLP hidden sizes for BUN/TDN/HYBN, e.g. 64 or 256,128.",
    )
    parser.add_argument(
        "--mlp-dropout",
        type=float,
        default=0.2,
        help="Dropout inside nonlinear reconciliation MLP heads.",
    )
    parser.add_argument(
        "--mlp-layer-norm",
        type=lambda v: str(v).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="Whether nonlinear reconciliation MLP heads use LayerNorm.",
    )
    parser.add_argument(
        "--mlp-activation",
        type=str,
        default="relu",
        choices=["relu", "gelu", "tanh"],
        help="Activation for nonlinear reconciliation MLP heads.",
    )
    parser.add_argument(
        "--mixing-hidden-dims",
        type=str,
        default="64,32",
        help="HYBN mixing MLP hidden sizes.",
    )
    parser.add_argument(
        "--mixing-dropout",
        type=float,
        default=0.2,
        help="Dropout inside the HYBN mixing MLP.",
    )
    # Stage-linking
    parser.add_argument("--override-graph", type=str, default=None,
                        help="Override graph for Stages 2/3/5/6 with Stage 1 result.")
    parser.add_argument("--best-graph", type=str, default="H",
                        help="Best graph (Stage 4/6 proposed model).")
    parser.add_argument("--best-reconcile", type=str, default=None,
                        help="Best reconcile model from Stage 3 (e.g. GCN-GRU-LP-BUL).")
    parser.add_argument("--best-backbone", type=str, default=None,
                        help="Best backbone from Stage 2 (e.g. gcn-gru).")
    parser.add_argument(
        "--select-from-experiment-id",
        type=str,
        default=None,
        help=(
            "When Stage 2 reads Stage 1 or Stage 3 reads Stage 2, restrict automatic "
            "top-k selection to this source experiment_id."
        ),
    )
    parser.add_argument(
        "--selection-graph-sparsity-policy",
        type=str,
        default=FINAL_GRAPH_SOURCE_POLICY,
        help=(
            "Only use prior-stage runs with this graph_sparsity_policy when auto-selecting "
            "graphs/architectures."
        ),
    )
    parser.add_argument(
        "--architecture-graph-topk",
        type=int,
        default=2,
        help="Stage 2: keep top-k graphs from Stage 1 for graph-ST joint selection.",
    )
    parser.add_argument(
        "--architecture-graph-metric",
        type=str,
        default="val_loss",
        help=(
            "Stage 2 graph preselection metric read from Stage 1 model_info files. "
            "Default val_loss avoids test-set model selection."
        ),
    )
    parser.add_argument(
        "--architecture-include-graphs",
        type=str,
        default="H",
        help="Stage 2 graph preselection always includes these comma-separated reference graphs.",
    )
    parser.add_argument(
        "--architecture-fallback-graphs",
        type=str,
        default="H,HG",
        help="Stage 2 graph candidates used when Stage 1 results are unavailable.",
    )
    parser.add_argument(
        "--architecture-st-modes",
        type=str,
        default=",".join(ARCH_ST_MODES),
        help="Stage 2 ST interaction modes, e.g. sequential,alternating,hier_fusion.",
    )
    parser.add_argument(
        "--architecture-backbones",
        type=str,
        default=",".join(f"{g}-{t}" for g, t in ARCH_BACKBONES),
        help=(
            "Stage 2 backbones. Default uses the Applied Energy candidate backbones "
            "plus temporal-only controls. Supplementary candidates include: "
            + ",".join(f"{g}-{t}" for g, t in ARCH_BACKBONES_SUPPLEMENTARY)
        ),
    )
    parser.add_argument(
        "--graph-temporal-gnn-types",
        type=str,
        default="gcn",
        help=(
            "Stage graph_temporal graph module candidates. Priority lane uses gcn; "
            "supplementary can pass gatv2,graphsage,transformer."
        ),
    )
    parser.add_argument(
        "--graph-adapter-temporal-types",
        type=str,
        default=",".join(GRAPH_ADAPTER_TEMPORAL_TYPES),
        help="Temporal backbones for GRAPH_ADAPTER in graph_temporal/baseline stages.",
    )
    parser.add_argument(
        "--stgnn-graph-sources",
        type=str,
        default="project,native,hybrid",
        help=(
            "Stage stgnn graph sources for GWNET/MTGNN/AGCRN. DCRNN/STGCN always use project."
        ),
    )
    # Reconcile
    parser.add_argument("--reconcile-topk", type=int, default=2)
    parser.add_argument(
        "--reconcile-metric",
        type=str,
        default="val_loss",
        help="Stage 3 architecture preselection metric from Stage 2; default val_loss avoids test leakage.",
    )
    parser.add_argument("--reconcile-backbones", type=str, default=None)
    parser.add_argument(
        "--reconcile-architecture-topk",
        type=int,
        default=2,
        help="Stage 3: keep top-k graph/backbone/ST choices from Stage 2.",
    )
    parser.add_argument(
        "--reconcile-st-modes",
        type=str,
        default="sequential",
        help="Fallback ST modes when Stage 2 results are unavailable or --reconcile-backbones is used.",
    )
    # Tuning
    parser.add_argument("--tuning-seeds", type=str, default="42,43",
                        help="Seeds for tuning stage (fewer for speed).")
    # Best HPs produced by pick_best_hparams.py from Stage 0 results.
    parser.add_argument("--best-hparams", type=str,
                        default="scripts/best_hparams.json",
                        help="Path to best_hparams.json (per-dataset per-model lr/hidden).")
    parser.add_argument("--ignore-best-hparams", action="store_true",
                        help="Do not read best_hparams.json (use CLI args / defaults).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    preset = STAGE_PRESETS[args.stage]
    datasets = _parse_csv_list(args.datasets)
    seeds = _parse_int_list(args.seeds)
    override_horizons = _parse_int_list(args.horizons) if args.horizons else None
    dataset_horizons = {
        dataset: list(override_horizons or preset.horizons)
        for dataset in datasets
    }

    # --override-graph remains available for manual reruns. Without it, Stage 2
    # auto-selects Stage-1 top-k graphs and Stage 3 auto-selects Stage-2 top-k
    # architecture choices.
    effective_graph_modes = list(preset.graph_modes)
    if args.graph_modes:
        effective_graph_modes = [normalize_graph_mode(g) for g in _parse_csv_list(args.graph_modes)]
        invalid_graphs = [g for g in effective_graph_modes if g not in GRAPH_MODES_ALL]
        if invalid_graphs:
            raise ValueError(f"Invalid --graph-modes values: {invalid_graphs}")
        print(f"[INFO] Graph modes override: {effective_graph_modes}")
    if args.override_graph and args.stage in ("architecture", "reconcile", "ablation", "sensitivity"):
        effective_graph_modes = [normalize_graph_mode(args.override_graph)]
        print(f"[INFO] Graph mode override: {effective_graph_modes}")

    best_backbone = _parse_backbone_list(args.best_backbone) if args.best_backbone else None
    architecture_backbones = _parse_backbone_list(args.architecture_backbones)
    architecture_st_modes = _parse_st_modes(args.architecture_st_modes)
    reconcile_st_modes = _parse_st_modes(args.reconcile_st_modes)
    graph_temporal_gnn_types = [g.lower() for g in _parse_csv_list(args.graph_temporal_gnn_types)]
    invalid_graph_temporal_gnn = [g for g in graph_temporal_gnn_types if g not in GNN_TYPES or g == "none"]
    if invalid_graph_temporal_gnn:
        raise ValueError(f"Invalid --graph-temporal-gnn-types values: {invalid_graph_temporal_gnn}")
    graph_adapter_temporal_types = [t.lower() for t in _parse_csv_list(args.graph_adapter_temporal_types)]
    invalid_adapter_temporal = [t for t in graph_adapter_temporal_types if t not in GRAPH_ADAPTER_TEMPORAL_TYPES]
    if invalid_adapter_temporal:
        raise ValueError(f"Invalid --graph-adapter-temporal-types values: {invalid_adapter_temporal}")
    stgnn_graph_sources = [s.lower() for s in _parse_csv_list(args.stgnn_graph_sources)]
    invalid_sources = [s for s in stgnn_graph_sources if s not in STGNN_GRAPH_SOURCES]
    if invalid_sources:
        raise ValueError(f"Invalid --stgnn-graph-sources values: {invalid_sources}")
    selection_graph_sparsity_policy = str(args.selection_graph_sparsity_policy).strip() or None

    # Load per-(dataset, model) best HPs from Stage 0 (optional).
    best_hp = {}
    if not args.ignore_best_hparams:
        best_hp = _load_best_hparams(Path(args.best_hparams))
        if best_hp:
            print(f"[INFO] loaded best_hparams from {args.best_hparams} "
                  f"({sum(len(v) for v in best_hp.values())} entries)")
        else:
            print(f"[INFO] no best_hparams found at {args.best_hparams}; "
                  f"falling back to --lr and _default_hidden_dim().")

    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_root = Path(args.data_root).resolve()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.experiment_id = _safe_segment(args.experiment_id or run_id)
    args.paper_scope = _safe_segment(args.paper_scope)
    stage_label = _stage_label(args.stage)
    manifest_dir = (
        Path(args.manifest_dir).resolve()
        if args.manifest_dir
        else _paper_results_root(project_root, args.paper_scope) / "raw_manifests"
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / paper_manifest_filename(stage_label, args.experiment_id)
    selected_runs_dir = _paper_results_root(project_root, args.paper_scope) / "selected_runs"

    # === Stage 0: Tuning ===
    if args.stage == "tuning":
        runs = _build_tuning_runs(args, datasets)
        n_per_ds = len(TUNING_MODELS) * len(HP_GRID["lr"]) * len(HP_GRID["hidden_dim"]) * len(_parse_int_list(args.tuning_seeds))
        print(f"Stage=tuning | {len(datasets)} datasets × {n_per_ds} runs/dataset = {len(runs)} total")
        print(f"  Models: {[m[0] for m in TUNING_MODELS]}")
        print(f"  Grid: lr={HP_GRID['lr']}, hidden={HP_GRID['hidden_dim']}")
        print(f"  Seeds: {args.tuning_seeds}")
        for ds in datasets:
            print(f"  [{ds}]: {n_per_ds} runs")

        with manifest_path.open("w", encoding="utf-8") as f:
            for item in runs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Manifest: {manifest_path}")

        for i, run in enumerate(runs, start=1):
            print(f"[{i}/{len(runs)}] {run['dataset']} model={run['model_name']} "
                  f"lr={run['lr']} hidden={run['hidden_dim']} seed={run['seed']}")
            if args.dry_run:
                continue
            subprocess.run(run["cmd"], cwd=project_root, check=True)
        return

    # === Architecture graph cache (Stage 2) ===
    architecture_graph_cache: dict[tuple[str, int], list[str]] = {}
    if args.stage == "architecture":
        include_graphs = _parse_csv_list(args.architecture_include_graphs)
        fallback_graphs = [g.upper() for g in _parse_csv_list(args.architecture_fallback_graphs)]
        for ds in datasets:
            for h in dataset_horizons[ds]:
                if args.override_graph:
                    selected_graphs = list(effective_graph_modes)
                elif args.graph_modes:
                    selected_graphs = list(effective_graph_modes)
                else:
                    selected_graphs = _select_top_graphs_from_stage1(
                        data_root,
                        ds,
                        h,
                        metric_name=args.architecture_graph_metric,
                        topk=args.architecture_graph_topk,
                        paper_scope=args.paper_scope,
                        experiment_stage=STAGE_LABELS["graph"],
                        experiment_id=args.select_from_experiment_id,
                        include_graphs=include_graphs,
                        graph_sparsity_policy=selection_graph_sparsity_policy,
                    )
                    if not selected_graphs:
                        selected_graphs = fallback_graphs or list(effective_graph_modes)
                        print(
                            f"[WARN] No Stage-1 graph results for {ds} h={h}. "
                            f"Fallback graphs: {selected_graphs}"
                        )
                architecture_graph_cache[(ds, h)] = selected_graphs
                print(f"[INFO] Architecture graphs | {ds} h={h}: {selected_graphs}")
        if not args.dry_run:
            selected_runs_dir.mkdir(parents=True, exist_ok=True)
            selected_graph_path = selected_runs_dir / selected_runs_filename("stage1_graphs", stage_label, args.experiment_id)
            selected_graph_payload = {
                f"{ds}|h{h}": graphs
                for (ds, h), graphs in sorted(architecture_graph_cache.items())
            }
            selected_graph_path.write_text(
                json.dumps(selected_graph_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Selected Stage-1 graphs: {selected_graph_path}")

    # === Reconcile architecture cache (Stage 3) ===
    reconcile_arch_cache: dict[tuple[str, int], list[ArchitectureChoice]] = {}
    if args.stage == "reconcile":
        if args.reconcile_backbones:
            fixed_bb = _parse_backbone_list(args.reconcile_backbones)
            for ds in datasets:
                for h in dataset_horizons[ds]:
                    choices = [
                        ArchitectureChoice(gm, gnn, temp, st_mode)
                        for gm in effective_graph_modes
                        for gnn, temp in fixed_bb
                        for st_mode in reconcile_st_modes
                    ]
                    reconcile_arch_cache[(ds, h)] = choices
                    print(f"[INFO] Reconcile manual architectures | {ds} h={h}: {choices}")
        else:
            for ds in datasets:
                for h in dataset_horizons[ds]:
                    sel = _select_top_architectures_from_stage2(
                        data_root,
                        ds,
                        h,
                        metric_name=args.reconcile_metric,
                        topk=args.reconcile_architecture_topk,
                        paper_scope=args.paper_scope,
                        experiment_stage=STAGE_LABELS["architecture"],
                        experiment_id=args.select_from_experiment_id,
                        graph_sparsity_policy=selection_graph_sparsity_policy,
                    )
                    if not sel:
                        fallback_bb = best_backbone or preset.backbones
                        sel = [
                            ArchitectureChoice(gm, gnn, temp, st_mode)
                            for gm in effective_graph_modes
                            for gnn, temp in fallback_bb
                            for st_mode in reconcile_st_modes
                        ]
                        print(f"[WARN] No Stage-2 architecture results for {ds} h={h}. Fallback: {sel}")
                    reconcile_arch_cache[(ds, h)] = sel
                    print(f"[INFO] Reconcile architectures | {ds} h={h}: {sel}")
        if not args.dry_run:
            selected_runs_dir.mkdir(parents=True, exist_ok=True)
            selected_arch_path = selected_runs_dir / selected_runs_filename("stage2_architectures", stage_label, args.experiment_id)
            selected_arch_payload = {
                f"{ds}|h{h}": [choice.__dict__ for choice in choices]
                for (ds, h), choices in sorted(reconcile_arch_cache.items())
            }
            selected_arch_path.write_text(
                json.dumps(selected_arch_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"Selected Stage-2 architectures: {selected_arch_path}")

    # === Model list (handle --best-reconcile) ===
    model_names = list(preset.model_names)
    if args.models:
        model_names = [m.strip().upper() for m in _parse_csv_list(args.models)]
        print(f"[INFO] Model list override: {model_names}")
    if args.stage in ("baseline", "sensitivity") and args.best_reconcile:
        best_rec = args.best_reconcile.upper()
        if best_rec not in model_names:
            model_names.insert(0, best_rec)
        print(f"[INFO] Added proposed model: {best_rec}")

    # === Build runs ===
    runs = []
    for dataset in datasets:
        for horizon in dataset_horizons[dataset]:
            if args.stage == "reconcile":
                arch_choices = reconcile_arch_cache[(dataset, horizon)]
                for choice in arch_choices:
                    for model_name in model_names:
                        for seed in seeds:
                            hp_lr, hp_hidden = _lookup_hp(best_hp, dataset, model_name)
                            extra_args = ["--st-mode", choice.st_mode]
                            variant = _describe_extra_args(extra_args)
                            run_label = _make_run_label(
                                args.stage,
                                model_name,
                                choice.gnn_type,
                                choice.temporal_type,
                                choice.graph_mode,
                                horizon,
                                seed,
                                variant=variant,
                            )
                            cmd = _build_command(
                                args,
                                dataset,
                                seed,
                                choice.graph_mode,
                                model_name,
                                horizon,
                                choice.gnn_type,
                                choice.temporal_type,
                                extra_args=extra_args,
                                override_lr=hp_lr,
                                override_hidden=hp_hidden,
                                run_label=run_label,
                            )
                            runs.append({
                                "paper_scope": args.paper_scope,
                                "experiment_stage": stage_label,
                                "experiment_id": args.experiment_id,
                                "output_namespace": _output_namespace(args, stage_label, dataset),
                                "run_label": run_label,
                                "dataset": dataset,
                                **_feature_metadata(args, dataset),
                                "graph_mode": choice.graph_mode,
                                "model_name": model_name,
                                "model_family": _model_family(model_name),
                                "gnn_type": choice.gnn_type,
                                "temporal_type": choice.temporal_type,
                                "st_mode": choice.st_mode,
                                "seed": seed,
                                "horizon": horizon,
                                "lr": hp_lr if hp_lr is not None else args.lr,
                                "hidden_dim": _resolved_hidden_dim(args, dataset, hp_hidden),
                                "extra_args": extra_args,
                                "variant": variant,
                                **_graph_sparsity_metadata(args),
                                "cmd": cmd,
                            })
                continue

            graph_modes_for_stage = (
                architecture_graph_cache[(dataset, horizon)]
                if args.stage == "architecture"
                else effective_graph_modes
            )
            for graph_mode in graph_modes_for_stage:
                for model_name in model_names:
                    if model_name == "DEEPHGNN_SPECTGNN" and graph_mode != "H":
                        continue
                    run_graph = _resolve_graph_for_model(
                        args.stage, model_name, args.best_graph, graph_mode,
                    )
                    if (
                        best_backbone
                        and args.stage in ("baseline", "ablation", "sensitivity")
                        and model_name.startswith("GCN-GRU-LP-")
                    ):
                        stage_bb = best_backbone
                    elif args.stage == "architecture":
                        stage_bb = architecture_backbones
                    elif args.stage == "graph_temporal" and model_name in {"GRAPH_ADAPTER", "LAGTCN"}:
                        stage_bb = [(g, temp) for g in graph_temporal_gnn_types for temp in graph_adapter_temporal_types]
                    elif args.stage == "graph_temporal":
                        stage_bb = [(g, "gru") for g in graph_temporal_gnn_types]
                    else:
                        stage_bb = preset.backbones
                    stage_bb = _resolve_stage_backbones(args.stage, model_name, stage_bb)
                    if args.stage == "architecture" and graph_mode != "H":
                        stage_bb = [(gnn, temp) for gnn, temp in stage_bb if _uses_graph_encoder(gnn)]
                        if not stage_bb:
                            continue

                    for gnn, temp in stage_bb:
                        if args.stage == "architecture":
                            modes_for_backbone = (
                                architecture_st_modes if _uses_graph_encoder(gnn) else ["sequential"]
                            )
                            extra_args_variants = [["--st-mode", mode] for mode in modes_for_backbone]
                        elif args.stage == "stgnn":
                            extra_args_variants = []
                            for source in _stgnn_sources_for_model(model_name, stgnn_graph_sources):
                                if source == "native" and graph_mode != effective_graph_modes[0]:
                                    continue
                                extra_args_variants.append(["--stgnn-graph-source", source])
                        else:
                            extra_args_variants = preset.extra_args if preset.extra_args else [[]]
                        for ea in extra_args_variants:
                            for seed in seeds:
                                hp_lr, hp_hidden = _lookup_hp(best_hp, dataset, model_name)
                                variant = _describe_extra_args(ea) if ea else None
                                run_label = _make_run_label(
                                    args.stage,
                                    model_name,
                                    gnn,
                                    temp,
                                    run_graph,
                                    horizon,
                                    seed,
                                    variant=variant,
                                )
                                cmd = _build_command(
                                    args, dataset, seed, run_graph, model_name,
                                    horizon, gnn, temp,
                                    extra_args=ea if ea else None,
                                    override_lr=hp_lr,
                                    override_hidden=hp_hidden,
                                    run_label=run_label,
                                )
                                ri = {
                                    "paper_scope": args.paper_scope,
                                    "experiment_stage": stage_label,
                                    "experiment_id": args.experiment_id,
                                    "output_namespace": _output_namespace(args, stage_label, dataset),
                                    "run_label": run_label,
                                    "dataset": dataset,
                                    **_feature_metadata(args, dataset),
                                    "graph_mode": run_graph,
                                    "model_name": model_name,
                                    "model_family": _model_family(model_name),
                                    "gnn_type": gnn,
                                    "temporal_type": temp,
                                    "st_mode": _st_mode_from_extra_args(ea),
                                    "stgnn_graph_source": _stgnn_graph_source_from_extra_args(ea),
                                    "seed": seed,
                                    "horizon": horizon,
                                    "lr": hp_lr if hp_lr is not None else args.lr,
                                    "hidden_dim": _resolved_hidden_dim(args, dataset, hp_hidden),
                                    **_graph_sparsity_metadata(args),
                                    "cmd": cmd,
                                }
                                if ea:
                                    ri["extra_args"] = ea
                                    ri["variant"] = _describe_extra_args(ea)
                                runs.append(ri)

    for run in runs:
        if str(run.get("model_name", "")).upper() == "LAGTCN":
            run["lagtcn_graph_source_version"] = LAGTCN_GRAPH_SOURCE_VERSION_CURRENT

    # === Summary ===
    print(f"\nStage={args.stage} | planned runs={len(runs)}")
    models_set = sorted(set(r["model_name"] for r in runs))
    graphs_set = sorted(set(r["graph_mode"] for r in runs))
    families_set = sorted(set(r.get("model_family", "other") for r in runs))
    st_modes_set = sorted(set(r.get("st_mode", "sequential") for r in runs))
    print(f"  Models ({len(models_set)}): {models_set}")
    print(f"  Model families: {families_set}")
    print(f"  Graphs: {graphs_set}")
    print(f"  Graph sparsity: {args.graph_sparsity_policy}")
    if args.stage in ("architecture", "reconcile", "ablation"):
        print(f"  ST modes: {st_modes_set}")
    unique_horizon_sets = {tuple(v) for v in dataset_horizons.values()}
    if len(unique_horizon_sets) == 1:
        print(f"  Horizons: {next(iter(unique_horizon_sets))}")
    else:
        print(f"  Horizons by dataset: {dataset_horizons}")
    print(f"  Datasets: {datasets}")
    if args.stage == "ablation":
        variants = sorted(set(r.get("variant", "default") for r in runs))
        print(f"  Ablation variants ({len(variants)}): {variants}")

    with manifest_path.open("w", encoding="utf-8") as f:
        for item in runs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Manifest: {manifest_path}")

    for i, run in enumerate(runs, start=1):
        v = f" variant={run['variant']}" if "variant" in run else ""
        print(f"[{i}/{len(runs)}] {run['dataset']} graph={run['graph_mode']} "
              f"model={run['model_name']} bb={run['gnn_type']}-{run['temporal_type']} "
              f"st={run.get('st_mode', 'sequential')} h={run['horizon']} seed={run['seed']}{v}")
        if args.dry_run:
            continue
        subprocess.run(run["cmd"], cwd=project_root, check=True)


if __name__ == "__main__":
    main()
