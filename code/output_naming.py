"""Short, stable names for experiment output artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


MODEL_ALIASES = {
    "GCN-GRU-LP-NO": "gcnlpNO",
    "GCN-GRU-LP-BUD": "bud",
    "GCN-GRU-LP-BUL": "bul",
    "GCN-GRU-LP-BUN": "bun",
    "GCN-GRU-LP-TDD": "tdd",
    "GCN-GRU-LP-TDL": "tdl",
    "GCN-GRU-LP-TDN": "tdn",
    "GCN-GRU-LP-HYBD": "hybd",
    "GCN-GRU-LP-HYBL": "hybl",
    "GCN-GRU-LP-HYBN": "hybn",
    "GRAPH_DLINEAR": "gdlin",
    "GRAPH_PATCHTST": "gptst",
    "GRAPH_ITRANSFORMER": "gitrans",
    "GRAPH_ADAPTER": "gadapt",
    "LAGTCN": "lagtcn",
    "DEEPHGNN_SPECTGNN": "deephgnn",
    "DLINEAR": "dlin",
    "PATCHTST": "ptst",
    "NHITS": "nhits",
    "TIMESNET": "tnet",
    "ITRANSFORMER": "itrans",
}

STAGE_ALIASES = {
    "stage_0_tuning": "j0",
    "stage_1_graph": "j2",
    "stage_2_architecture": "j3a",
    "stage_3_reconcile": "j4b",
    "stage_4_baseline": "j5base",
    "stage_5_ablation": "j5abl",
    "stage_6_sensitivity": "j7",
    "stage_a_temporal": "j1",
    "stage_b_stgnn": "j3c",
    "stage_c_graph_temporal": "j3b",
    "j3a_arch": "j3a",
    "j3b_gtemp": "j3b",
    "j3c_stgnn": "j3c",
    "tuning": "j0",
    "graph": "j2",
    "architecture": "j3a",
    "reconcile": "j4b",
    "baseline": "j5base",
    "ablation": "j5abl",
    "sensitivity": "j7",
    "temporal": "j1",
    "stgnn": "j3c",
    "graph_temporal": "j3b",
    "legacy_base_matrix": "base",
    "step0_base_selection": "s0base",
    "step0_h_only_graph_temporal_screen": "s0h",
    "step0_rsf_base_lock": "s0rsfbase",
    "stage0_graph_temporal_enhancement": "s0gte",
    "stage0_gef_priority_graph_temporal": "s0gef",
    "stage0_rsf_priority_graph_temporal": "s0rsf",
    "stage0_native_graph_temporal": "s0nat",
    "stage0_strong_stgnn_baselines": "s0stgnn",
}

GRAPH_ALIASES = {
    "I": "I",
    "H": "H",
    "HG": "HG",
    "S": "S",
    "A": "A",
    "D": "D",
    "S+A+D": "SAD",
    "H+S": "HS",
    "H+A": "HA",
    "H+D": "HD",
    "H+S+A+D": "HSAD",
}

GRAPH_SUBDIR_ALIASES = {
    "I": "I",
    "H": "H",
    "HG": "HG",
    "S": "S",
    "A": "A",
    "D": "D",
    "S+A+D": "SAD",
    "H+S": "HS",
    "H+A": "HA",
    "H+D": "HD",
    "H+S+A+D": "HSAD",
}

ST_MODE_ALIASES = {
    "sequential": "seq",
    "alternating": "alt",
    "hier_fusion": "hf",
}

STGNN_SOURCE_ALIASES = {
    "project": "proj",
    "native": "nat",
    "hybrid": "hyb",
}

DATASET_ALIASES = {
    "GEFCom2012_2level": "2l",
    "GEFCom2017QualifyingMatch_3level": "3l",
    "GEFCom2017FinalMatch_4level": "4l",
    "2level": "2l",
    "3level": "3l",
    "4level": "4l",
    "2l3l": "2l3l",
    "FinalMatch": "4l",
    "QualifyingMatch": "3l",
}

PAPER_SCOPE_ALIASES = {
    "project": "proj",
    "proj": "proj",
    "journal_applied_energy": "ae",
    "journal/applied_energy": "ae",
    "applied_energy": "ae",
    "ae": "ae",
}

FEATURE_ALIASES = {
    "without_exog": "target",
    "target": "target",
    "target_calendar": "cal",
    "calendar": "cal",
    "target_calendar_weather": "calwx",
}

NAMESPACE_SEGMENT_ALIASES = {
    **PAPER_SCOPE_ALIASES,
    **FEATURE_ALIASES,
    "hierarchy": "H",
    "hierarchy_graph": "H",
    "hierarchy_enhanced": "HG",
    "hierarchy_static": "HS",
    "hierarchy_adaptive": "HA",
    "hierarchy_dynamic": "HD",
    "hierarchy_static_adaptive_dynamic": "HSAD",
    "static": "S",
    "similarity": "S",
    "adaptive": "A",
    "dynamic": "D",
    "static_adaptive_dynamic": "SAD",
}

PAPER_FILENAME_MAX_LEN = 56
RUN_LABEL_MAX_LEN = 48

_PAPER_SLUG_DROP_TOKENS = {
    "ae",
    "journal",
    "applied",
    "energy",
    "manifest",
    "mahuika",
    "scott",
    "graph",
    "h100",
    "a100",
    "l4",
    "gpu",
    "gpu1",
    "cuda",
    "48h",
}

_PAPER_SLUG_TOKEN_REPLACEMENTS = {
    "seed42": "s42",
    "seed43": "s43",
    "priority": "prio",
    "finalmatch": "4l",
    "qualifyingmatch": "3l",
    "2level": "2l",
    "3level": "3l",
    "4level": "4l",
}

ARTIFACT_FILENAMES = {
    "config": "config.json",
    "training_results": "train.json",
    "best_model": "best_model.pth",
    "last_checkpoint": "last_checkpoint.pth",
    "metrics": "metrics.json",
    "classical_reconcile_metrics": "classical_rec.json",
    "graph_edges": "graph_edges.csv",
    "graph_info": "graph_info.json",
    "level_metrics": "level_metrics.csv",
    "model_info": "model_info.json",
    "loss_curve": "loss.png",
}

# Canonical prediction artifact filenames, used directly by all writers and
# readers. pred.csv is the model's saved forecast (the base forecast for
# base-only models); neural-reconciliation runs additionally save the
# pre-reconciliation forecast as base_pred.csv and the reconciled forecast as
# rec_pred.csv. true.csv always holds the matching targets.
PRED_FILENAME = "pred.csv"
BASE_PRED_FILENAME = "base_pred.csv"
REC_PRED_FILENAME = "rec_pred.csv"
TRUE_FILENAME = "true.csv"


def safe_segment(value: str) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._+-]+", "-", text)
    text = text.strip("-._")
    return text or "default"


def compact_timestamp(timestamp: str) -> str:
    text = str(timestamp)
    match = re.fullmatch(r"20(\d{6})_(\d{6})", text)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return safe_segment(text)


def model_alias(model_name: str) -> str:
    model = str(model_name)
    return MODEL_ALIASES.get(model, safe_segment(model).lower())


def normalize_graph_mode(graph_mode: str) -> str:
    graph = str(graph_mode).replace(" ", "").upper()
    token_aliases = {
        "NO": "I",
        "NONE": "I",
        "NOGRAPH": "I",
        "IDENTITY": "I",
        "SELF": "I",
        "SELFLOOP": "I",
        "HIER": "H",
        "HIERARCHY": "H",
        "SIM": "S",
        "STATIC": "S",
        "STATICSIM": "S",
        "SIMILARITY": "S",
        "ADA": "A",
        "ADAPTIVE": "A",
        "DYN": "D",
        "DYNAMIC": "D",
    }
    if "+" not in graph:
        return token_aliases.get(graph, graph)
    tokens = []
    for token in graph.split("+"):
        if not token:
            continue
        tokens.append(token_aliases.get(token, token))
    ordered = []
    for token in ("H", "S", "A", "D"):
        if token in tokens and token not in ordered:
            ordered.append(token)
    extras = [token for token in tokens if token not in {"H", "S", "A", "D"} and token not in ordered]
    return "+".join(ordered + extras)


def graph_components(graph_mode: str) -> set[str]:
    graph = normalize_graph_mode(graph_mode)
    if graph == "I":
        return set()
    if graph == "HG":
        return {"HG"}
    return {token for token in graph.split("+") if token}


LAGTCN_INFORMATIVE_GRAPH_SOURCES = ("hierarchy", "similarity", "adaptive", "dynamic")
LAGTCN_GRAPH_SOURCE_VERSION_CURRENT = "threshold_topk_hanchor_residual_v4"
LAGTCN_GRAPH_SOURCE_VERSION_NOT_APPLICABLE = "not_applicable"


def normalize_lagtcn_graph_source_version(
    model_name: str | None,
    version: str | None,
) -> str:
    """Validate and return the current explicit graph-source version label."""
    base_model = str(model_name or "").upper().split("+", 1)[0]
    if base_model != "LAGTCN":
        return LAGTCN_GRAPH_SOURCE_VERSION_NOT_APPLICABLE
    value = str(version or "").strip()
    if value != LAGTCN_GRAPH_SOURCE_VERSION_CURRENT:
        raise ValueError(
            "LAGTCN artifacts must explicitly use graph-source version "
            f"{LAGTCN_GRAPH_SOURCE_VERSION_CURRENT!r}; got {value or None!r}."
        )
    return value


def lagtcn_graph_sources(graph_mode: str) -> tuple[str, ...]:
    """Return the independent runtime graph sources used by LAGTCN.

    Identity is a dedicated no-cross-node-edge anchor.  It is never mixed into
    configurations that contain one or more informative graph sources.
    """
    graph = normalize_graph_mode(graph_mode)
    if graph == "I":
        return ("identity",)
    if graph == "HG":
        return ("hierarchy",)

    components = graph_components(graph)
    source_by_token = {
        "H": "hierarchy",
        "S": "similarity",
        "A": "adaptive",
        "D": "dynamic",
    }
    sources = tuple(
        source_by_token[token]
        for token in ("H", "S", "A", "D")
        if token in components
    )
    if not sources:
        raise ValueError(f"Graph mode {graph_mode!r} does not define a LAGTCN graph source.")
    return sources


def graph_alias(graph_mode: str) -> str:
    graph = normalize_graph_mode(graph_mode)
    return GRAPH_ALIASES.get(graph, safe_segment(graph.replace("+", "")))


def graph_subdir(graph_mode: str, sim_type: str = "cosine") -> str:
    graph = normalize_graph_mode(graph_mode)
    alias = GRAPH_SUBDIR_ALIASES.get(graph, safe_segment(graph.replace("+", "_").lower()))
    if "S" in graph_components(graph):
        sim = safe_segment(str(sim_type).lower())[:3]
        return f"{alias}_{sim}"
    return alias


def stage_alias(stage: str) -> str:
    return STAGE_ALIASES.get(str(stage), safe_segment(stage).lower())


def paper_scope_alias(scope: str) -> str:
    text = str(scope).strip().replace(chr(92), "/").strip("/")
    key = text.lower()
    return PAPER_SCOPE_ALIASES.get(key, safe_segment(text).lower())


def feature_alias(feature_tag: str) -> str:
    key = str(feature_tag).strip().lower()
    return FEATURE_ALIASES.get(key, safe_segment(feature_tag).lower())


def compact_output_namespace(paper_scope: str, experiment_stage: str, feature_tag: str) -> str:
    return "/".join(
        [
            paper_scope_alias(paper_scope),
            stage_alias(experiment_stage),
            feature_alias(feature_tag),
        ]
    )


def compact_namespace(namespace: str) -> str:
    text = str(namespace).strip().replace(chr(92), "/").strip("/")
    if not text:
        return "default"
    parts: list[str] = []
    for raw_part in text.split("/"):
        if not raw_part:
            continue
        key = raw_part.strip().lower()
        alias = (
            NAMESPACE_SEGMENT_ALIASES.get(key)
            or STAGE_ALIASES.get(raw_part)
            or STAGE_ALIASES.get(key)
        )
        if alias is None:
            alias = safe_segment(raw_part).lower()
        parts.append(alias)
    return "/".join(parts) or "default"


def namespace_matches(namespace: str | None, prefix: str | None) -> bool:
    if not prefix:
        return True
    raw_namespace = str(namespace or "").strip().replace(chr(92), "/").strip("/")
    raw_prefix = str(prefix or "").strip().replace(chr(92), "/").strip("/")
    if not raw_namespace:
        return False
    if raw_namespace.startswith(raw_prefix):
        return True
    return compact_namespace(raw_namespace).startswith(compact_namespace(raw_prefix))


def _fit_stem(stem: str, max_len: int) -> str:
    text = safe_segment(stem)
    if len(text) <= max_len:
        return text
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    keep = max(8, max_len - len(digest) - 1)
    return f"{text[:keep].rstrip('_-.')}_{digest}"


def _normalise_paper_slug_text(value: str) -> str:
    text = safe_segment(value)
    replacements = {
        "journal_applied_energy": "ae",
        "stage_0_tuning": "j0",
        "stage_1_graph": "j2",
        "stage_2_architecture": "j3a",
        "stage_3_reconcile": "j4b",
        "stage_4_baseline": "j5base",
        "stage_5_ablation": "j5abl",
        "stage_6_sensitivity": "j7",
        "stage_a_temporal": "j1",
        "stage_b_stgnn": "j3c",
        "stage_c_graph_temporal": "j3b",
    }
    for old, new in replacements.items():
        text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
    for dataset, alias in DATASET_ALIASES.items():
        text = re.sub(re.escape(dataset), alias, text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=_)|^)h(\d+)[_-](\d+)(?=_|$)", r"h\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:(?<=_)|^)s(\d+)[_-](\d+)(?=_|$)", r"s\1-\2", text, flags=re.IGNORECASE)
    text = re.sub(r"\bseed(\d+)\b", r"s\1", text, flags=re.IGNORECASE)
    return text


def short_paper_slug(stage: str, experiment_id: str, max_len: int = 56) -> str:
    """Return a compact slug for paper-facing manifests, summaries, and pointers."""
    stage_short = stage_alias(stage)
    text = _normalise_paper_slug_text(experiment_id)
    tokens: list[str] = []
    seen: set[str] = set()
    for raw_token in re.split(r"_+", text):
        if not raw_token:
            continue
        token = _PAPER_SLUG_TOKEN_REPLACEMENTS.get(raw_token.lower(), raw_token)
        token_key = token.lower()
        if token_key in _PAPER_SLUG_DROP_TOKENS:
            continue
        if token_key == stage_short.lower() and tokens:
            continue
        if token_key in seen and not re.fullmatch(r"\d{8}", token):
            continue
        tokens.append(token)
        seen.add(token_key)

    if not tokens or tokens[0].lower() != stage_short.lower():
        tokens.insert(0, stage_short)
    return _fit_stem("_".join(tokens), max_len)


def paper_manifest_filename(stage: str, experiment_id: str, max_len: int = PAPER_FILENAME_MAX_LEN) -> str:
    suffix = ".jsonl"
    stem = f"manifest_{short_paper_slug(stage, experiment_id)}"
    return f"{_fit_stem(stem, max_len - len(suffix))}{suffix}"


def selected_runs_filename(kind: str, stage: str, experiment_id: str, max_len: int = PAPER_FILENAME_MAX_LEN) -> str:
    suffix = ".json"
    kind_aliases = {
        "stage1_graphs": "selected_j2graphs",
        "stage2_architectures": "selected_j3arch",
    }
    stem_prefix = kind_aliases.get(kind, f"selected_{safe_segment(kind).lower()}")
    slug_budget = max(12, max_len - len(stem_prefix) - 1 - len(suffix))
    stem = f"{stem_prefix}_{short_paper_slug(stage, experiment_id, max_len=slug_budget)}"
    return f"{_fit_stem(stem, max_len - len(suffix))}{suffix}"


def st_mode_alias(st_mode: str | None) -> str | None:
    if not st_mode:
        return None
    return ST_MODE_ALIASES.get(str(st_mode), safe_segment(st_mode).lower())


def stgnn_source_alias(source: str | None) -> str | None:
    if not source:
        return None
    return STGNN_SOURCE_ALIASES.get(str(source), safe_segment(source).lower())


def short_run_label(
    *,
    stage: str,
    model_name: str,
    gnn_type: str,
    temporal_type: str,
    graph_mode: str,
    horizon: int,
    seed: int,
    st_mode: str | None = None,
    stgnn_graph_source: str | None = None,
    variant: str | None = None,
) -> str:
    parts = [
        stage_alias(stage),
        model_alias(model_name),
        safe_segment(f"{gnn_type}-{temporal_type}").lower(),
        graph_alias(graph_mode),
        f"h{int(horizon)}",
        f"s{int(seed)}",
    ]
    st_alias = st_mode_alias(st_mode)
    if st_alias:
        parts.append(st_alias)
    source_alias = stgnn_source_alias(stgnn_graph_source)
    if source_alias and source_alias != "hyb":
        parts.append(source_alias)
    if variant:
        parts.append(safe_segment(variant).lower())
    return _fit_stem("_".join(parts), RUN_LABEL_MAX_LEN)


def shorten_existing_run_label(label: str, max_len: int = RUN_LABEL_MAX_LEN) -> str:
    """Compress an externally supplied run label without changing its meaning too much."""
    text = safe_segment(label)
    replacements = {
        "stage_2_architecture": "j3a",
        "stage_c_graph_temporal": "j3b",
        "stage_b_stgnn": "j3c",
        "stage_1_graph": "j2",
        "stage_a_temporal": "j1",
        "GCN-GRU-LP-NO": "gcnlpNO",
        "GRAPH_DLINEAR": "gdlin",
        "GRAPH_PATCHTST": "gptst",
        "GRAPH_ITRANSFORMER": "gitrans",
        "GRAPH_ADAPTER": "gadapt",
        "LAGTCN": "lagtcn",
        "DEEPHGNN_SPECTGNN": "deephgnn",
        "graphsrc-project": "proj",
        "graphsrc-native": "nat",
        "st-sequential": "seq",
        "st-alternating": "alt",
        "st-hier_fusion": "hf",
        "H-A": "HA",
        "H-S": "HS",
        "H-D": "HD",
        "S-A-D": "SAD",
        "H-S-A-D": "HSAD",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"_+", "_", text).strip("_")
    if len(text) <= max_len:
        return text
    # Preserve uniqueness when the distinguishing variant is near the end of
    # a long label. Plain prefix truncation can otherwise make different
    # hyperparameter candidates share one output/checkpoint lookup prefix.
    return _fit_stem(text, max_len)


def artifact_filename(kind: str) -> str:
    return ARTIFACT_FILENAMES[kind]


def legacy_artifact_patterns(kind: str, model_name: str, timestamp: str) -> list[str]:
    old_stem = f"{model_name}_{timestamp}"
    patterns = {
        "best_model": [f"best_model_{old_stem}.pth", "best_model_*.pth"],
        "last_checkpoint": [f"last_checkpoint_{old_stem}.pth", "last_checkpoint_*.pth"],
        "model_info": [f"model_info_{old_stem}.json", "model_info_*.json"],
        "predictions": [f"predictions_{old_stem}.csv", "predictions_*.csv"],
        "true_values": [f"true_values_{old_stem}.csv", "true_values_*.csv"],
    }
    return patterns.get(kind, [])


_PREDICTION_ARTIFACT_KINDS = {
    "predictions": PRED_FILENAME,
    "base_predictions": BASE_PRED_FILENAME,
    "reconciled_predictions": REC_PRED_FILENAME,
    "true_values": TRUE_FILENAME,
}


def find_artifact(output_dir: str | Path, kind: str, model_name: str, timestamp: str) -> Path | None:
    root = Path(output_dir)
    short_name = ARTIFACT_FILENAMES.get(kind) or _PREDICTION_ARTIFACT_KINDS.get(kind)
    if short_name:
        path = root / short_name
        if path.exists():
            return path
    for pattern in legacy_artifact_patterns(kind, model_name, timestamp):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[-1]
    return None
