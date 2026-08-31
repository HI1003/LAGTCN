#!/usr/bin/env python3
"""Aggregate experiment outputs into AE-ready summary tables (no extra deps)."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from output_naming import namespace_matches, normalize_lagtcn_graph_source_version
from graph_sparsity import FINAL_GRAPH_SOURCE_POLICY


METRIC_COLUMNS = [
    "MAE",
    "RMSE",
    "MAPE",
    "WAPE",
    "MASE",
    "coherency_mae",
    "coherency_rmse",
    "coherency_nmae_pct",
]

EFFICIENCY_COLUMNS = [
    "params_total",
    "params_trainable",
    "train_time_sec",
    "train_peak_gpu_mem_mb",
    "infer_latency_ms_per_batch",
    "infer_latency_ms_per_sample",
    "infer_throughput_samples_per_sec",
    "infer_peak_gpu_mem_mb",
]

STRONG_TEMPORAL_MODELS = {"DLINEAR", "PATCHTST", "NHITS", "TIMESNET"}
EXTRA_TEMPORAL_MODELS = {"ITRANSFORMER"}
GRAPH_NATIVE_TEMPORAL_MODELS = {"GRAPH_DLINEAR", "GRAPH_PATCHTST", "GRAPH_ITRANSFORMER", "GRAPH_ADAPTER", "LAGTCN"}
DEDICATED_GRAPH_MODELS = {"DCRNN", "STGCN", "GWNET", "MTGNN", "AGCRN"}

DENSITY_COLUMNS = [
    "graph_sparsity_policy",
    "graph_protocol_version",
    "graph_design_protocol_version",
    "hierarchy_density",
    "physical_hierarchy_density",
    "static_threshold",
    "adaptive_top_k",
    "dynamic_threshold",
    "static_component_density_actual",
    "base_graph_density_actual",
]

GROUP_FIELDS = [
    "paper_scope",
    "experiment_stage",
    "model_family",
    "dataset",
    "num_timesteps_out",
    "graph_sparsity_policy",
    "graph_design_protocol_version",
    "static_threshold",
    "adaptive_top_k",
    "dynamic_threshold",
    "graph_mode",
    "lagtcn_graph_source_version",
    "stgnn_graph_source",
    "model_name",
    "gnn_type",
    "temporal_type",
    "st_mode",
]


def _group_key(record: dict, fields: list[str]) -> tuple:
    values = []
    for field in fields:
        if field == "lagtcn_graph_source_version":
            model_name = record.get("base_model") or record.get("model_name")
            values.append(
                normalize_lagtcn_graph_source_version(
                    model_name,
                    record.get(field),
                )
            )
        else:
            values.append(record.get(field))
    return tuple(values)


def _iter_model_info_files(root: Path):
    seen = set()
    for pattern in ("**/model_info.json", "**/model_info_*.json"):
        for path in sorted(root.glob(pattern)):
            if path not in seen:
                seen.add(path)
                yield path


def _model_family(model_name: str | None, source: str | None = None) -> str:
    if source == "posthoc":
        return "posthoc-reconciliation"
    model = str(model_name or "").upper()
    base_model = model.split("+", 1)[0]
    if base_model in STRONG_TEMPORAL_MODELS | EXTRA_TEMPORAL_MODELS:
        return "temporal-only"
    if base_model in GRAPH_NATIVE_TEMPORAL_MODELS:
        return "graph-enhanced-temporal"
    if base_model in DEDICATED_GRAPH_MODELS:
        return "dedicated-STGNN"
    if base_model == "GCN-GRU-LP-NO":
        return "GCN-GRU reference"
    if base_model.startswith("GCN-GRU-LP-"):
        return "GCN-GRU reconciliation"
    return "other"


def _base_model_family(rec: dict) -> str:
    return str(rec.get("base_model_family") or _model_family(rec.get("base_model")))


def _is_graph_based_family(family: str) -> bool:
    return family in {
        "graph-enhanced-temporal",
        "dedicated-STGNN",
        "GCN-GRU reference",
        "GCN-GRU reconciliation",
    }


def _is_graph_based_record(rec: dict) -> bool:
    family = str(rec.get("model_family") or _model_family(rec.get("model_name")))
    if family in {"posthoc-reconciliation", "neural-reconciliation"}:
        return _is_graph_based_family(_base_model_family(rec))
    return _is_graph_based_family(family)


def _density_metadata_from_config(cfg: dict) -> dict:
    return {
        "graph_sparsity_policy": cfg.get("graph_sparsity_policy"),
        "graph_protocol_version": cfg.get("graph_protocol_version"),
        "graph_design_protocol_version": cfg.get("graph_design_protocol_version"),
        "hierarchy_density": _to_float(cfg.get("hierarchy_density")),
        "static_threshold": _to_float(cfg.get("static_threshold")),
        "adaptive_top_k": _to_float(cfg.get("adaptive_top_k")),
        "dynamic_threshold": _to_float(cfg.get("dynamic_threshold")),
        "static_component_density_actual": _to_float(cfg.get("static_component_density_actual")),
        "base_graph_density_actual": _to_float(cfg.get("base_graph_density_actual")),
    }


def _density_metadata_from_row(row: dict) -> dict:
    return {
        key: (
            row.get(key)
            if key in {"graph_sparsity_policy", "graph_protocol_version", "graph_design_protocol_version"}
            else _to_float(row.get(key))
        )
        for key in DENSITY_COLUMNS
    }


def _parse_int_list(value: str) -> list[int]:
    items = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return items


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mu = _mean(values)
    var = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _two_sided_sign_test_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail_prob = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail_prob)


def _one_sided_sign_test_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    # H1: model is better than baseline => wins is large.
    return sum(math.comb(n, i) for i in range(wins, n + 1)) / (2 ** n)


def _find_model_info_files(data_root: Path) -> list[Path]:
    return sorted(_iter_model_info_files(data_root))




def _load_record(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cfg = payload.get("config", {})
    metrics = payload.get("metrics", {})
    training = payload.get("training_results", {})
    params = payload.get("params", {})
    raw_data_dir = cfg.get("raw_data_dir", "")
    dataset = Path(raw_data_dir).name if raw_data_dir else ""

    record = {
        "dataset": dataset,
        "paper_scope": cfg.get("paper_scope"),
        "experiment_stage": cfg.get("experiment_stage"),
        "experiment_id": cfg.get("experiment_id"),
        "output_namespace": cfg.get("output_namespace"),
        "run_label": cfg.get("run_label"),
        "seed": cfg.get("seed"),
        "graph_mode": cfg.get("graph_mode"),
        "lagtcn_graph_source_version": normalize_lagtcn_graph_source_version(
            cfg.get("model_name"),
            cfg.get("lagtcn_graph_source_version"),
        ),
        "sim_type": cfg.get("sim_type"),
        "model_name": cfg.get("model_name"),
        "model_family": _model_family(cfg.get("model_name")),
        "base_model": cfg.get("base_model"),
        "base_model_family": _model_family(cfg.get("base_model")) if cfg.get("base_model") else None,
        "reconcile_method": cfg.get("reconcile_method"),
        "gnn_type": cfg.get("gnn_type", "gcn"),
        "temporal_type": cfg.get("temporal_type", "gru"),
        "st_mode": cfg.get("st_mode", "sequential"),
        "stgnn_graph_source": cfg.get("stgnn_graph_source"),
        "num_timesteps_in": cfg.get("num_timesteps_in"),
        "num_timesteps_out": cfg.get("num_timesteps_out"),
        "feature_set": cfg.get("feature_set"),
        "feature_tag": cfg.get("feature_tag"),
        "input_dim": cfg.get("input_dim"),
        "value_file": cfg.get("value_file"),
        "output_dir": cfg.get("output_dir"),
        "model_info_path": str(path),
        "params_total": _to_float(params.get("total")),
        "params_trainable": _to_float(params.get("trainable")),
        "train_time_sec": _to_float(training.get("train_time_sec", training.get("train_time"))),
        "train_peak_gpu_mem_mb": _to_float(training.get("train_peak_gpu_mem_mb")),
        "infer_latency_ms_per_batch": _to_float(metrics.get("infer_latency_ms_per_batch")),
        "infer_latency_ms_per_sample": _to_float(metrics.get("infer_latency_ms_per_sample")),
        "infer_throughput_samples_per_sec": _to_float(metrics.get("infer_throughput_samples_per_sec")),
        "infer_peak_gpu_mem_mb": _to_float(metrics.get("infer_peak_gpu_mem_mb")),
        "source": "model_info",
        **_density_metadata_from_config(cfg),
    }
    for key in METRIC_COLUMNS:
        record[key] = _to_float(metrics.get(key))
    return record




def _load_posthoc_records(posthoc_csv: Path) -> list[dict]:
    records = []
    with posthoc_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_model = row.get("base_model")
            rec_method = row.get("reconcile_method")
            if not base_model or not rec_method:
                continue
            model_name = f"{base_model}+POSTHOC-{str(rec_method).upper()}"
            rec = {
                "dataset": row.get("dataset"),
                "paper_scope": row.get("paper_scope"),
                "experiment_stage": row.get("experiment_stage"),
                "experiment_id": row.get("experiment_id"),
                "output_namespace": row.get("output_namespace"),
                "run_label": row.get("run_label"),
                "seed": _to_float(row.get("seed")),
                "graph_mode": row.get("graph_mode"),
                "lagtcn_graph_source_version": normalize_lagtcn_graph_source_version(
                    base_model,
                    row.get("lagtcn_graph_source_version"),
                ),
                "sim_type": row.get("sim_type"),
                "model_name": model_name,
                "model_family": row.get("model_family") or _model_family(base_model, source="posthoc"),
                "base_model": base_model,
                "base_model_family": row.get("base_model_family") or _model_family(base_model),
                "reconcile_method": rec_method,
                "gnn_type": row.get("gnn_type", "gcn"),
                "temporal_type": row.get("temporal_type", "gru"),
                "st_mode": row.get("st_mode", "sequential"),
                "stgnn_graph_source": row.get("stgnn_graph_source"),
                "num_timesteps_in": _to_float(row.get("num_timesteps_in")),
                "num_timesteps_out": _to_float(row.get("num_timesteps_out")),
                "output_dir": row.get("output_dir"),
                "model_info_path": row.get("model_info_path"),
                "params_total": None,
                "params_trainable": None,
                "train_time_sec": None,
                "train_peak_gpu_mem_mb": None,
                "infer_latency_ms_per_batch": None,
                "infer_latency_ms_per_sample": None,
                "infer_throughput_samples_per_sec": None,
                "infer_peak_gpu_mem_mb": None,
                "source": "posthoc",
                **_density_metadata_from_row(row),
            }
            for key in METRIC_COLUMNS:
                rec[key] = _to_float(row.get(key))
            records.append(rec)
    return records


def _matches_filters(rec: dict, args: argparse.Namespace) -> bool:
    if args.paper_scope and str(rec.get("paper_scope")) != args.paper_scope:
        return False
    if args.experiment_stage and str(rec.get("experiment_stage")) != args.experiment_stage:
        return False
    if args.experiment_id and str(rec.get("experiment_id")) != args.experiment_id:
        return False
    if args.output_namespace_prefix:
        namespace = str(rec.get("output_namespace") or "")
        if not namespace_matches(namespace, args.output_namespace_prefix):
            return False
    if args.model_family and str(rec.get("model_family")) != args.model_family:
        return False
    if _is_graph_based_record(rec):
        required_policy = args.graph_sparsity_policy or FINAL_GRAPH_SOURCE_POLICY
        if rec.get("graph_sparsity_policy") != required_policy:
            return False
    return True


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _collect_fieldnames(rows: list[dict]) -> list[str]:
    fields = set()
    for row in rows:
        fields.update(row.keys())
    return sorted(fields)


def _aggregate(records: list[dict]) -> list[dict]:
    group_map = defaultdict(list)
    for rec in records:
        key = _group_key(rec, GROUP_FIELDS)
        group_map[key].append(rec)

    summary_rows = []
    for key, rows in sorted(group_map.items()):
        out = {k: v for k, v in zip(GROUP_FIELDS, key)}
        out["n_runs"] = len(rows)
        for metric in METRIC_COLUMNS:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            out[f"{metric}_mean"] = _mean(values)
            out[f"{metric}_std"] = _std(values)
        summary_rows.append(out)
    return summary_rows


def _paired_delta(records: list[dict], baseline_model: str) -> list[dict]:
    pair_fields = [
        "paper_scope",
        "experiment_stage",
        "dataset",
        "num_timesteps_out",
        "graph_sparsity_policy",
        "static_threshold",
        "adaptive_top_k",
        "dynamic_threshold",
        "graph_mode",
        "stgnn_graph_source",
        "gnn_type",
        "temporal_type",
        "st_mode",
        "seed",
    ]
    baseline_index = {}
    for rec in records:
        if rec.get("model_name") != baseline_model:
            continue
        key = tuple(rec.get(k) for k in pair_fields)
        baseline_index[key] = rec

    deltas = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("model_name") == baseline_model:
            continue
        pair_key = tuple(rec.get(k) for k in pair_fields)
        base = baseline_index.get(pair_key)
        if base is None:
            continue
        group_key = _group_key(rec, GROUP_FIELDS)
        for metric in METRIC_COLUMNS:
            cur = rec.get(metric)
            ref = base.get(metric)
            if cur is None or ref is None:
                continue
            deltas[group_key][metric].append(cur - ref)

    out_rows = []
    for key, metric_map in sorted(deltas.items()):
        out = {k: v for k, v in zip(GROUP_FIELDS, key)}
        out["baseline_model"] = baseline_model
        for metric in METRIC_COLUMNS:
            vals = metric_map.get(metric, [])
            out[f"delta_{metric}_mean"] = _mean(vals)
            out[f"delta_{metric}_std"] = _std(vals)
            out[f"delta_{metric}_count"] = len(vals)
        out_rows.append(out)
    return out_rows


def _paired_significance(records: list[dict], baseline_model: str, alpha: float = 0.05) -> list[dict]:
    pair_fields = [
        "paper_scope",
        "experiment_stage",
        "dataset",
        "num_timesteps_out",
        "graph_sparsity_policy",
        "static_threshold",
        "adaptive_top_k",
        "dynamic_threshold",
        "graph_mode",
        "stgnn_graph_source",
        "gnn_type",
        "temporal_type",
        "st_mode",
        "seed",
    ]
    baseline_index = {}
    for rec in records:
        if rec.get("model_name") != baseline_model:
            continue
        key = tuple(rec.get(k) for k in pair_fields)
        baseline_index[key] = rec

    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        if rec.get("model_name") == baseline_model:
            continue
        key = tuple(rec.get(k) for k in pair_fields)
        base = baseline_index.get(key)
        if base is None:
            continue

        group_key = _group_key(rec, GROUP_FIELDS)
        for metric in METRIC_COLUMNS:
            cur = rec.get(metric)
            ref = base.get(metric)
            if cur is None or ref is None:
                continue
            grouped[group_key][metric].append(cur - ref)

    rows = []
    for group_key, metric_map in sorted(grouped.items()):
        group_values = {k: v for k, v in zip(GROUP_FIELDS, group_key)}
        for metric in METRIC_COLUMNS:
            deltas = metric_map.get(metric, [])
            if not deltas:
                continue
            wins = sum(1 for d in deltas if d < 0)
            losses = sum(1 for d in deltas if d > 0)
            ties = len(deltas) - wins - losses
            p_value = _two_sided_sign_test_pvalue(wins=wins, losses=losses)
            p_one_sided = _one_sided_sign_test_pvalue(wins=wins, losses=losses)
            row = dict(group_values)
            row.update({
                "baseline_model": baseline_model,
                "metric": metric,
                "n_pairs": len(deltas),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "delta_mean": _mean(deltas),
                "delta_median": _median(deltas),
                "sign_test_pvalue": p_value,
                "sign_test_pvalue_one_sided": p_one_sided,
                "significant_alpha_0_05": bool(p_value is not None and p_value < alpha),
                "significant_alpha_0_05_one_sided": bool(p_one_sided is not None and p_one_sided < alpha),
            })
            rows.append(row)
    return rows


def _aggregate_efficiency(records: list[dict]) -> list[dict]:
    group_map = defaultdict(list)
    for rec in records:
        key = _group_key(rec, GROUP_FIELDS)
        group_map[key].append(rec)

    rows = []
    for key, group in sorted(group_map.items()):
        out = {k: v for k, v in zip(GROUP_FIELDS, key)}
        out["n_runs"] = len(group)
        for metric in EFFICIENCY_COLUMNS:
            vals = [r.get(metric) for r in group if r.get(metric) is not None]
            out[f"{metric}_mean"] = _mean(vals)
            out[f"{metric}_std"] = _std(vals)
        rows.append(out)
    return rows


def _seed_coverage(
    records: list[dict],
    expected_seeds: list[int],
    skip_model_prefixes: list[str] | None = None,
) -> tuple[list[dict], int]:
    skip_model_prefixes = skip_model_prefixes or []
    expected_set = set(int(s) for s in expected_seeds)
    grouped = defaultdict(set)
    skipped_groups = set()

    for rec in records:
        model_name = str(rec.get("model_name", ""))
        if any(model_name.startswith(pfx) for pfx in skip_model_prefixes):
            key = _group_key(rec, GROUP_FIELDS)
            skipped_groups.add(key)
            continue

        seed = rec.get("seed")
        if seed is None:
            continue
        try:
            seed_int = int(float(seed))
        except Exception:
            continue

        key = _group_key(rec, GROUP_FIELDS)
        grouped[key].add(seed_int)

    rows = []
    failed = 0
    for key, got in sorted(grouped.items()):
        missing = sorted(expected_set - got)
        extra = sorted(got - expected_set)
        ok = (len(missing) == 0 and len(extra) == 0)
        if not ok:
            failed += 1
        row = {k: v for k, v in zip(GROUP_FIELDS, key)}
        row.update({
            "expected_seeds": ",".join(str(s) for s in sorted(expected_set)),
            "found_seeds": ",".join(str(s) for s in sorted(got)),
            "missing_seeds": ",".join(str(s) for s in missing),
            "extra_seeds": ",".join(str(s) for s in extra),
            "seed_coverage_ok": ok,
        })
        rows.append(row)

    for key in sorted(skipped_groups):
        row = {k: v for k, v in zip(GROUP_FIELDS, key)}
        row.update({
            "expected_seeds": ",".join(str(s) for s in sorted(expected_set)),
            "found_seeds": "",
            "missing_seeds": "",
            "extra_seeds": "",
            "seed_coverage_ok": True,
            "note": "skipped_by_model_prefix",
        })
        rows.append(row)

    return rows, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize model_info outputs into CSV tables.")
    parser.add_argument("--data-root", type=str, default="Data", help="Root folder containing datasets.")
    parser.add_argument("--out-dir", type=str, default="results", help="Directory for summary CSV files.")
    parser.add_argument(
        "--posthoc-csvs",
        type=str,
        default=None,
        help="Optional comma-separated posthoc raw CSV paths (e.g., results/reconcile_matrix/reconcile_matrix_runs_raw.csv).",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="GCN-GRU-LP-NO",
        help="Baseline model for paired seed-wise deltas.",
    )
    parser.add_argument("--paper-scope", type=str, default=None, help="Optional paper_scope filter.")
    parser.add_argument("--experiment-stage", type=str, default=None, help="Optional experiment_stage filter.")
    parser.add_argument("--experiment-id", type=str, default=None, help="Optional experiment_id filter.")
    parser.add_argument(
        "--model-family",
        type=str,
        default=None,
        help=(
            "Optional model family filter, e.g. temporal-only, dedicated-STGNN, "
            "graph-enhanced-temporal."
        ),
    )
    parser.add_argument(
        "--graph-sparsity-policy",
        type=str,
        default=FINAL_GRAPH_SOURCE_POLICY,
        choices=[FINAL_GRAPH_SOURCE_POLICY],
        help="Graph protocol filter for graph-based runs.",
    )
    parser.add_argument(
        "--output-namespace-prefix",
        type=str,
        default=None,
        help="Optional output_namespace prefix filter, e.g. ae/j2 or journal_applied_energy/stage_1_graph.",
    )
    parser.add_argument(
        "--expected-seeds",
        type=str,
        default="42,43,44,45,46",
        help="Expected seed list for coverage checking.",
    )
    parser.add_argument(
        "--seed-check-skip-model-prefixes",
        type=str,
        default="",
        help="Comma-separated model-name prefixes excluded from seed coverage check.",
    )
    parser.add_argument(
        "--strict-seed-check",
        action="store_true",
        help="Exit with code 2 if any non-skipped group has missing/extra seeds.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    files = _find_model_info_files(data_root)
    for path in files:
        rec = _load_record(path)
        if rec is not None:
            records.append(rec)


    posthoc_count = 0
    if args.posthoc_csvs:
        for path_str in [p.strip() for p in args.posthoc_csvs.split(",") if p.strip()]:
            posthoc_path = Path(path_str).resolve()
            if not posthoc_path.exists():
                raise FileNotFoundError(f"Posthoc CSV not found: {posthoc_path}")
            posthoc_records = _load_posthoc_records(posthoc_path)
            posthoc_count += len(posthoc_records)
            records.extend(posthoc_records)

    records = [rec for rec in records if _matches_filters(rec, args)]

    if not records:
        raise RuntimeError("No valid records matched the requested filters.")

    raw_path = out_dir / "experiment_runs_raw.csv"
    raw_fields = _collect_fieldnames(records)
    _write_csv(raw_path, records, raw_fields)

    summary_rows = _aggregate(records)
    summary_path = out_dir / "experiment_summary_mean_std.csv"
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []
    _write_csv(summary_path, summary_rows, summary_fields)

    delta_rows = _paired_delta(records, args.baseline_model)
    delta_path = out_dir / "experiment_paired_delta_vs_baseline.csv"
    if delta_rows:
        delta_fields = list(delta_rows[0].keys())
        _write_csv(delta_path, delta_rows, delta_fields)

    sig_rows = _paired_significance(records, args.baseline_model, alpha=0.05)
    sig_path = out_dir / "experiment_significance_vs_baseline.csv"
    if sig_rows:
        sig_fields = list(sig_rows[0].keys())
        _write_csv(sig_path, sig_rows, sig_fields)

    efficiency_rows = _aggregate_efficiency(records)
    efficiency_path = out_dir / "experiment_efficiency_summary.csv"
    if efficiency_rows:
        efficiency_fields = _collect_fieldnames(efficiency_rows)
        _write_csv(efficiency_path, efficiency_rows, efficiency_fields)

    # Optional dedicated posthoc summary
    posthoc_rows = [r for r in records if r.get("source") == "posthoc"]
    posthoc_summary_path = out_dir / "experiment_posthoc_summary_mean_std.csv"
    if posthoc_rows:
        posthoc_group_map = defaultdict(list)
        for rec in posthoc_rows:
            key = (
                rec.get("paper_scope"),
                rec.get("experiment_stage"),
                rec.get("model_family"),
                rec.get("base_model_family"),
                rec.get("dataset"),
                rec.get("num_timesteps_out"),
                rec.get("graph_sparsity_policy"),
                rec.get("static_threshold"),
                rec.get("adaptive_top_k"),
                rec.get("dynamic_threshold"),
                rec.get("graph_mode"),
                rec.get("lagtcn_graph_source_version"),
                rec.get("stgnn_graph_source"),
                rec.get("base_model"),
                rec.get("gnn_type"),
                rec.get("temporal_type"),
                rec.get("st_mode", "sequential"),
                rec.get("reconcile_method"),
            )
            posthoc_group_map[key].append(rec)
        posthoc_summary = []
        for key, rows in sorted(posthoc_group_map.items()):
            (
                paper_scope,
                experiment_stage,
                model_family,
                base_model_family,
                dataset,
                horizon,
                graph_sparsity_policy,
                static_threshold,
                adaptive_top_k,
                dynamic_threshold,
                graph_mode,
                lagtcn_graph_source_version,
                stgnn_graph_source,
                base_model,
                gnn_type,
                temporal_type,
                st_mode,
                rec_method,
            ) = key
            out = {
                "paper_scope": paper_scope,
                "experiment_stage": experiment_stage,
                "model_family": model_family,
                "base_model_family": base_model_family,
                "dataset": dataset,
                "num_timesteps_out": horizon,
                "graph_sparsity_policy": graph_sparsity_policy,
                "static_threshold": static_threshold,
                "adaptive_top_k": adaptive_top_k,
                "dynamic_threshold": dynamic_threshold,
                "graph_mode": graph_mode,
                "lagtcn_graph_source_version": lagtcn_graph_source_version,
                "stgnn_graph_source": stgnn_graph_source,
                "base_model": base_model,
                "gnn_type": gnn_type,
                "temporal_type": temporal_type,
                "st_mode": st_mode,
                "reconcile_method": rec_method,
                "n_runs": len(rows),
            }
            for metric in METRIC_COLUMNS:
                vals = [r.get(metric) for r in rows if r.get(metric) is not None]
                out[f"{metric}_mean"] = _mean(vals)
                out[f"{metric}_std"] = _std(vals)
            posthoc_summary.append(out)
        _write_csv(posthoc_summary_path, posthoc_summary, _collect_fieldnames(posthoc_summary))


    # Seed coverage hard check
    expected_seeds = _parse_int_list(args.expected_seeds)
    skip_prefixes = [s.strip() for s in args.seed_check_skip_model_prefixes.split(",") if s.strip()]
    seed_rows, failed_groups = _seed_coverage(
        records=records,
        expected_seeds=expected_seeds,
        skip_model_prefixes=skip_prefixes,
    )
    seed_cov_path = out_dir / "experiment_seed_coverage.csv"
    if seed_rows:
        _write_csv(seed_cov_path, seed_rows, _collect_fieldnames(seed_rows))

    print(f"Loaded runs: {len(records)}")
    if posthoc_count:
        print(f"Loaded posthoc rows: {posthoc_count}")
    print(f"Raw runs CSV: {raw_path}")
    print(f"Summary CSV: {summary_path}")
    if delta_rows:
        print(f"Paired delta CSV (baseline={args.baseline_model}): {delta_path}")
    else:
        print("Paired delta CSV skipped (no matched seed-wise pairs).")
    if sig_rows:
        print(f"Significance CSV (baseline={args.baseline_model}): {sig_path}")
    else:
        print("Significance CSV skipped (no matched seed-wise pairs).")
    if efficiency_rows:
        print(f"Efficiency CSV: {efficiency_path}")
    else:
        print("Efficiency CSV skipped (no valid efficiency records).")
    if posthoc_rows:
        print(f"Posthoc summary CSV: {posthoc_summary_path}")
    else:
        print("Posthoc summary CSV skipped (no posthoc rows).")
    if seed_rows:
        print(f"Seed coverage CSV: {seed_cov_path}")
    else:
        print("Seed coverage CSV skipped (no rows).")

    if args.strict_seed_check and failed_groups > 0:
        print(f"Seed coverage strict check failed: {failed_groups} group(s) incomplete.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
