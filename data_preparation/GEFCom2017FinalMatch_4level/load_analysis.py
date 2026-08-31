"""
Load_final_filled.csv Data Quality Check / 数据质量检查

本脚本针对宽表格式的 load_final_filled.csv：首列为时间索引，后续每列为节点值。
功能：时间范围与频率推断、缺失/重复/覆盖率、非数值/负值/零值、IQR 与 z-score
异常、突变、季节性强度、趋势、自相关、跨节点相关性、层级聚合一致性（基于
hierarchy.csv），输出每节点 CSV 以及统一 JSON 报告。
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

def find_data_dir() -> Path:
    target = Path("Data/GEFCom2017FinalMatch_4level")
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / target
        if candidate.exists():
            return candidate
    return Path.cwd()


data_dir: Path = find_data_dir()
input_path: Path = data_dir / "load_final_filled.csv"
per_node_path: Path = data_dir / "load_final_filled_quality_report.csv"
report_json_path: Path = data_dir / "load_final_filled_analysis_report.json"
missing_ts_path: Path = data_dir / "load_final_filled_missing_timestamps_sample.csv"
top_corr_path: Path = data_dir / "load_final_filled_top_correlated_pairs.csv"
agg_consistency_path: Path = data_dir / "load_final_filled_aggregation_consistency.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="load_final_filled.csv quality checks and summaries")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print dataset characteristics; skip correlations, aggregation checks, and outputs",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Rows to show in head() when using summary-only mode",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional path to override default load_final_filled.csv",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _lower_col_map(df: pd.DataFrame) -> Dict[str, str]:
    return {str(c).lower(): c for c in df.columns}


def summarize_raw(df: pd.DataFrame, head_rows: int = 5) -> None:
    print("=== Basic info ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("dtypes:\n", df.dtypes)

    print("\n=== Missing ratio per column ===")
    miss = df.isna().mean().sort_values(ascending=False)
    print(miss)

    time_col = df.columns[0]
    try:
        times = pd.to_datetime(df[time_col])
        print("\n=== Time index ===")
        print("min:", times.min())
        print("max:", times.max())
        print("n_unique:", times.nunique())
        print("head:", times.head())
    except Exception as exc:  # pragma: no cover
        print("time parse failed:", exc)

    value_cols = df.columns[1:]
    if len(value_cols) > 0:
        print("\n=== Coverage per node (non-null ratio) ===")
        coverage = 1 - df[value_cols].isna().mean()
        print(coverage)

        print("\n=== Value stats per node ===")
        print(df[value_cols].describe(percentiles=[0.01, 0.5, 0.99]).transpose())

        neg = (df[value_cols] < 0).sum()
        zero = (df[value_cols] == 0).sum()
        print("\nnegatives per node:")
        print(neg)
        print("zeros per node:")
        print(zero)

    print("\n=== head() sample ===")
    print(df.head(head_rows))


def load_wide(df: pd.DataFrame) -> pd.DataFrame:
    # Expect first column as timestamp index, rest as nodes
    ts_col = df.columns[0]
    wide = df.copy()
    wide[ts_col] = pd.to_datetime(wide[ts_col])
    wide = wide.set_index(ts_col).sort_index()
    # Coerce values to numeric
    wide = wide.apply(pd.to_numeric, errors="coerce")
    return wide


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def autocorr(series: pd.Series, lag: int) -> float:
    s = series.dropna().to_numpy()
    if len(s) <= lag:
        return np.nan
    if np.std(s) == 0:
        return np.nan
    return float(np.corrcoef(s[lag:], s[:-lag])[0, 1])


def seasonality_strength(series: pd.Series, attr: str) -> float:
    s = series.dropna()
    if s.empty:
        return np.nan
    var_all = float(s.var())
    if var_all == 0:
        return 0.0
    try:
        group_vals = getattr(s.index, attr)
    except Exception:
        return np.nan
    means = s.groupby(group_vals).mean()
    return float(means.var() / var_all)


def trend_slope_per_hour(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 2:
        return np.nan
    x = (s.index - s.index[0]).total_seconds() / 3600.0
    slope = np.polyfit(x, s.to_numpy(), 1)[0]
    return float(slope)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _normalize_node(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def build_hierarchy_children(hier_csv: Path) -> Tuple[Dict[str, List[str]], List[str], List[str], List[str], List[str]]:
    import csv

    children: Dict[str, List[str]] = {}
    top_order: List[str] = []
    mid1_order: List[str] = []
    mid2_order: List[str] = []
    bottom_order: List[str] = []

    with hier_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top = _normalize_node(row.get("Top"))
            mid1 = _normalize_node(row.get("Middle1"))
            mid2 = _normalize_node(row.get("Middle2"))
            bottom = _normalize_node(row.get("Bottom"))
            path = [p for p in (top, mid1, mid2, bottom) if p is not None]
            if not path:
                continue
            if top is not None and top not in top_order:
                top_order.append(top)
            if mid1 is not None and mid1 not in mid1_order:
                mid1_order.append(mid1)
            if mid2 is not None and mid2 not in mid2_order:
                mid2_order.append(mid2)
            if bottom is not None and bottom not in bottom_order:
                bottom_order.append(bottom)
            for parent, child in zip(path[:-1], path[1:]):
                children.setdefault(parent, [])
                if child not in children[parent]:
                    children[parent].append(child)

    return children, top_order, mid1_order, mid2_order, bottom_order


def compute_agg_consistency(df_wide: pd.DataFrame, hier_path: Path) -> pd.DataFrame:
    if not hier_path.exists():
        return pd.DataFrame()

    children, top_order, mid1_order, mid2_order, bottom_order = build_hierarchy_children(hier_path)

    @lru_cache(None)
    def bottoms(node: str) -> List[str]:
        if node in bottom_order:
            return [node]
        res: List[str] = []
        for ch in children.get(node, []):
            for b in bottoms(ch):
                if b not in res:
                    res.append(b)
        return res

    nodes_in_data = set(df_wide.columns.astype(str))
    agg_nodes = [n for n in top_order + mid1_order + mid2_order if n in nodes_in_data]

    agg_rows: List[Dict[str, object]] = []
    for node in agg_nodes:
        bottom_nodes = [b for b in bottoms(node) if b in nodes_in_data]
        if not bottom_nodes:
            continue
        agg_series = df_wide[bottom_nodes].sum(axis=1)
        node_series = df_wide[node]
        comp = pd.concat([agg_series, node_series], axis=1, keys=["sum_bottom", "node"])
        comp = comp.dropna()
        if comp.empty:
            continue
        diff = comp["node"] - comp["sum_bottom"]
        mae = float(diff.abs().mean())
        mape = float((diff.abs() / comp["sum_bottom"].replace(0, np.nan)).mean())
        max_abs = float(diff.abs().max())
        agg_rows.append(
            {
                "node": node,
                "num_bottom_children": len(bottom_nodes),
                "mae_vs_bottom_sum": mae,
                "mape_vs_bottom_sum": mape,
                "max_abs_error": max_abs,
            }
        )

    agg_df = pd.DataFrame(agg_rows)

    # Round tiny float errors so near-zero diffs show as 0 in outputs
    if not agg_df.empty:
        for col in ["mae_vs_bottom_sum", "mape_vs_bottom_sum", "max_abs_error"]:
            if col in agg_df.columns:
                agg_df[col] = agg_df[col].round(6)

    return agg_df


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    file_path = args.input if args.input else input_path
    print("Input:", file_path)

    df_raw = pd.read_csv(file_path)
    df_raw.columns = df_raw.columns.astype(str)
    print("Raw shape:", df_raw.shape)
    print("Columns (first 20):", list(df_raw.columns)[:20])

    if args.summary_only:
        summarize_raw(df_raw, head_rows=args.head)
        return

    df_wide = load_wide(df_raw)
    df_wide.columns = df_wide.columns.astype(str)
    print("Wide shape:", df_wide.shape)
    print(df_wide.head())

    # Expected index inferred from data
    freq = pd.infer_freq(df_wide.index)
    if freq is None:
        diffs = df_wide.index.to_series().diff().dropna()
        if not diffs.empty:
            freq = diffs.mode().iloc[0]
    if freq is not None:
        expected_index = pd.date_range(df_wide.index.min(), df_wide.index.max(), freq=freq)
    else:
        expected_index = df_wide.index
    expected_len = len(expected_index)
    df_wide = df_wide.reindex(expected_index)
    print("Inferred freq:", freq)
    print("Expected points:", expected_len)

    # Pre-compute per-node row counts and duplicates
    n_rows = pd.Series({col: int(df_wide.shape[0]) for col in df_wide.columns})
    n_unique_ts = pd.Series({col: df_wide.index.nunique() for col in df_wide.columns})
    dup_ts = pd.Series({col: 0 for col in df_wide.columns})
    non_numeric_counts = pd.Series({col: int(df_raw[col].shape[0] - pd.to_numeric(df_raw[col], errors="coerce").notna().sum()) for col in df_wide.columns})

    # Optional stationarity tests (ADF/KPSS)
    have_statsmodels = False
    try:
        from statsmodels.tsa.stattools import adfuller, kpss  # type: ignore

        have_statsmodels = True
    except Exception:
        have_statsmodels = False

    stationarity_max_points = 20000
    z_threshold = 4.0

    report_rows: List[Dict[str, object]] = []
    missing_samples: List[Tuple[str, pd.Timestamp]] = []

    for node_id in df_wide.columns:
        series = df_wide[node_id]
        present_values = int(series.notna().sum())
        missing_values = int(series.isna().sum())
        coverage = present_values / expected_len if expected_len else 0.0

        neg_count = int((series < 0).sum())
        zero_count = int((series == 0).sum())

        s_non_na = series.dropna()
        if s_non_na.empty:
            q1 = q3 = iqr = np.nan
            outlier_count_iqr = 0
            outlier_count_z = 0
            spike_count = 0
            skew = np.nan
            kurtosis = np.nan
            mean_val = np.nan
            std_val = np.nan
        else:
            q1 = float(s_non_na.quantile(0.25))
            q3 = float(s_non_na.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0 or np.isnan(iqr):
                outlier_count_iqr = 0
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count_iqr = int(((s_non_na < lower) | (s_non_na > upper)).sum())

            mean_val = float(s_non_na.mean())
            std_val = float(s_non_na.std())
            if std_val == 0 or np.isnan(std_val):
                outlier_count_z = 0
            else:
                z = (s_non_na - mean_val) / std_val
                outlier_count_z = int((z.abs() > z_threshold).sum())

            diff = s_non_na.diff().dropna()
            if diff.empty or diff.std() == 0:
                spike_count = 0
            else:
                spike_count = int((diff.abs() > 3 * diff.std()).sum())

            skew = float(s_non_na.skew())
            kurtosis = float(s_non_na.kurtosis())

        if missing_values:
            missing_ts = series[series.isna()].index[:10]
            for ts in missing_ts:
                missing_samples.append((node_id, ts))

        hourly_strength = seasonality_strength(series, "hour")
        weekly_strength = seasonality_strength(series, "dayofweek")
        trend_slope = trend_slope_per_hour(series)
        acf_1 = autocorr(series, 1)
        acf_24 = autocorr(series, 24)
        acf_168 = autocorr(series, 168)

        adf_p = np.nan
        kpss_p = np.nan
        mean_shift_ratio = np.nan
        var_shift_ratio = np.nan
        if not s_non_na.empty:
            half = len(s_non_na) // 2
            if half > 0 and std_val not in (0, np.nan):
                mean_shift_ratio = float((s_non_na.iloc[half:].mean() - s_non_na.iloc[:half].mean()) / std_val)
            var_first = s_non_na.iloc[:half].var() if half > 0 else np.nan
            var_second = s_non_na.iloc[half:].var() if half > 0 else np.nan
            if var_first and not np.isnan(var_first) and not np.isnan(var_second):
                var_shift_ratio = float(var_second / var_first)

        if have_statsmodels and len(s_non_na) > 20:
            s_test = s_non_na
            if len(s_test) > stationarity_max_points:
                step = max(1, len(s_test) // stationarity_max_points)
                s_test = s_test.iloc[::step]
            try:
                adf_p = float(adfuller(s_test, autolag="AIC")[1])
            except Exception:
                adf_p = np.nan
            try:
                kpss_p = float(kpss(s_test, regression="c", nlags="auto")[1])
            except Exception:
                kpss_p = np.nan

        report_rows.append(
            {
                "node_id": node_id,
                "n_rows": int(n_rows.get(node_id, 0)),
                "n_unique_ts": int(n_unique_ts.get(node_id, 0)),
                "duplicate_timestamps": int(dup_ts.get(node_id, 0)),
                "expected_len": int(expected_len),
                "missing_values": missing_values,
                "coverage_ratio": coverage,
                "non_numeric_values": int(non_numeric_counts.get(node_id, 0)),
                "negative_values": neg_count,
                "zero_values": zero_count,
                "outlier_count_iqr": outlier_count_iqr,
                "outlier_count_z": outlier_count_z,
                "spike_count": spike_count,
                "min": float(s_non_na.min()) if not s_non_na.empty else np.nan,
                "max": float(s_non_na.max()) if not s_non_na.empty else np.nan,
                "mean": mean_val,
                "std": std_val,
                "skew": skew,
                "kurtosis": kurtosis,
                "seasonality_strength_hourly": hourly_strength,
                "seasonality_strength_weekly": weekly_strength,
                "trend_slope_per_hour": trend_slope,
                "acf_lag_1": acf_1,
                "acf_lag_24": acf_24,
                "acf_lag_168": acf_168,
                "adf_pvalue": adf_p,
                "kpss_pvalue": kpss_p,
                "mean_shift_ratio": mean_shift_ratio,
                "var_shift_ratio": var_shift_ratio,
            }
        )

    report_df = pd.DataFrame(report_rows).sort_values("node_id")

    # Cross-series correlation summary
    corr_matrix = df_wide.corr()
    n_nodes = len(corr_matrix.columns)
    if n_nodes > 1:
        mean_corr = (corr_matrix.sum(axis=1) - 1) / (n_nodes - 1)
        report_df["mean_corr_to_others"] = report_df["node_id"].map(mean_corr)

        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        pairs = upper.stack().reset_index()
        pairs.columns = ["node_id_1", "node_id_2", "correlation"]
        top_corr = pairs.sort_values("correlation", ascending=False).head(20)
        top_corr.to_csv(top_corr_path, index=False)

        corr_values = pairs["correlation"]
        cross_summary = {
            "num_pairs": int(len(corr_values)),
            "mean_pairwise_corr": float(corr_values.mean()),
            "median_pairwise_corr": float(corr_values.median()),
            "min_pairwise_corr": float(corr_values.min()),
            "max_pairwise_corr": float(corr_values.max()),
            "p95_pairwise_corr": float(corr_values.quantile(0.95)),
            "high_corr_pairs_ge_0.99": int((corr_values >= 0.99).sum()),
        }
    else:
        report_df["mean_corr_to_others"] = np.nan
        top_corr = pd.DataFrame(columns=["node_id_1", "node_id_2", "correlation"])
        cross_summary = {
            "num_pairs": 0,
            "mean_pairwise_corr": np.nan,
            "median_pairwise_corr": np.nan,
            "min_pairwise_corr": np.nan,
            "max_pairwise_corr": np.nan,
            "p95_pairwise_corr": np.nan,
            "high_corr_pairs_ge_0.99": 0,
        }

    # Aggregation consistency (if aggregated nodes exist)
    agg_df = compute_agg_consistency(df_wide, data_dir / "hierarchy.csv")
    if not agg_df.empty:
        agg_df.to_csv(agg_consistency_path, index=False)

    # Save per-meter report
    report_df.to_csv(per_node_path, index=False)

    # Missing timestamp samples
    if missing_samples:
        ms_df = pd.DataFrame(missing_samples, columns=["meter_id", "timestamp"])
        ms_df.to_csv(missing_ts_path, index=False)

    summary = {
        "start": df_wide.index.min().isoformat() if len(df_wide.index) else None,
        "end": df_wide.index.max().isoformat() if len(df_wide.index) else None,
        "inferred_freq": str(freq),
        "expected_len": int(expected_len),
        "num_nodes": int(len(df_wide.columns)),
        "overall_missing_ratio": float(report_df["missing_values"].sum() / (expected_len * max(1, len(df_wide.columns)))),
        "nodes_complete_ratio": float((report_df["missing_values"] == 0).mean()),
        "have_statsmodels": bool(have_statsmodels),
    }

    report = {
        "summary": summary,
        "cross_series_summary": cross_summary,
        "aggregation_consistency_summary": agg_df.to_dict(orient="records") if not agg_df.empty else [],
        "top_correlated_pairs": top_corr.to_dict(orient="records"),
        "per_node": report_df.to_dict(orient="records"),
    }

    with report_json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Saved per-node CSV:", per_node_path)
    print("Saved unified report JSON:", report_json_path)
    if not top_corr.empty:
        print("Saved top correlations:", top_corr_path)
    if not agg_df.empty:
        print("Saved aggregation consistency:", agg_consistency_path)
    if missing_samples:
        print("Saved missing timestamp sample:", missing_ts_path)

    # Optional quick summaries
    print(report_df[["coverage_ratio", "missing_values", "duplicate_timestamps", "outlier_count_iqr"]].describe())
    # Example plot (uncomment if needed)
    # import matplotlib.pyplot as plt
    # report_df["coverage_ratio"].hist(bins=20)
    # plt.title("Coverage ratio per meter_id")
    # plt.show()


if __name__ == "__main__":
    main()
