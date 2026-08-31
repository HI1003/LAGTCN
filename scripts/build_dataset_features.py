#!/usr/bin/env python3
"""Build derived input feature tensors for hierarchical forecasting datasets.

The output tensors keep channel 0 identical to the existing target tensor.
Additional channels are input-only covariates such as calendar and
weather features. They are not reconciliation targets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from dateutil import parser as date_parser
from numpy.lib.format import open_memmap
from pandas.tseries.holiday import USFederalHolidayCalendar


TRAIN_RATIO = 0.8


@dataclass(frozen=True)
class FeatureOutput:
    feature_set: str
    base_value_file: str
    output_value_file: str
    raw_csv_file: str
    include_weather: bool = False


@dataclass(frozen=True)
class DatasetConfig:
    raw_csv_file: str
    calendar_builder: Callable[[Path, pd.DatetimeIndex], tuple[np.ndarray, list[str]]]
    outputs: tuple[FeatureOutput, ...]


def _safe_float32(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _cyclic(values: np.ndarray, period: float, name: str) -> list[tuple[str, np.ndarray]]:
    radians = 2.0 * math.pi * values.astype(np.float64) / period
    return [
        (f"{name}_sin", np.sin(radians).astype(np.float32)),
        (f"{name}_cos", np.cos(radians).astype(np.float32)),
    ]


def _add_feature(names: list[str], columns: list[np.ndarray], name: str, values: np.ndarray) -> None:
    names.append(name)
    columns.append(_safe_float32(values))


def _date_flags(ts: pd.DatetimeIndex, holiday_dates: set[pd.Timestamp], prefix: str) -> list[tuple[str, np.ndarray]]:
    normalized = pd.DatetimeIndex(ts.normalize())
    holiday_dates = {pd.Timestamp(d).normalize() for d in holiday_dates}
    pre_dates = {d - pd.Timedelta(days=1) for d in holiday_dates}
    post_dates = {d + pd.Timedelta(days=1) for d in holiday_dates}
    return [
        (f"{prefix}_holiday", normalized.isin(holiday_dates).astype(np.float32)),
        (f"{prefix}_pre_holiday", normalized.isin(pre_dates).astype(np.float32)),
        (f"{prefix}_post_holiday", normalized.isin(post_dates).astype(np.float32)),
    ]


def _us_federal_holidays(ts: pd.DatetimeIndex) -> set[pd.Timestamp]:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(
        start=ts.min().normalize() - pd.Timedelta(days=7),
        end=ts.max().normalize() + pd.Timedelta(days=7),
    )
    return {pd.Timestamp(d).normalize() for d in holidays}


def _parse_gefcom2012_holidays(path: Path) -> set[pd.Timestamp]:
    df = pd.read_csv(path)
    holidays: set[pd.Timestamp] = set()
    for year_col in [c for c in df.columns if str(c).isdigit()]:
        year = int(year_col)
        for value in df[year_col].dropna():
            text = str(value)
            try:
                parsed = date_parser.parse(text, fuzzy=True)
            except (TypeError, ValueError, OverflowError):
                continue
            if not re.search(r"\b\d{4}\b", text):
                parsed = parsed.replace(year=year)
            holidays.add(pd.Timestamp(parsed).normalize())
    return holidays








def _us_dst_flag(ts: pd.DatetimeIndex) -> np.ndarray:
    flags = np.zeros(len(ts), dtype=np.float32)
    for year in range(ts.min().year, ts.max().year + 1):
        if year >= 2007:
            march = pd.Timestamp(year=year, month=3, day=1)
            start = march + pd.Timedelta(days=(6 - march.weekday()) % 7 + 7, hours=2)
            november = pd.Timestamp(year=year, month=11, day=1)
            end = november + pd.Timedelta(days=(6 - november.weekday()) % 7, hours=2)
        else:
            april = pd.Timestamp(year=year, month=4, day=1)
            start = april + pd.Timedelta(days=(6 - april.weekday()) % 7, hours=2)
            october_last = pd.Timestamp(year=year, month=10, day=31)
            end = october_last - pd.Timedelta(days=(october_last.weekday() - 6) % 7) + pd.Timedelta(hours=2)
        flags[(ts >= start) & (ts < end)] = 1.0
    return flags


def _common_calendar_features(ts: pd.DatetimeIndex, hourly: bool) -> tuple[list[str], list[np.ndarray]]:
    names: list[str] = []
    columns: list[np.ndarray] = []

    if hourly:
        for name, values in _cyclic(ts.hour.to_numpy(), 24.0, "hour"):
            _add_feature(names, columns, name, values)

    for name, values in _cyclic(ts.dayofweek.to_numpy(), 7.0, "dow"):
        _add_feature(names, columns, name, values)
    _add_feature(names, columns, "is_weekend", (ts.dayofweek >= 5).astype(np.float32))

    day_of_year = ts.dayofyear.to_numpy() - 1
    for name, values in _cyclic(day_of_year, 365.2425, "dayofyear"):
        _add_feature(names, columns, name, values)

    if not hourly:
        for name, values in _cyclic(ts.month.to_numpy() - 1, 12.0, "month"):
            _add_feature(names, columns, name, values)

    return names, columns


def _hourly_us_calendar(data_dir: Path, ts: pd.DatetimeIndex, holiday_dates: set[pd.Timestamp]) -> tuple[np.ndarray, list[str]]:
    names, columns = _common_calendar_features(ts, hourly=True)
    for name, values in _date_flags(ts, holiday_dates, "us"):
        _add_feature(names, columns, name, values)
    _add_feature(names, columns, "us_dst", _us_dst_flag(ts))
    return np.stack(columns, axis=1).astype(np.float32), names


def build_gefcom2012_calendar(data_dir: Path, ts: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    return _hourly_us_calendar(data_dir, ts, _parse_gefcom2012_holidays(data_dir / "Holiday_List.csv"))


def build_gefcom2017_us_calendar(data_dir: Path, ts: pd.DatetimeIndex) -> tuple[np.ndarray, list[str]]:
    return _hourly_us_calendar(data_dir, ts, _us_federal_holidays(ts))








def _load_time_index(data_dir: Path, raw_csv_file: str) -> pd.DatetimeIndex:
    csv_path = data_dir / raw_csv_file
    columns = pd.read_csv(csv_path, nrows=0).columns
    time_col = columns[0]
    ts = pd.to_datetime(pd.read_csv(csv_path, usecols=[time_col])[time_col], errors="coerce")
    if ts.isna().any():
        raise ValueError(f"{csv_path} has unparseable timestamps in first column {time_col!r}.")
    return pd.DatetimeIndex(ts)


def _weather_table(path: Path, ts: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_excel(path)
    weather_ts = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(df["hr"].astype(int) - 1, unit="h")
    value_cols = [c for c in df.columns if c not in {"date", "hr"}]
    values = df[value_cols].apply(pd.to_numeric, errors="coerce")
    values.index = weather_ts
    values = values.groupby(level=0).mean().sort_index()
    values = values.reindex(ts)
    values = values.interpolate(method="time", limit_direction="both")
    values = values.ffill().bfill()
    return values


def _zscore_train(values: pd.DataFrame) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    arr = values.to_numpy(dtype=np.float32)
    train_len = max(1, int(len(arr) * TRAIN_RATIO))
    mean = arr[:train_len].mean(axis=0)
    std = arr[:train_len].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    normed = (arr - mean) / std
    params = {
        str(col): {"mean": float(mean[idx]), "std": float(std[idx])}
        for idx, col in enumerate(values.columns)
    }
    return normed.astype(np.float32), params


def build_gefcom2017final_weather(data_dir: Path, ts: pd.DatetimeIndex) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
    temp = _weather_table(data_dir / "temperature.xlsx", ts)
    humidity = _weather_table(data_dir / "relative humidity.xlsx", ts)

    temp_mean = temp.mean(axis=1)
    weather = pd.DataFrame(
        {
            "temp_mean_z": temp_mean,
            "temp_min_z": temp.min(axis=1),
            "temp_max_z": temp.max(axis=1),
            "humidity_mean_z": humidity.mean(axis=1),
            "humidity_min_z": humidity.min(axis=1),
            "humidity_max_z": humidity.max(axis=1),
            "heating_degree_65f_z": np.maximum(65.0 - temp_mean, 0.0),
            "cooling_degree_65f_z": np.maximum(temp_mean - 65.0, 0.0),
        },
        index=ts,
    )
    values, params = _zscore_train(weather)
    return values, list(weather.columns), params


DATASETS: dict[str, DatasetConfig] = {
    "GEFCom2012_2level": DatasetConfig(
        raw_csv_file="Load_GEFCom2012_hourly.csv",
        calendar_builder=build_gefcom2012_calendar,
        outputs=(
            FeatureOutput(
                feature_set="target_calendar",
                base_value_file="node_values.npy",
                output_value_file="node_values_calendar.npy",
                raw_csv_file="Load_GEFCom2012_hourly.csv",
            ),
        ),
    ),
    "GEFCom2017QualifyingMatch_3level": DatasetConfig(
        raw_csv_file="GEFCom2017QualifyingMatchDemand.csv",
        calendar_builder=build_gefcom2017_us_calendar,
        outputs=(
            FeatureOutput(
                feature_set="target_calendar",
                base_value_file="node_values.npy",
                output_value_file="node_values_calendar.npy",
                raw_csv_file="GEFCom2017QualifyingMatchDemand.csv",
            ),
        ),
    ),
    "GEFCom2017FinalMatch_4level": DatasetConfig(
        raw_csv_file="load_final_filled.csv",
        calendar_builder=build_gefcom2017_us_calendar,
        outputs=(
            FeatureOutput(
                feature_set="target_calendar",
                base_value_file="node_values.npy",
                output_value_file="node_values_calendar.npy",
                raw_csv_file="load_final_filled.csv",
            ),
            FeatureOutput(
                feature_set="target_calendar_weather",
                base_value_file="node_values.npy",
                output_value_file="node_values_calendar_weather.npy",
                raw_csv_file="load_final_filled.csv",
                include_weather=True,
            ),
        ),
    ),
}


def _append_global_features(base_path: Path, out_path: Path, global_features: np.ndarray) -> tuple[int, int, int]:
    base = np.load(base_path, mmap_mode="r")
    if base.ndim != 3:
        raise ValueError(f"Expected {base_path} to have shape [T, N, F], got {base.shape}.")
    timesteps, nodes, base_dim = map(int, base.shape)
    if global_features.shape[0] != timesteps:
        raise ValueError(
            f"Feature length {global_features.shape[0]} does not match {base_path} timesteps {timesteps}."
        )
    extra_dim = int(global_features.shape[1])
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    out = open_memmap(tmp_path, mode="w+", dtype=np.float32, shape=(timesteps, nodes, base_dim + extra_dim))
    out[:, :, :base_dim] = base
    out[:, :, base_dim:] = global_features[:, None, :]
    out.flush()
    del out
    tmp_path.replace(out_path)
    return timesteps, nodes, base_dim + extra_dim


def _read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_dataset(data_root: Path, dataset: str, requested_sets: set[str], overwrite: bool) -> None:
    cfg = DATASETS[dataset]
    data_dir = data_root / dataset
    ts = _load_time_index(data_dir, cfg.raw_csv_file)
    calendar, calendar_names = cfg.calendar_builder(data_dir, ts)
    metadata_path = data_dir / "feature_metadata.json"
    metadata = _read_metadata(metadata_path)

    weather_cache: tuple[np.ndarray, list[str], dict[str, dict[str, float]]] | None = None

    for output in cfg.outputs:
        if requested_sets and output.feature_set not in requested_sets:
            continue

        out_path = data_dir / output.output_value_file
        if out_path.exists() and not overwrite:
            print(f"[skip] {dataset} {output.feature_set}: {out_path} exists")
            continue

        features = calendar
        feature_names = list(calendar_names)
        weather_norm_params = None
        if output.include_weather:
            if weather_cache is None:
                weather_cache = build_gefcom2017final_weather(data_dir, ts)
            weather, weather_names, weather_norm_params = weather_cache
            features = np.concatenate([features, weather], axis=1)
            feature_names.extend(weather_names)

        shape = _append_global_features(data_dir / output.base_value_file, out_path, features)
        metadata[output.feature_set] = {
            "value_file": output.output_value_file,
            "base_value_file": output.base_value_file,
            "raw_csv_file": output.raw_csv_file,
            "shape": list(shape),
            "base_channels": int(np.load(data_dir / output.base_value_file, mmap_mode="r").shape[2]),
            "added_feature_names": feature_names,
            "train_ratio_for_continuous_feature_normalization": TRAIN_RATIO,
            "weather_normalization": weather_norm_params,
            "note": "Channel 0 remains the forecasting target; added channels are input covariates only.",
        }
        print(f"[write] {dataset} {output.feature_set}: {out_path} shape={shape}")

    _write_metadata(metadata_path, metadata)


def _parse_csv_arg(value: str, available: set[str], label: str) -> list[str]:
    if value.strip().lower() == "all":
        return sorted(available)
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - available)
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Available: {sorted(available)}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Applied Energy calendar and weather feature tensors.")
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[1] / "Data")
    parser.add_argument("--datasets", type=str, default="all", help="Comma-separated dataset names or all.")
    parser.add_argument(
        "--feature-sets",
        type=str,
        default="all",
        help="Comma-separated generated feature sets or all (target_calendar, target_calendar_weather).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing generated feature tensors.")
    args = parser.parse_args()

    datasets = _parse_csv_arg(args.datasets, set(DATASETS), "datasets")
    available_sets = {out.feature_set for cfg in DATASETS.values() for out in cfg.outputs}
    requested_sets = set(_parse_csv_arg(args.feature_sets, available_sets, "feature sets"))
    if args.feature_sets.strip().lower() == "all":
        requested_sets = set()

    for dataset in datasets:
        build_dataset(args.data_root, dataset, requested_sets, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
