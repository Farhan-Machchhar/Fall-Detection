"""
tests/test_preprocessing.py
============================
Unit tests for src/preprocessing.py.
Uses synthetic DataFrames — no real CSV files required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing import create_windows, normalize_window, segment_all_files

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(
    duration_ms: int = 5000,
    sample_rate_hz: int = 100,
    sensor_type: int = 0,
    sensor_id: int = 3,
    label: bool = True,
    subject_id: int = 1,
    start_ms: int = 0,
) -> pd.DataFrame:
    """Create a synthetic sensor DataFrame."""
    n = int(duration_ms * sample_rate_hz / 1000)
    timestamps = np.linspace(start_ms, start_ms + duration_ms - 1, n, dtype=int)
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "timestamp":   timestamps,
        "sample_no":   np.arange(n),
        "x":           rng.normal(0, 1, n),
        "y":           rng.normal(0, 1, n),
        "z":           rng.normal(9.8, 0.5, n),
        "sensor_type": sensor_type,
        "sensor_id":   sensor_id,
        "position":    "WRIST",
        "label":       label,
        "subject_id":  subject_id,
        "filename":    "test.csv",
    })


# ── create_windows ────────────────────────────────────────────────────────────

def test_create_windows_returns_list():
    df = _make_df()
    wins = create_windows(df, window_size_ms=2000, stride_ms=1000, sensor_type=0)
    assert isinstance(wins, list)

def test_create_windows_non_empty():
    df = _make_df(duration_ms=5000)
    wins = create_windows(df, window_size_ms=2000, stride_ms=1000, sensor_type=0)
    assert len(wins) > 0

def test_create_windows_correct_duration():
    df = _make_df(duration_ms=10000, sample_rate_hz=200, sensor_type=0)
    wins = create_windows(df, window_size_ms=2000, stride_ms=1000, sensor_type=0)
    for w in wins:
        t_span = int(w["timestamp"].max()) - int(w["timestamp"].min())
        assert t_span <= 2000, f"Window spans {t_span} ms > 2000 ms"

def test_create_windows_sensor_type_filter():
    # Mix accelerometer (0) and gyroscope (1) rows
    df0 = _make_df(sensor_type=0, duration_ms=5000)
    df1 = _make_df(sensor_type=1, duration_ms=5000)
    df  = pd.concat([df0, df1], ignore_index=True)
    wins = create_windows(df, sensor_type=0)
    for w in wins:
        assert all(w["sensor_type"] == 0)

def test_create_windows_sensor_id_filter():
    df_wrist = _make_df(sensor_id=3, duration_ms=5000)
    df_chest = _make_df(sensor_id=1, duration_ms=5000)
    df       = pd.concat([df_wrist, df_chest], ignore_index=True)
    wins = create_windows(df, sensor_type=0, sensor_id=3)
    for w in wins:
        assert all(w["sensor_id"] == 3)

def test_create_windows_empty_df():
    wins = create_windows(pd.DataFrame(), sensor_type=0)
    assert wins == []

def test_create_windows_none_df():
    wins = create_windows(None, sensor_type=0)
    assert wins == []

def test_create_windows_handles_gap():
    """A large timestamp gap should split the stream."""
    df_part1 = _make_df(duration_ms=3000, start_ms=0)
    df_part2 = _make_df(duration_ms=3000, start_ms=100_000)  # 100 s gap
    df = pd.concat([df_part1, df_part2], ignore_index=True)
    wins = create_windows(df, window_size_ms=2000, stride_ms=1000, sensor_type=0, max_gap_ms=500)
    # Windows should not span the gap
    for w in wins:
        t_span = int(w["timestamp"].max()) - int(w["timestamp"].min())
        assert t_span < 10_000  # never crosses the 100 s gap

def test_create_windows_window_start_ms_column():
    df = _make_df(duration_ms=5000)
    wins = create_windows(df, window_size_ms=2000, stride_ms=1000, sensor_type=0)
    for w in wins:
        assert "window_start_ms" in w.columns

def test_create_windows_stride_produces_overlap():
    df = _make_df(duration_ms=10000, sample_rate_hz=200)
    wins = create_windows(df, window_size_ms=2000, stride_ms=1000, sensor_type=0)
    if len(wins) >= 2:
        s0 = int(wins[0]["window_start_ms"].iloc[0])
        s1 = int(wins[1]["window_start_ms"].iloc[0])
        assert s1 - s0 == 1000


# ── segment_all_files ─────────────────────────────────────────────────────────

def test_segment_all_files_returns_list():
    dfs = [_make_df(duration_ms=5000) for _ in range(3)]
    result = segment_all_files(dfs, sensor_type=0)
    assert isinstance(result, list)

def test_segment_all_files_dict_keys():
    dfs = [_make_df(duration_ms=5000)]
    result = segment_all_files(dfs, sensor_type=0)
    for item in result:
        assert "window" in item
        assert "label"  in item
        assert "subject_id" in item
        assert "filename"   in item


# ── normalize_window ──────────────────────────────────────────────────────────

def test_normalize_window_zero_mean():
    df = _make_df(duration_ms=2000)
    normed = normalize_window(df)
    for col in ["x", "y", "z"]:
        assert abs(normed[col].mean()) < 1e-10

def test_normalize_window_unit_std():
    df = _make_df(duration_ms=2000)
    normed = normalize_window(df)
    for col in ["x", "y", "z"]:
        assert abs(normed[col].std(ddof=0) - 1.0) < 1e-10

def test_normalize_window_constant_column():
    df = _make_df(duration_ms=2000)
    df["x"] = 5.0  # constant → std = 0
    normed = normalize_window(df)
    assert not normed["x"].isna().any()  # should not produce NaN
