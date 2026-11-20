"""
preprocessing.py
================
Sliding-window segmentation of UMAFall sensor streams.

The core function :func:`create_windows` slices a sensor DataFrame (from
:mod:`data_loader`) into time-aligned windows of configurable duration and
stride, handling timestamp gaps gracefully.

Public API
----------
create_windows(df, window_size_ms, stride_ms, sensor_type, sensor_id) → list[pd.DataFrame]
segment_all_files(dfs, **kwargs) → list[dict]
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_WINDOW_MS = 2000   # 2-second window
DEFAULT_STRIDE_MS = 1000   # 1-second stride
MIN_SAMPLES_IN_WINDOW = 10  # discard windows with fewer samples


# ── Core windowing ───────────────────────────────────────────────────────────

def create_windows(
    df: pd.DataFrame,
    window_size_ms: int = DEFAULT_WINDOW_MS,
    stride_ms: int = DEFAULT_STRIDE_MS,
    sensor_type: int = 0,
    sensor_id: Optional[int] = None,
    max_gap_ms: int = 500,
) -> list[pd.DataFrame]:
    """
    Segment a sensor DataFrame into time-aligned sliding windows.

    The function:
    1. Filters by *sensor_type* (and optionally *sensor_id*).
    2. Sorts by timestamp.
    3. Splits the stream at large gaps (> *max_gap_ms* ms) to prevent
       windows from spanning discontinuities.
    4. Slides a window of *window_size_ms* ms with a step of *stride_ms* ms
       over each continuous segment.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame returned by :func:`data_loader.load_file`, containing at
        least columns ``timestamp``, ``x``, ``y``, ``z``, ``sensor_type``,
        ``sensor_id``.
    window_size_ms : int
        Duration of each window in milliseconds (default 2 000).
    stride_ms : int
        Step between consecutive window start times in milliseconds
        (default 1 000).
    sensor_type : int
        Sensor type to keep (0=Accelerometer, 1=Gyroscope, 2=Magnetometer).
    sensor_id : int or None
        If given, further filter to this sensor placement.
    max_gap_ms : int
        Timestamp gap larger than this value splits the stream into separate
        continuous segments (default 500 ms).

    Returns
    -------
    list[pd.DataFrame]
        Each element is a copy of the rows falling inside one window, with
        column ``window_start_ms`` added.  Windows with fewer than
        ``MIN_SAMPLES_IN_WINDOW`` rows are discarded.
    """
    if df is None or df.empty:
        return []

    # ── Filter ───────────────────────────────────────────────────────────────
    mask = df["sensor_type"] == sensor_type
    if sensor_id is not None:
        mask &= df["sensor_id"] == sensor_id

    sub = df.loc[mask].copy()
    if sub.empty:
        return []

    sub = sub.sort_values("timestamp").reset_index(drop=True)

    # ── Split at gaps ─────────────────────────────────────────────────────────
    diffs = sub["timestamp"].diff().fillna(0)
    gap_mask = diffs > max_gap_ms
    segment_ids = gap_mask.cumsum()

    windows: list[pd.DataFrame] = []

    for _seg_id, segment in sub.groupby(segment_ids, sort=False):
        segment = segment.reset_index(drop=True)
        t_start = int(segment["timestamp"].iloc[0])
        t_end   = int(segment["timestamp"].iloc[-1])

        win_start = t_start
        while win_start + window_size_ms <= t_end + 1:
            win_end = win_start + window_size_ms
            win_df = segment[
                (segment["timestamp"] >= win_start) &
                (segment["timestamp"] <  win_end)
            ].copy()

            if len(win_df) >= MIN_SAMPLES_IN_WINDOW:
                win_df["window_start_ms"] = win_start
                windows.append(win_df)

            win_start += stride_ms

    return windows


def segment_all_files(
    dfs: list[pd.DataFrame],
    window_size_ms: int = DEFAULT_WINDOW_MS,
    stride_ms: int = DEFAULT_STRIDE_MS,
    sensor_type: int = 0,
    sensor_id: Optional[int] = None,
) -> list[dict]:
    """
    Apply :func:`create_windows` across a list of file DataFrames.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        DataFrames returned by :func:`data_loader.load_all_files`.
    window_size_ms, stride_ms, sensor_type, sensor_id
        Forwarded to :func:`create_windows`.

    Returns
    -------
    list[dict]
        Each dict has keys:
        ``window`` (pd.DataFrame), ``label`` (bool), ``subject_id`` (int),
        ``filename`` (str).
    """
    results: list[dict] = []
    for df in dfs:
        label      = df["label"].iloc[0]      if not df.empty else None
        subject_id = df["subject_id"].iloc[0] if not df.empty else None
        filename   = df["filename"].iloc[0]   if not df.empty else ""

        wins = create_windows(
            df,
            window_size_ms=window_size_ms,
            stride_ms=stride_ms,
            sensor_type=sensor_type,
            sensor_id=sensor_id,
        )
        for w in wins:
            results.append(
                {
                    "window":     w,
                    "label":      label,
                    "subject_id": subject_id,
                    "filename":   filename,
                }
            )
    return results


# ── Utilities ────────────────────────────────────────────────────────────────

def normalize_window(window_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalize x, y, z columns within a window.

    Parameters
    ----------
    window_df : pd.DataFrame
        A single window DataFrame with columns ``x``, ``y``, ``z``.

    Returns
    -------
    pd.DataFrame
        Copy with normalized ``x``, ``y``, ``z``.
    """
    out = window_df.copy()
    for col in ["x", "y", "z"]:
        mu  = out[col].mean()
        std = out[col].std(ddof=0)
        out[col] = (out[col] - mu) / (std if std > 0 else 1.0)
    return out
