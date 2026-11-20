"""
feature_extraction.py
=====================
Time-domain feature engineering for UMAFall sensor windows.

Features computed per axis (x, y, z)
--------------------------------------
mean, std, min, max, rms, energy, variance, median, IQR

Magnitude features (‖[x, y, z]‖₂)
------------------------------------
magnitude_mean, magnitude_std, magnitude_max, magnitude_min

Jerk features (first-order difference of magnitude)
----------------------------------------------------
jerk_mean, jerk_max

Combined extractor helpers
---------------------------
extract_phone_features(file_df)  — RIGHTPOCKET accelerometer + gyroscope
extract_wrist_features(file_df)  — WRIST accelerometer + gyroscope

Both return a list of feature dicts with 'label' and 'subject_id' included.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from preprocessing import create_windows

logger = logging.getLogger(__name__)

# ── Sensor ID constants ───────────────────────────────────────────────────────
SENSOR_ID_RIGHTPOCKET = 0
SENSOR_ID_WRIST       = 3

SENSOR_TYPE_ACCEL = 0
SENSOR_TYPE_GYRO  = 1

# Window parameters
WINDOW_MS = 2000
STRIDE_MS = 1000


# ── Low-level feature computation ─────────────────────────────────────────────

def _rms(arr: np.ndarray) -> float:
    """Root mean square."""
    return float(np.sqrt(np.mean(arr ** 2)))


def _energy(arr: np.ndarray) -> float:
    """Signal energy (sum of squared values)."""
    return float(np.sum(arr ** 2))


def _iqr(arr: np.ndarray) -> float:
    """Inter-quartile range (Q3 − Q1)."""
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def extract_features_from_window(window_df: pd.DataFrame, prefix: str = "") -> dict:
    """
    Compute time-domain features for a single sensor-stream window.

    Parameters
    ----------
    window_df : pd.DataFrame
        Rows of a single window containing at least ``x``, ``y``, ``z``
        and ``timestamp`` columns.
    prefix : str
        Optional string prepended to every feature key (e.g. ``"acc_"``
        or ``"gyro_"``).

    Returns
    -------
    dict
        Feature name → scalar float mapping.  Returns an empty dict if
        the window has fewer than 3 rows.
    """
    if window_df is None or len(window_df) < 3:
        return {}

    x = window_df["x"].to_numpy(dtype=float)
    y = window_df["y"].to_numpy(dtype=float)
    z = window_df["z"].to_numpy(dtype=float)

    # ── Per-axis features ─────────────────────────────────────────────────────
    features: dict = {}
    for axis_name, axis in [("x", x), ("y", y), ("z", z)]:
        k = f"{prefix}{axis_name}"
        features[f"{k}_mean"]     = float(np.mean(axis))
        features[f"{k}_std"]      = float(np.std(axis,  ddof=0))
        features[f"{k}_min"]      = float(np.min(axis))
        features[f"{k}_max"]      = float(np.max(axis))
        features[f"{k}_rms"]      = _rms(axis)
        features[f"{k}_energy"]   = _energy(axis)
        features[f"{k}_variance"] = float(np.var(axis, ddof=0))
        features[f"{k}_median"]   = float(np.median(axis))
        features[f"{k}_iqr"]      = _iqr(axis)

    # ── Magnitude features ────────────────────────────────────────────────────
    mag = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    pk = f"{prefix}mag"
    features[f"{pk}_mean"] = float(np.mean(mag))
    features[f"{pk}_std"]  = float(np.std(mag,  ddof=0))
    features[f"{pk}_max"]  = float(np.max(mag))
    features[f"{pk}_min"]  = float(np.min(mag))

    # ── Jerk features (Δmagnitude / Δtime) ───────────────────────────────────
    if "timestamp" in window_df.columns and len(mag) > 1:
        ts    = window_df["timestamp"].to_numpy(dtype=float)
        dt    = np.diff(ts)
        dt[dt == 0] = 1.0          # avoid division by zero
        jerk  = np.abs(np.diff(mag) / dt)
        features[f"{prefix}jerk_mean"] = float(np.mean(jerk))
        features[f"{prefix}jerk_max"]  = float(np.max(jerk))
    else:
        features[f"{prefix}jerk_mean"] = 0.0
        features[f"{prefix}jerk_max"]  = 0.0

    return features


# ── Combined sensor extractors ────────────────────────────────────────────────

def _extract_dual_sensor_features(
    file_df: pd.DataFrame,
    sensor_id: int,
    location_name: str,
) -> list[dict]:
    """
    Extract features from accelerometer + gyroscope windows for one placement.

    Parameters
    ----------
    file_df : pd.DataFrame
        Full-file DataFrame from :func:`data_loader.load_file`.
    sensor_id : int
        Sensor placement ID (0=RIGHTPOCKET, 3=WRIST, etc.).
    location_name : str
        Human-readable prefix used in feature names.

    Returns
    -------
    list[dict]
        One feature dict per accelerometer window (gyro features merged in
        where available; filled with 0.0 otherwise).
    """
    if file_df is None or file_df.empty:
        return []

    label      = bool(file_df["label"].iloc[0])
    subject_id = int(file_df["subject_id"].iloc[0]) if pd.notna(file_df["subject_id"].iloc[0]) else -1

    # Accelerometer windows (primary anchor)
    acc_windows = create_windows(
        file_df,
        window_size_ms=WINDOW_MS,
        stride_ms=STRIDE_MS,
        sensor_type=SENSOR_TYPE_ACCEL,
        sensor_id=sensor_id,
    )

    # Gyroscope windows (look up by window_start_ms)
    gyro_windows_map: dict[int, pd.DataFrame] = {}
    gyro_wins = create_windows(
        file_df,
        window_size_ms=WINDOW_MS,
        stride_ms=STRIDE_MS,
        sensor_type=SENSOR_TYPE_GYRO,
        sensor_id=sensor_id,
    )
    for gw in gyro_wins:
        ws = int(gw["window_start_ms"].iloc[0])
        gyro_windows_map[ws] = gw

    result: list[dict] = []
    for aw in acc_windows:
        ws       = int(aw["window_start_ms"].iloc[0])
        acc_feat = extract_features_from_window(aw, prefix=f"{location_name}_acc_")

        if not acc_feat:
            continue

        gw       = gyro_windows_map.get(ws)
        gyro_feat = extract_features_from_window(gw, prefix=f"{location_name}_gyro_") if gw is not None else {}

        # If gyro data is unavailable for this window, fill with zeros
        if not gyro_feat:
            # Build zero-valued gyro features using the same keys as acc but replace prefix
            sample_acc_key = next(iter(acc_feat))
            dummy_prefix   = f"{location_name}_gyro_"
            suffix_start   = sample_acc_key.index(f"{location_name}_acc_") + len(f"{location_name}_acc_")
            # generate dummy zeros for all expected gyro keys
            for k in list(acc_feat.keys()):
                gyro_key = k.replace(f"{location_name}_acc_", dummy_prefix)
                gyro_feat[gyro_key] = 0.0

        combined = {**acc_feat, **gyro_feat}
        combined["label"]      = label
        combined["subject_id"] = subject_id
        combined["window_start_ms"] = ws
        result.append(combined)

    return result


def extract_phone_features(file_df: pd.DataFrame) -> list[dict]:
    """
    Extract features from the RIGHTPOCKET (phone) accelerometer and gyroscope.

    Parameters
    ----------
    file_df : pd.DataFrame
        Full-file DataFrame from :func:`data_loader.load_file`.

    Returns
    -------
    list[dict]
        Feature dicts, one per window, each including ``label`` and
        ``subject_id``.
    """
    return _extract_dual_sensor_features(file_df, SENSOR_ID_RIGHTPOCKET, "phone")


def extract_wrist_features(file_df: pd.DataFrame) -> list[dict]:
    """
    Extract features from the WRIST accelerometer and gyroscope.

    Parameters
    ----------
    file_df : pd.DataFrame
        Full-file DataFrame from :func:`data_loader.load_file`.

    Returns
    -------
    list[dict]
        Feature dicts, one per window, each including ``label`` and
        ``subject_id``.
    """
    return _extract_dual_sensor_features(file_df, SENSOR_ID_WRIST, "wrist")


# ── Batch extraction helpers ──────────────────────────────────────────────────

def build_feature_dataset(
    dfs: list[pd.DataFrame],
    extractor,  # callable: (pd.DataFrame) → list[dict]
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Apply an extractor to all file DataFrames and optionally save to CSV.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        One DataFrame per UMAFall CSV file.
    extractor : callable
        One of :func:`extract_phone_features` or :func:`extract_wrist_features`.
    output_csv : str or None
        If given, save the resulting DataFrame here.

    Returns
    -------
    pd.DataFrame
        Concatenated feature table.
    """
    all_rows: list[dict] = []
    for df in dfs:
        rows = extractor(df)
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No features extracted.")
        return pd.DataFrame()

    out_df = pd.DataFrame(all_rows).fillna(0.0)

    if output_csv:
        out_df.to_csv(output_csv, index=False)
        logger.info("Feature dataset saved to %s (%d rows)", output_csv, len(out_df))

    return out_df
