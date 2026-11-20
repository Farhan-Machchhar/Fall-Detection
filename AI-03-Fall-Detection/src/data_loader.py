"""
data_loader.py
==============
Robust parser for UMAFall CSV files.

UMAFall file format
-------------------
Lines starting with ``%`` are either metadata or the data header.
The data header line contains the string ``TimeStamp`` and defines the
column layout::

    % TimeStamp; Sample No; X-Axis; Y-Axis; Z-Axis; Sensor Type; Sensor ID;

Data rows are plain semicolon-delimited text::

    <timestamp>;<sample_no>;<x>;<y>;<z>;<sensor_type>;<sensor_id>

Metadata lines of interest
---------------------------
``% Type of Movement: TRUE``  → Fall
``% Type of Movement: FALSE`` → ADL (Activities of Daily Living)
``% Subject ID: <n>``         → Participant number

Public API
----------
load_file(path) → pd.DataFrame | None
load_all_files(dataset_dir, *, n_jobs=1, verbose=True) → list[pd.DataFrame]
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SENSOR_ID_TO_POSITION: dict[int, str] = {
    0: "RIGHTPOCKET",
    1: "CHEST",
    2: "WAIST",
    3: "WRIST",
    4: "ANKLE",
}

SENSOR_TYPE_TO_NAME: dict[int, str] = {
    0: "Accelerometer",
    1: "Gyroscope",
    2: "Magnetometer",
}

# Column names after parsing
OUTPUT_COLUMNS = [
    "timestamp",
    "sample_no",
    "x",
    "y",
    "z",
    "sensor_type",
    "sensor_id",
    "position",
    "label",
    "subject_id",
    "filename",
]

# ── Filename pattern ─────────────────────────────────────────────────────────
# UMAFall_Subject_01_Fall_ForwardFall_Trial1_20151019.csv
_FILENAME_RE = re.compile(
    r"UMAFall_Subject_(?P<subject>\d+)_(?P<activity>ADL|Fall)_",
    re.IGNORECASE,
)


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parse_subject_from_filename(filename: str) -> Optional[int]:
    """Extract subject ID from the filename using regex."""
    m = _FILENAME_RE.search(filename)
    if m:
        return int(m.group("subject"))
    return None


def _activity_from_filename(filename: str) -> Optional[str]:
    """Return 'Fall' or 'ADL' inferred from the filename."""
    m = _FILENAME_RE.search(filename)
    if m:
        return m.group("activity").capitalize()
    return None


def _parse_metadata(lines: list[str]) -> dict:
    """
    Scan comment lines for metadata keys.

    Parameters
    ----------
    lines : list[str]
        Raw lines from the CSV file.

    Returns
    -------
    dict with keys: label (bool), subject_id (int | None), header_line_idx (int | None)
    """
    meta: dict = {
        "label": None,          # True = Fall, False = ADL
        "subject_id": None,
        "header_line_idx": None,
    }

    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("%"):
            # First non-comment line — header scanning is over
            break

        # Data header detection
        if "TimeStamp" in line:
            meta["header_line_idx"] = idx
            continue

        # Fall / ADL label
        lower = line.lower()
        if "type of movement" in lower:
            if "true" in lower:
                meta["label"] = True
            elif "false" in lower:
                meta["label"] = False
            continue

        # Subject ID (may appear in metadata; filename is the fallback)
        m = re.search(r"subject\s*(id)?[:\s]+(\d+)", lower)
        if m:
            meta["subject_id"] = int(m.group(2))

    return meta


def load_file(path: str | Path) -> Optional[pd.DataFrame]:
    """
    Parse a single UMAFall CSV file into a tidy DataFrame.

    Parameters
    ----------
    path : str or Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame with columns defined by OUTPUT_COLUMNS, or ``None`` if the
    file cannot be parsed (error is logged at WARNING level).
    """
    path = Path(path)
    filename = path.name

    # ── Read raw lines ───────────────────────────────────────────────────────
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError as exc:
        logger.warning("Cannot open %s: %s", filename, exc)
        return None

    if not lines:
        logger.warning("Empty file: %s", filename)
        return None

    # ── Parse metadata ───────────────────────────────────────────────────────
    meta = _parse_metadata(lines)

    # Subject ID: prefer metadata line; fall back to filename
    subject_id = meta["subject_id"] or _parse_subject_from_filename(filename)

    # Label: prefer metadata line; fall back to filename
    label_bool = meta["label"]
    if label_bool is None:
        activity = _activity_from_filename(filename)
        if activity is not None:
            label_bool = activity == "Fall"
        else:
            logger.warning("Cannot determine label for %s", filename)
            label_bool = None   # will be NaN in DataFrame

    # ── Find data start ──────────────────────────────────────────────────────
    # Data rows start after the header comment line.
    # Fallback: skip all lines beginning with '%'.
    if meta["header_line_idx"] is not None:
        data_start = meta["header_line_idx"] + 1
    else:
        data_start = 0
        while data_start < len(lines) and lines[data_start].strip().startswith("%"):
            data_start += 1

    data_lines = lines[data_start:]
    if not data_lines:
        logger.warning("No data rows found in %s", filename)
        return None

    # ── Parse data rows ──────────────────────────────────────────────────────
    records = []
    for raw in data_lines:
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 7:
            continue
        try:
            records.append(
                {
                    "timestamp":   int(float(parts[0])),
                    "sample_no":   int(float(parts[1])),
                    "x":           float(parts[2]),
                    "y":           float(parts[3]),
                    "z":           float(parts[4]),
                    "sensor_type": int(float(parts[5])),
                    "sensor_id":   int(float(parts[6])),
                }
            )
        except (ValueError, IndexError):
            # Silently skip malformed rows
            continue

    if not records:
        logger.warning("No valid data rows parsed in %s", filename)
        return None

    df = pd.DataFrame(records)

    # ── Enrich with metadata ─────────────────────────────────────────────────
    df["position"]   = df["sensor_id"].map(SENSOR_ID_TO_POSITION)
    df["label"]      = label_bool
    df["subject_id"] = subject_id
    df["filename"]   = filename

    # Ensure column order
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def load_all_files(
    dataset_dir: str | Path,
    *,
    max_files: Optional[int] = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> list[pd.DataFrame]:
    """
    Load UMAFall CSV files found under *dataset_dir*.

    Parameters
    ----------
    dataset_dir : str or Path
        Directory containing the raw UMAFall CSV files.
    max_files : int or None
        Optional maximum number of files to load.
    n_jobs : int
        Number of parallel workers (1 = sequential).  Requires ``joblib``
        when > 1.
    verbose : bool
        Print a progress summary to stdout.

    Returns
    -------
    list[pd.DataFrame]
        One DataFrame per successfully parsed file (files that return ``None``
        from :func:`load_file` are excluded).
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    csv_files = sorted(dataset_dir.glob("*.csv"))
    if max_files is not None:
        csv_files = csv_files[:max_files]
    if not csv_files:
        logger.warning("No CSV files found in %s", dataset_dir)
        return []

    if verbose:
        print(f"Found {len(csv_files)} CSV files in {dataset_dir}")

    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed  # type: ignore

            results = Parallel(n_jobs=n_jobs, verbose=0)(
                delayed(load_file)(p) for p in csv_files
            )
        except ImportError:
            logger.warning("joblib not installed — falling back to sequential loading.")
            results = [load_file(p) for p in csv_files]
    else:
        results = []
        for i, p in enumerate(csv_files):
            df = load_file(p)
            results.append(df)
            if verbose and (i + 1) % 100 == 0:
                print(f"  Loaded {i + 1}/{len(csv_files)} files …")

    valid = [df for df in results if df is not None]
    failed = len(results) - len(valid)

    if verbose:
        print(f"Successfully loaded: {len(valid)} | Failed / skipped: {failed}")

    return valid


# ── CLI convenience ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python data_loader.py <dataset_dir>")
        sys.exit(1)

    dfs = load_all_files(sys.argv[1], verbose=True)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(combined.head())
        print(f"\nTotal rows: {len(combined):,}")
