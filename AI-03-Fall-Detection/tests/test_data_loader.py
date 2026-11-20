"""
tests/test_data_loader.py
=========================
Unit tests for src/data_loader.py using pytest and mock CSV content.
No real dataset files are required.
"""

from __future__ import annotations

import sys
import textwrap
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    OUTPUT_COLUMNS,
    SENSOR_ID_TO_POSITION,
    _activity_from_filename,
    _parse_metadata,
    _parse_subject_from_filename,
    load_file,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_CSV_CONTENT = textwrap.dedent("""\
    % UMAFall Dataset
    % Subject ID: 05
    % Type of Movement: TRUE
    % TimeStamp; Sample No; X-Axis; Y-Axis; Z-Axis; Sensor Type; Sensor ID;
    1000;1;0.1;0.2;9.8;0;3
    1005;2;0.2;0.3;9.7;0;3
    1010;3;0.3;0.1;9.9;0;3
    1015;4;0.1;0.4;9.6;1;3
    1020;5;0.0;0.5;9.5;1;3
""")

FALL_FILENAME   = "UMAFall_Subject_05_Fall_ForwardFall_Trial1_20151019.csv"
ADL_FILENAME    = "UMAFall_Subject_12_ADL_Walking_Trial2_20151020.csv"
UNKNOWN_FILENAME = "unknown_file.csv"


# ── _parse_subject_from_filename ──────────────────────────────────────────────

def test_parse_subject_fall():
    assert _parse_subject_from_filename(FALL_FILENAME) == 5

def test_parse_subject_adl():
    assert _parse_subject_from_filename(ADL_FILENAME) == 12

def test_parse_subject_unknown():
    assert _parse_subject_from_filename(UNKNOWN_FILENAME) is None


# ── _activity_from_filename ───────────────────────────────────────────────────

def test_activity_fall():
    assert _activity_from_filename(FALL_FILENAME) == "Fall"

def test_activity_adl():
    assert _activity_from_filename(ADL_FILENAME) == "Adl"

def test_activity_unknown():
    assert _activity_from_filename(UNKNOWN_FILENAME) is None


# ── _parse_metadata ───────────────────────────────────────────────────────────

def test_parse_metadata_fall_label():
    lines = VALID_CSV_CONTENT.splitlines(keepends=True)
    meta = _parse_metadata(lines)
    assert meta["label"] is True

def test_parse_metadata_adl_label():
    adl_lines = ["% Type of Movement: FALSE\n", "% TimeStamp; Sample No;\n"]
    meta = _parse_metadata(adl_lines)
    assert meta["label"] is False

def test_parse_metadata_subject_id():
    lines = VALID_CSV_CONTENT.splitlines(keepends=True)
    meta = _parse_metadata(lines)
    assert meta["subject_id"] == 5

def test_parse_metadata_header_line():
    lines = VALID_CSV_CONTENT.splitlines(keepends=True)
    meta = _parse_metadata(lines)
    assert meta["header_line_idx"] is not None


# ── load_file ─────────────────────────────────────────────────────────────────

def _mock_load(content: str, filename: str):
    """Helper: patch open() so load_file reads *content* as the file."""
    with patch("builtins.open", mock_open(read_data=content)):
        with patch("pathlib.Path.name", new_callable=lambda: property(lambda self: filename)):
            path_mock = MagicMock(spec=Path)
            path_mock.name = filename
            path_mock.__str__ = lambda s: f"/fake/{filename}"
            return load_file(path_mock)


def test_load_file_returns_dataframe():
    df = _mock_load(VALID_CSV_CONTENT, FALL_FILENAME)
    assert df is not None
    assert isinstance(df, pd.DataFrame)

def test_load_file_columns():
    df = _mock_load(VALID_CSV_CONTENT, FALL_FILENAME)
    assert df is not None
    for col in OUTPUT_COLUMNS:
        assert col in df.columns, f"Column '{col}' missing"

def test_load_file_label_fall():
    df = _mock_load(VALID_CSV_CONTENT, FALL_FILENAME)
    assert df is not None
    assert bool(df["label"].iloc[0]) is True

def test_load_file_subject_id():
    df = _mock_load(VALID_CSV_CONTENT, FALL_FILENAME)
    assert df is not None
    assert df["subject_id"].iloc[0] == 5

def test_load_file_position_mapping():
    df = _mock_load(VALID_CSV_CONTENT, FALL_FILENAME)
    assert df is not None
    # sensor_id=3 should map to WRIST
    wrist_rows = df[df["sensor_id"] == 3]
    assert all(wrist_rows["position"] == "WRIST")

def test_load_file_numeric_types():
    df = _mock_load(VALID_CSV_CONTENT, FALL_FILENAME)
    assert df is not None
    assert pd.api.types.is_integer_dtype(df["timestamp"])
    assert pd.api.types.is_float_dtype(df["x"])

def test_load_file_empty_returns_none():
    with patch("builtins.open", mock_open(read_data="")):
        path_mock = MagicMock(spec=Path)
        path_mock.name = FALL_FILENAME
        result = load_file(path_mock)
    assert result is None

def test_load_file_oserror_returns_none():
    path_mock = MagicMock(spec=Path)
    path_mock.name = FALL_FILENAME
    with patch("builtins.open", side_effect=OSError("disk error")):
        result = load_file(path_mock)
    assert result is None

def test_load_file_no_data_rows_returns_none():
    content = "% Type of Movement: TRUE\n% TimeStamp; Sample No;\n"
    df = _mock_load(content, FALL_FILENAME)
    assert df is None

def test_load_file_malformed_rows_skipped():
    content = (
        "% Type of Movement: TRUE\n"
        "% TimeStamp; Sample No; X-Axis; Y-Axis; Z-Axis; Sensor Type; Sensor ID;\n"
        "bad;data\n"            # malformed row
        "1000;1;0.1;0.2;9.8;0;3\n"   # valid row
    )
    df = _mock_load(content, FALL_FILENAME)
    assert df is not None
    assert len(df) == 1
