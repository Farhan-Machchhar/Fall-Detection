"""
tests/test_requirements_19.py
==============================
Systematic test suite validating all 19 mandatory verification items
from Section 13 of the project specification:

1. Loading a fall recording.
2. Loading a normal ADL recording.
3. Extracting phone features.
4. Extracting wrist features.
5. Processing multiple CSV files.
6. Handling missing sensor streams.
7. Training both models (verification of trained model artifacts).
8. Generating evaluation metrics (checking reports/evaluation_results.json).
9. ONNX inference (verifying runtime execution).
10. Phone-only detection.
11. Wearable-only detection.
12. Both-device fusion.
13. Wearable disconnection fallback.
14. Wearable not-worn fallback.
15. Sensor unavailable state.
16. Fall alert countdown.
17. Alert cancellation.
18. Simulated caregiver notification.
19. Application startup verification.
"""

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure src is on path
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATASET_DIR = (
    PROJECT_ROOT / "Dataset"
    if (PROJECT_ROOT / "Dataset").exists()
    else (
        PROJECT_ROOT.parent / "Dataset"
        if (PROJECT_ROOT.parent / "Dataset").exists()
        else Path(os.getenv("DATASET_DIR", r"d:\Farhan\Assignment\Dataset"))
    )
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from alert_manager import AlertManager, AlertState
from data_loader import load_all_files, load_file
from feature_extraction import (
    extract_features_from_window,
    extract_phone_features,
    extract_wrist_features,
)
from inference import load_onnx_model
from sensor_fusion import FusionEngine, FusionResult
from sensor_simulator import SensorSimulator

FALL_SAMPLE = DATASET_DIR / "UMAFall_Subject_02_Fall_forwardFall_5_2016-06-13_20-54-39.csv"
ADL_SAMPLE = DATASET_DIR / "UMAFall_Subject_01_ADL_Walking_1_2017-04-14_23-25-52.csv"


# 1. Loading a fall recording
def test_item_01_loading_fall_recording():
    assert FALL_SAMPLE.exists(), f"Sample file {FALL_SAMPLE} missing"
    df = load_file(FALL_SAMPLE)
    assert df is not None
    assert len(df) > 0
    assert bool(df["label"].iloc[0]) is True
    assert "timestamp" in df.columns
    assert "x" in df.columns
    assert "y" in df.columns
    assert "z" in df.columns


# 2. Loading a normal ADL recording
def test_item_02_loading_adl_recording():
    assert ADL_SAMPLE.exists(), f"Sample file {ADL_SAMPLE} missing"
    df = load_file(ADL_SAMPLE)
    assert df is not None
    assert len(df) > 0
    assert bool(df["label"].iloc[0]) is False


# 3. Extracting phone features
def test_item_03_extracting_phone_features():
    df = load_file(FALL_SAMPLE)
    feats = extract_phone_features(df)
    assert isinstance(feats, list)
    assert len(feats) > 0
    first = feats[0]
    assert "phone_acc_x_mean" in first
    assert "phone_acc_mag_mean" in first
    assert "phone_acc_jerk_mean" in first
    assert first["label"] is True


# 4. Extracting wrist features
def test_item_04_extracting_wrist_features():
    df = load_file(FALL_SAMPLE)
    feats = extract_wrist_features(df)
    assert isinstance(feats, list)
    assert len(feats) > 0
    first = feats[0]
    assert "wrist_acc_x_mean" in first
    assert "wrist_acc_mag_mean" in first
    assert first["label"] is True


# 5. Processing multiple CSV files
def test_item_05_processing_multiple_csv_files():
    dfs = load_all_files(str(DATASET_DIR), max_files=4)
    assert len(dfs) == 4
    for df in dfs:
        assert isinstance(df, pd.DataFrame)
        assert not df.empty


# 6. Handling missing sensor streams
def test_item_06_handling_missing_sensor_streams():
    df = load_file(FALL_SAMPLE)
    # Filter out wrist data completely
    df_no_wrist = df[df["sensor_id"] != 3].copy()
    wrist_feats = extract_wrist_features(df_no_wrist)
    # Should handle gracefully without crashing, returning empty list
    assert wrist_feats == []


# 7. Training both models
def test_item_07_trained_models_exist():
    phone_onnx = PROJECT_ROOT / "models" / "phone_model.onnx"
    wearable_onnx = PROJECT_ROOT / "models" / "wearable_model.onnx"
    phone_pkl = PROJECT_ROOT / "models" / "phone_rf_model.pkl"
    wearable_pkl = PROJECT_ROOT / "models" / "wrist_rf_model.pkl"

    assert phone_onnx.exists() and phone_onnx.stat().st_size > 1000
    assert wearable_onnx.exists() and wearable_onnx.stat().st_size > 1000
    assert phone_pkl.exists() and phone_pkl.stat().st_size > 1000
    assert wearable_pkl.exists() and wearable_pkl.stat().st_size > 1000


# 8. Generating evaluation metrics
def test_item_08_evaluation_metrics_generated():
    rep_path = PROJECT_ROOT / "reports" / "evaluation_results.json"
    assert rep_path.exists()
    with open(rep_path) as f:
        rep = json.load(f)

    for k in ["phone / random_forest", "wrist / random_forest", "phone / mlp", "wrist / mlp"]:
        assert k in rep
        metrics = rep[k]
        assert "accuracy" in metrics
        assert "recall" in metrics
        assert "specificity" in metrics
        assert "f1" in metrics
        assert "false_alarms_per_hour" in metrics


# 9. ONNX inference
def test_item_09_onnx_inference():
    phone_model = load_onnx_model(
        PROJECT_ROOT / "models" / "phone_model.onnx",
        metadata_path=PROJECT_ROOT / "models" / "model_metadata.json",
        placement="phone",
    )
    wrist_model = load_onnx_model(
        PROJECT_ROOT / "models" / "wearable_model.onnx",
        metadata_path=PROJECT_ROOT / "models" / "model_metadata.json",
        placement="wrist",
    )
    p_ph = phone_model.predict_proba({"phone_acc_x_mean": 2.0})
    p_wr = wrist_model.predict_proba({"wrist_acc_x_mean": 2.0})
    assert 0.0 <= p_ph <= 1.0
    assert 0.0 <= p_wr <= 1.0


# 10. Phone-only detection
def test_item_10_phone_only_detection():
    engine = FusionEngine()
    result = engine.combine_predictions(p_phone=0.88, p_wearable=None)
    assert result.source_label == "phone_only"
    assert result.probability == 0.88
    assert result.is_fall is True


# 11. Wearable-only detection
def test_item_11_wearable_only_detection():
    engine = FusionEngine()
    result = engine.combine_predictions(p_phone=None, p_wearable=0.92)
    assert result.source_label == "wearable_only"
    assert result.probability == 0.92
    assert result.is_fall is True


# 12. Both-device fusion
def test_item_12_both_device_fusion():
    engine = FusionEngine(w_phone=0.5, w_wearable=0.5, conflict_tolerance=0.4)
    result = engine.combine_predictions(p_phone=0.80, p_wearable=0.90)
    assert result.source_label == "weighted_fusion"
    assert abs(result.probability - 0.85) < 1e-4
    assert result.is_fall is True


# 13. Wearable disconnection fallback
def test_item_13_wearable_disconnection_fallback():
    sim = SensorSimulator(DATASET_DIR)
    sim.disconnect()  # simulate disconnected
    sim.put_on()
    _, wrist_f = sim.get_next_pair()
    # Wrist features must be None when disconnected
    assert wrist_f is None
    # Engine falls back to phone
    engine = FusionEngine()
    result = engine.combine_predictions(p_phone=0.85, p_wearable=None)
    assert result.source_label == "phone_only"
    assert result.probability == 0.85


# 14. Wearable not-worn fallback
def test_item_14_wearable_not_worn_fallback():
    sim = SensorSimulator(DATASET_DIR)
    sim.connect()
    sim.take_off()  # connected but not worn
    _, wrist_f = sim.get_next_pair()
    # Wrist features must be None when not worn
    assert wrist_f is None
    # Engine falls back to phone
    engine = FusionEngine()
    result = engine.combine_predictions(p_phone=0.82, p_wearable=None)
    assert result.source_label == "phone_only"
    assert result.probability == 0.82


# 15. Sensor unavailable state
def test_item_15_sensor_unavailable_state():
    engine = FusionEngine()
    result = engine.combine_predictions(p_phone=None, p_wearable=None)
    assert result.source_label == "no_data"
    assert result.probability is None
    assert result.is_fall is None


# 16. Fall alert countdown
def test_item_16_fall_alert_countdown():
    mgr = AlertManager(countdown_seconds=30)
    with patch("time.monotonic", side_effect=[0.0, 5.0]):
        ok = mgr.trigger_alert(fall_probability=0.95, source_label="weighted_fusion")
        rem = mgr.seconds_remaining
    assert ok is True
    assert mgr.status == AlertState.ALERTING
    assert rem is not None
    assert 24.0 <= rem <= 26.0


# 17. Alert cancellation
def test_item_17_alert_cancellation():
    mgr = AlertManager(countdown_seconds=30)
    mgr.trigger_alert(fall_probability=0.95)
    assert mgr.status == AlertState.ALERTING
    ok = mgr.cancel_alert()
    assert ok is True
    assert mgr.status == AlertState.CANCELLED
    assert mgr.seconds_remaining is None


# 18. Simulated caregiver notification
def test_item_18_simulated_caregiver_notification():
    callback = MagicMock()
    mgr = AlertManager(countdown_seconds=5, notify_callback=callback)
    with patch("time.monotonic", side_effect=[0.0, 10.0, 10.0]):
        mgr.trigger_alert(fall_probability=0.99)
        mgr.tick()
    assert mgr.status == AlertState.NOTIFIED
    callback.assert_called_once()


# 19. Application startup verification
def test_item_19_application_startup():
    app_path = PROJECT_ROOT / "app.py"
    assert app_path.exists()
    # Verify that app.py has valid python syntax
    with open(app_path, "r", encoding="utf-8") as f:
        code = f.read()
    compiled = compile(code, str(app_path), "exec")
    assert compiled is not None
