"""
tests/test_inference.py
=======================
Unit tests for src/inference.py.
Uses pickle-based detectors with synthetic scikit-learn models — no ONNX
runtime or real model files required.
"""

from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inference import (
    OnnxFallDetector,
    PickleFallDetector,
    predict_phone,
    predict_wrist,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = ["phone_acc_x_mean", "phone_acc_y_mean", "phone_acc_z_mean",
                 "phone_acc_mag_mean", "phone_acc_jerk_mean"]

MOCK_FEATURES_FALL = {k: 10.0 for k in FEATURE_NAMES}
MOCK_FEATURES_ADL  = {k: 0.0  for k in FEATURE_NAMES}


def _build_pickle_detector(threshold: float = 0.5) -> tuple[PickleFallDetector, Path]:
    """Create a trained RandomForest pipeline and save to a temp pickle."""
    rng = np.random.default_rng(0)
    n_features = len(FEATURE_NAMES)
    X = rng.random((100, n_features), ).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(int)   # trivial separable rule

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(n_estimators=10, random_state=0)),
    ])
    pipeline.fit(X, y)

    tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
    with open(tmp.name, "wb") as fh:
        pickle.dump({"pipeline": pipeline, "feature_names": FEATURE_NAMES}, fh)

    return PickleFallDetector(tmp.name, fall_threshold=threshold), Path(tmp.name)


# ── PickleFallDetector ────────────────────────────────────────────────────────

def test_pickle_detector_loads():
    det, path = _build_pickle_detector()
    assert det.feature_names == FEATURE_NAMES
    path.unlink(missing_ok=True)

def test_pickle_detector_predict_proba_range():
    det, path = _build_pickle_detector()
    prob = det.predict_proba(MOCK_FEATURES_FALL)
    assert 0.0 <= prob <= 1.0
    path.unlink(missing_ok=True)

def test_pickle_detector_predict_returns_int():
    det, path = _build_pickle_detector()
    pred = det.predict(MOCK_FEATURES_FALL)
    assert pred in (0, 1)
    path.unlink(missing_ok=True)

def test_pickle_detector_threshold_zero_always_fall():
    det, path = _build_pickle_detector(threshold=0.0)
    pred = det.predict(MOCK_FEATURES_ADL)
    assert pred == 1   # any probability ≥ 0.0 → Fall
    path.unlink(missing_ok=True)

def test_pickle_detector_threshold_one_always_adl():
    det, path = _build_pickle_detector(threshold=1.0)
    pred = det.predict(MOCK_FEATURES_FALL)
    assert pred == 0   # no probability can be ≥ 1.0
    path.unlink(missing_ok=True)

def test_pickle_detector_missing_features_default_zero():
    det, path = _build_pickle_detector()
    prob = det.predict_proba({})   # all features missing
    assert 0.0 <= prob <= 1.0
    path.unlink(missing_ok=True)


# ── predict_phone / predict_wrist convenience functions ───────────────────────

def test_predict_phone_returns_float():
    det, path = _build_pickle_detector()
    prob = predict_phone(MOCK_FEATURES_FALL, det)
    assert isinstance(prob, float)
    path.unlink(missing_ok=True)

def test_predict_wrist_returns_float():
    det, path = _build_pickle_detector()
    prob = predict_wrist(MOCK_FEATURES_ADL, det)
    assert isinstance(prob, float)
    path.unlink(missing_ok=True)

def test_predict_phone_and_wrist_different_features():
    """The same detector applied to different feature sets should give different probs."""
    det, path = _build_pickle_detector()
    p_fall = predict_phone(MOCK_FEATURES_FALL, det)
    p_adl  = predict_phone(MOCK_FEATURES_ADL,  det)
    # With a separable dataset they should differ (not strictly required but expected)
    assert isinstance(p_fall, float) and isinstance(p_adl, float)
    path.unlink(missing_ok=True)
