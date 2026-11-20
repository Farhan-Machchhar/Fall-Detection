"""
tests/test_sensor_fusion.py
============================
Unit tests for src/sensor_fusion.py.
Pure logic tests — no I/O, no models required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sensor_fusion import FusionEngine, FusionResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine():
    return FusionEngine(w_phone=0.5, w_wearable=0.5, threshold=0.5, conflict_tolerance=0.3)


# ── Row 1: Both sources available, no conflict ────────────────────────────────

def test_both_available_no_conflict_weighted_avg(engine):
    result = engine.combine_predictions(0.4, 0.6)
    assert result.source_label == "weighted_fusion"
    assert abs(result.probability - 0.5) < 1e-9

def test_both_available_equal_weights(engine):
    result = engine.combine_predictions(0.3, 0.7)
    # diff = 0.4 > conflict_tolerance 0.3 → conflict
    assert "conflict" in result.source_label

def test_weighted_avg_asymmetric_weights():
    eng = FusionEngine(w_phone=0.8, w_wearable=0.2, threshold=0.5)
    result = eng.combine_predictions(0.6, 0.4)
    expected = (0.8 * 0.6 + 0.2 * 0.4) / 1.0
    assert abs(result.probability - expected) < 1e-9

def test_fusion_is_fall_above_threshold(engine):
    result = engine.combine_predictions(0.7, 0.8)
    # diff = 0.1 < 0.3 → weighted avg = 0.75 → fall
    assert result.is_fall is True

def test_fusion_not_fall_below_threshold(engine):
    result = engine.combine_predictions(0.1, 0.2)
    # weighted avg = 0.15 → not fall
    assert result.is_fall is False


# ── Row 2: Phone only ─────────────────────────────────────────────────────────

def test_phone_only(engine):
    result = engine.combine_predictions(0.8, None)
    assert result.source_label == "phone_only"
    assert result.probability  == 0.8
    assert result.is_fall      is True

def test_phone_only_adl(engine):
    result = engine.combine_predictions(0.2, None)
    assert result.source_label == "phone_only"
    assert result.is_fall      is False


# ── Row 3: Wearable only ──────────────────────────────────────────────────────

def test_wearable_only(engine):
    result = engine.combine_predictions(None, 0.9)
    assert result.source_label == "wearable_only"
    assert result.probability  == 0.9
    assert result.is_fall      is True

def test_wearable_only_adl(engine):
    result = engine.combine_predictions(None, 0.1)
    assert result.is_fall is False


# ── Row 4: No data ────────────────────────────────────────────────────────────

def test_no_data(engine):
    result = engine.combine_predictions(None, None)
    assert result.source_label == "no_data"
    assert result.probability  is None
    assert result.is_fall      is None


# ── Row 5: Conflict resolution ────────────────────────────────────────────────

def test_conflict_phone_wins(engine):
    # phone p=0.9 (conf=0.4), wearable p=0.3 (conf=0.2) → phone wins
    result = engine.combine_predictions(0.9, 0.3)
    assert result.source_label == "conflict_phone_wins"
    assert result.probability  == 0.9
    assert result.is_fall      is True

def test_conflict_wearable_wins(engine):
    # phone p=0.3 (conf=0.2), wearable p=0.9 (conf=0.4) → wearable wins
    result = engine.combine_predictions(0.3, 0.9)
    assert result.source_label == "conflict_wearable_wins"
    assert result.probability  == 0.9
    assert result.is_fall      is True

def test_conflict_low_prob(engine):
    # phone=0.05 (conf=0.45), wearable=0.8 (conf=0.3) → diff=0.75 > 0.3 → conflict
    result = engine.combine_predictions(0.05, 0.8)
    assert "conflict" in result.source_label


# ── Custom threshold ──────────────────────────────────────────────────────────

def test_custom_threshold():
    eng = FusionEngine(threshold=0.8)
    result = eng.combine_predictions(0.7, None)
    assert result.is_fall is False   # 0.7 < 0.8

def test_custom_threshold_above():
    eng = FusionEngine(threshold=0.6)
    result = eng.combine_predictions(0.7, None)
    assert result.is_fall is True    # 0.7 >= 0.6


# ── FusionResult dataclass ────────────────────────────────────────────────────

def test_fusion_result_is_dataclass():
    fr = FusionResult(probability=0.5, source_label="test", is_fall=True)
    assert fr.probability == 0.5
    assert fr.source_label == "test"
    assert fr.is_fall is True
