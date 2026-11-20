"""
tests/test_alert_manager.py
============================
Unit tests for src/alert_manager.py.
Time is controlled via unittest.mock.patch so tests run instantly.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alert_manager import AlertManager, AlertState


# ── Helpers ───────────────────────────────────────────────────────────────────

def _manager(countdown: int = 30, cooldown: int = 60) -> AlertManager:
    return AlertManager(countdown_seconds=countdown, cooldown_seconds=cooldown)


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_state_is_idle():
    mgr = _manager()
    assert mgr.status == AlertState.IDLE

def test_initial_seconds_remaining_none():
    mgr = _manager()
    assert mgr.seconds_remaining is None


# ── trigger_alert ─────────────────────────────────────────────────────────────

def test_trigger_alert_transitions_to_alerting():
    mgr = _manager()
    ok  = mgr.trigger_alert(fall_probability=0.9, source_label="phone_only")
    assert ok is True
    assert mgr.status == AlertState.ALERTING

def test_trigger_alert_twice_suppressed():
    mgr = _manager()
    mgr.trigger_alert(0.9)
    ok2 = mgr.trigger_alert(0.8)
    assert ok2 is False
    assert mgr.status == AlertState.ALERTING

def test_trigger_alert_stores_probability():
    mgr = _manager()
    mgr.trigger_alert(fall_probability=0.75)
    assert mgr.state["fall_probability"] == 0.75

def test_trigger_alert_stores_source():
    mgr = _manager()
    mgr.trigger_alert(source_label="wearable_only")
    assert mgr.state["source_label"] == "wearable_only"


# ── seconds_remaining ─────────────────────────────────────────────────────────

def test_seconds_remaining_decreases():
    mgr = _manager(countdown=30)
    with patch("time.monotonic", side_effect=[0.0, 10.0]):
        mgr.trigger_alert()
        remaining = mgr.seconds_remaining
    assert remaining is not None
    assert 19.0 <= remaining <= 21.0

def test_seconds_remaining_not_negative():
    mgr = _manager(countdown=5)
    with patch("time.monotonic", side_effect=[0.0, 100.0]):
        mgr.trigger_alert()
        remaining = mgr.seconds_remaining
    assert remaining == 0.0


# ── cancel_alert ──────────────────────────────────────────────────────────────

def test_cancel_alert_from_alerting():
    mgr = _manager()
    mgr.trigger_alert()
    ok  = mgr.cancel_alert()
    assert ok is True
    assert mgr.status == AlertState.CANCELLED

def test_cancel_alert_from_idle_no_effect():
    mgr = _manager()
    ok  = mgr.cancel_alert()
    assert ok is False
    assert mgr.status == AlertState.IDLE

def test_cancel_alert_records_notified_time():
    mgr = _manager()
    mgr.trigger_alert()
    with patch("time.monotonic", return_value=42.0):
        mgr.cancel_alert()
    assert mgr.state["last_notified_at"] == 42.0


# ── tick → NOTIFIED ───────────────────────────────────────────────────────────

def test_tick_transitions_to_notified_after_countdown():
    callback = MagicMock()
    mgr = AlertManager(countdown_seconds=5, cooldown_seconds=60, notify_callback=callback)

    # Trigger at t=0, tick at t=10 (countdown expired)
    with patch("time.monotonic", side_effect=[0.0, 10.0, 10.0]):
        mgr.trigger_alert()
        mgr.tick()

    assert mgr.status == AlertState.NOTIFIED
    callback.assert_called_once()

def test_tick_does_not_fire_before_countdown():
    mgr = _manager(countdown=30)
    with patch("time.monotonic", side_effect=[0.0, 5.0]):
        mgr.trigger_alert()
        mgr.tick()
    assert mgr.status == AlertState.ALERTING


# ── acknowledge ───────────────────────────────────────────────────────────────

def test_acknowledge_notified_returns_to_idle():
    mgr = _manager(countdown=5)
    with patch("time.monotonic", side_effect=[0.0, 10.0, 10.0]):
        mgr.trigger_alert()
        mgr.tick()
    mgr.acknowledge()
    assert mgr.status == AlertState.IDLE

def test_acknowledge_cancelled_returns_to_idle():
    mgr = _manager()
    mgr.trigger_alert()
    mgr.cancel_alert()
    mgr.acknowledge()
    assert mgr.status == AlertState.IDLE

def test_acknowledge_idle_no_effect():
    mgr = _manager()
    mgr.acknowledge()
    assert mgr.status == AlertState.IDLE


# ── Cooldown suppression ──────────────────────────────────────────────────────

def test_cooldown_suppresses_new_alert():
    mgr = _manager(countdown=5, cooldown=60)
    # Fire and expire countdown at t=10
    with patch("time.monotonic", side_effect=[0.0, 10.0, 10.0]):
        mgr.trigger_alert()
        mgr.tick()
    assert mgr.status == AlertState.NOTIFIED

    # Attempt new alert at t=20 (within 60s cooldown)
    with patch("time.monotonic", return_value=20.0):
        ok = mgr.trigger_alert(0.95)
    assert ok is False

def test_cooldown_expires_then_new_alert_accepted():
    mgr = _manager(countdown=5, cooldown=60)
    with patch("time.monotonic", side_effect=[0.0, 10.0, 10.0]):
        mgr.trigger_alert()
        mgr.tick()
    mgr.acknowledge()

    # After cooldown (t=200)
    with patch("time.monotonic", return_value=200.0):
        # _in_cooldown checks last_notified_at against current time
        # last_notified_at was set at ~10.0; 200.0 - 10.0 = 190 > 60
        ok = mgr.trigger_alert(0.95)
    assert ok is True


# ── Serialisation ─────────────────────────────────────────────────────────────

def test_to_dict_round_trip():
    mgr = _manager()
    mgr.trigger_alert(fall_probability=0.88, source_label="fusion")
    d   = mgr.to_dict()
    mgr2 = AlertManager.from_dict(d)
    assert mgr2.status                   == AlertState.ALERTING
    assert mgr2.state["fall_probability"] == 0.88
    assert mgr2.state["source_label"]     == "fusion"
