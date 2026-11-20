"""
alert_manager.py
================
Fall-alert state machine compatible with Streamlit ``st.session_state``.

States
------
IDLE        No alert is active.
ALERTING    A fall was detected; a countdown is running (default 30 s).
            The user can cancel within this window.
CANCELLED   The user cancelled the alert before the countdown expired.
NOTIFIED    The countdown expired and notification was sent (or would be sent).

Transitions
-----------
IDLE       → ALERTING   : trigger_alert()
ALERTING   → CANCELLED  : cancel_alert()
ALERTING   → NOTIFIED   : tick() when countdown reaches 0
NOTIFIED   → IDLE       : acknowledge() after cooldown
CANCELLED  → IDLE       : acknowledge() or cooldown expires

A *cooldown* period after NOTIFIED/CANCELLED suppresses duplicate alerts.

Design note
-----------
All state lives in a plain Python dict (``state``) so it can be stored in
``st.session_state`` or passed around as a serialisable object.  No threads
are used; the caller drives time forward by calling :meth:`tick`.
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class AlertState(str, Enum):
    IDLE       = "IDLE"
    ALERTING   = "ALERTING"
    CANCELLED  = "CANCELLED"
    NOTIFIED   = "NOTIFIED"


class AlertManager:
    """
    Fall-alert state machine.

    Parameters
    ----------
    countdown_seconds : int
        Duration of the confirmation countdown (default 30 s).
    cooldown_seconds : int
        Minimum gap between consecutive alerts (default 60 s).
    notify_callback : callable or None
        Called with no arguments when an alert is confirmed (countdown → 0).
        Useful for sending SMS / push notifications.
    """

    def __init__(
        self,
        countdown_seconds: int = 30,
        cooldown_seconds: int  = 60,
        notify_callback = None,
    ) -> None:
        self.countdown_seconds = countdown_seconds
        self.cooldown_seconds  = cooldown_seconds
        self._notify_callback  = notify_callback

        # Mutable state dict (designed to be stored in st.session_state)
        self.state: dict = self._initial_state()

    # ── State helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _initial_state() -> dict:
        return {
            "status":           AlertState.IDLE,
            "countdown_start":  None,   # wall-clock time when ALERTING began
            "last_notified_at": None,   # wall-clock time of last NOTIFIED event
            "fall_probability": None,   # probability that triggered the alert
            "source_label":     None,   # e.g. "weighted_fusion"
        }

    @property
    def status(self) -> AlertState:
        return AlertState(self.state["status"])

    @property
    def seconds_remaining(self) -> Optional[float]:
        """Seconds left in the countdown, or None if not in ALERTING state."""
        if self.status != AlertState.ALERTING:
            return None
        elapsed = time.monotonic() - self.state["countdown_start"]
        return max(0.0, self.countdown_seconds - elapsed)

    def _in_cooldown(self) -> bool:
        """True if we are still within the post-alert cooldown period."""
        last = self.state["last_notified_at"]
        if last is None:
            return False
        return (time.monotonic() - last) < self.cooldown_seconds

    # ── External API ──────────────────────────────────────────────────────

    def trigger_alert(
        self,
        fall_probability: float = 1.0,
        source_label: str = "unknown",
    ) -> bool:
        """
        Trigger a new fall alert.

        Parameters
        ----------
        fall_probability : float
            The fused fall probability that triggered this alert.
        source_label : str
            Human-readable source (e.g. ``"weighted_fusion"``).

        Returns
        -------
        bool
            ``True`` if the alert was accepted (state → ALERTING), ``False``
            if it was suppressed by cooldown or an existing alert.
        """
        if self.status == AlertState.ALERTING:
            logger.debug("trigger_alert: already alerting — ignored")
            return False

        if self._in_cooldown():
            logger.debug("trigger_alert: in cooldown — ignored")
            return False

        self.state.update({
            "status":           AlertState.ALERTING,
            "countdown_start":  time.monotonic(),
            "fall_probability": fall_probability,
            "source_label":     source_label,
        })
        logger.info(
            "Alert triggered! p=%.3f source=%s", fall_probability, source_label
        )
        return True

    def cancel_alert(self) -> bool:
        """
        Cancel an active alert (user confirmed they are OK).

        Returns
        -------
        bool
            ``True`` if the alert was successfully cancelled.
        """
        if self.status != AlertState.ALERTING:
            logger.debug("cancel_alert: not alerting — ignored")
            return False

        self.state["status"]           = AlertState.CANCELLED
        self.state["last_notified_at"] = time.monotonic()
        logger.info("Alert CANCELLED by user.")
        return True

    def acknowledge(self) -> None:
        """
        Acknowledge a NOTIFIED or CANCELLED state and return to IDLE.

        Call this after showing the user a confirmation screen.
        """
        if self.status in (AlertState.NOTIFIED, AlertState.CANCELLED):
            self.state = self._initial_state()
            logger.info("Alert acknowledged — back to IDLE.")

    def tick(self) -> AlertState:
        """
        Advance the state machine; call on every UI / event loop iteration.

        In the ALERTING state this checks whether the countdown has expired and
        transitions to NOTIFIED if so (also fires the notify callback).

        Returns
        -------
        AlertState
            Current state after the tick.
        """
        if self.status == AlertState.ALERTING:
            remaining = self.seconds_remaining
            if remaining is not None and remaining <= 0:
                self._fire_notification()

        elif self.status in (AlertState.CANCELLED, AlertState.NOTIFIED):
            # Auto-clear after cooldown has elapsed
            if not self._in_cooldown():
                self.state = self._initial_state()
                logger.debug("Cooldown expired — IDLE.")

        return self.status

    def _fire_notification(self) -> None:
        """Transition to NOTIFIED and invoke the callback."""
        self.state["status"]           = AlertState.NOTIFIED
        self.state["last_notified_at"] = time.monotonic()
        logger.warning(
            "FALL ALERT SENT! probability=%.3f source=%s",
            self.state.get("fall_probability", 0.0),
            self.state.get("source_label", "unknown"),
        )
        if self._notify_callback is not None:
            try:
                self._notify_callback()
            except Exception as exc:
                logger.error("notify_callback raised: %s", exc)

    # ── Serialisation for st.session_state ───────────────────────────────

    def to_dict(self) -> dict:
        """Serialise state for Streamlit session_state storage."""
        status_val = self.state["status"].value if isinstance(self.state["status"], AlertState) else str(self.state["status"])
        return {
            **self.state,
            "status": status_val,
            "countdown_seconds": self.countdown_seconds,
            "cooldown_seconds":  self.cooldown_seconds,
        }

    @classmethod
    def from_dict(cls, d: dict, notify_callback=None) -> "AlertManager":
        """Restore from a dict (e.g. loaded from session_state)."""
        mgr = cls(
            countdown_seconds=d.get("countdown_seconds", 30),
            cooldown_seconds=d.get("cooldown_seconds",  60),
            notify_callback=notify_callback,
        )
        status_raw = d.get("status", AlertState.IDLE)
        if isinstance(status_raw, str) and "." in status_raw:
            status_raw = status_raw.split(".")[-1]
        try:
            status = AlertState(status_raw)
        except (ValueError, KeyError):
            try:
                status = AlertState[status_raw]
            except Exception:
                status = AlertState.IDLE

        mgr.state = {
            "status":           status,
            "countdown_start":  d.get("countdown_start"),
            "last_notified_at": d.get("last_notified_at"),
            "fall_probability": d.get("fall_probability"),
            "source_label":     d.get("source_label"),
        }
        return mgr
