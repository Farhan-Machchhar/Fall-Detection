"""
sensor_simulator.py
===================
Simulate a wrist-worn sensor streaming real-time data from UMAFall recordings.

The :class:`SensorSimulator` replays wrist accelerometer and gyroscope data
from pre-loaded UMAFall DataFrames, one window at a time, while simulating
realistic wearable states (connected / disconnected / worn / not-worn).

Public API
----------
SensorSimulator(dataset_dir, sensor_id=WRIST) → simulator
    .connect()
    .disconnect()
    .put_on()
    .take_off()
    .get_next_window() → feature dict or None
    .reset()
    .state_summary() → dict
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SENSOR_ID_WRIST   = 3
SENSOR_TYPE_ACCEL = 0
SENSOR_TYPE_GYRO  = 1
WINDOW_MS         = 2000
STRIDE_MS         = 1000


class SensorSimulator:
    """
    Replay UMAFall sensor data with simulated wearable states.

    States
    ------
    connected : bool
        Whether the BLE/wireless link is active.
    worn : bool
        Whether the device is on the user's wrist.

    When ``connected`` is ``False`` or ``worn`` is ``False``, wrist features
    are returned as ``None``.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the directory containing UMAFall CSV files.
    sensor_id : int
        Sensor placement to simulate (default 3 = WRIST).
    seed : int
        Random seed for file-ordering reproducibility.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        sensor_id: int = SENSOR_ID_WRIST,
        seed: int = 42,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.sensor_id   = sensor_id
        self._rng        = random.Random(seed)

        # Wearable state
        self.connected   = False
        self.worn        = False

        # Playback state
        self._files: list[Path] = []
        self._file_idx: int     = 0
        self._windows: list[pd.DataFrame] = []
        self._phone_windows: list[dict] = []
        self._wrist_windows: list[dict] = []
        self._current_df: Optional[pd.DataFrame] = None
        self._win_idx: int      = 0
        self.active_source: Optional[str] = None  # filename of current recording

        self._load_file_list()

    # ── Setup ──────────────────────────────────────────────────────────────

    def _load_file_list(self) -> None:
        """Scan dataset_dir for CSV files and shuffle."""
        files = sorted(self.dataset_dir.glob("*.csv"))
        self._rng.shuffle(files)
        self._files = files
        logger.info("SensorSimulator: found %d CSV files", len(files))

    def load_specific_file(self, filename: str) -> bool:
        """Load a specific CSV file by filename or path."""
        from data_loader import load_file
        from feature_extraction import extract_phone_features, extract_wrist_features

        target = Path(filename)
        if not target.is_absolute():
            target = self.dataset_dir / filename
        if not target.exists():
            logger.error("File not found: %s", target)
            return False

        df = load_file(target)
        if df is None:
            return False

        phone_wins = extract_phone_features(df)
        wrist_wins = extract_wrist_features(df)
        if not wrist_wins and not phone_wins:
            return False

        self._current_df    = df
        self._phone_windows = phone_wins
        self._wrist_windows = wrist_wins
        self._win_idx       = 0
        self.active_source  = target.name
        logger.info("SensorSimulator: loaded specific file %s (%d windows)", target.name, len(wrist_wins))
        return True

    def _advance_file(self) -> bool:
        """
        Load paired windows from the next file.
        Returns ``True`` if windows were loaded, ``False`` if no more files.
        """
        from data_loader import load_file
        from feature_extraction import extract_phone_features, extract_wrist_features

        while self._file_idx < len(self._files):
            path = self._files[self._file_idx]
            self._file_idx += 1

            df = load_file(path)
            if df is None:
                continue

            phone_wins = extract_phone_features(df)
            wrist_wins = extract_wrist_features(df)
            if phone_wins or wrist_wins:
                self._current_df    = df
                self._phone_windows = phone_wins
                self._wrist_windows = wrist_wins
                self._win_idx       = 0
                self.active_source  = path.name
                logger.debug("SensorSimulator: switched to %s (%d phone, %d wrist windows)",
                             path.name, len(phone_wins), len(wrist_wins))
                return True

        # Wrap around — replay from the start
        logger.info("SensorSimulator: all files played, wrapping around.")
        self._file_idx = 0
        self._rng.shuffle(self._files)
        return False

    # ── State control ──────────────────────────────────────────────────────

    def connect(self) -> None:
        """Simulate BLE connection."""
        self.connected = True
        logger.debug("SensorSimulator: connected")

    def disconnect(self) -> None:
        """Simulate BLE disconnection."""
        self.connected = False
        logger.debug("SensorSimulator: disconnected")

    def put_on(self) -> None:
        """Simulate user putting on the wearable."""
        self.worn = True
        logger.debug("SensorSimulator: worn")

    def take_off(self) -> None:
        """Simulate user removing the wearable."""
        self.worn = False
        logger.debug("SensorSimulator: taken off")

    def reset(self) -> None:
        """Reset playback to the beginning of the file list."""
        self._file_idx = 0
        self._win_idx  = 0
        self._windows  = []
        self._phone_windows = []
        self._wrist_windows = []
        self._current_df = None
        self.active_source = None
        self._rng.shuffle(self._files)
        logger.debug("SensorSimulator: reset")

    # ── Streaming API ──────────────────────────────────────────────────────

    def get_next_pair(self) -> tuple[Optional[dict], Optional[dict]]:
        """
        Advance simulation and return (phone_features, wrist_features).
        - phone_features: full 66-feature dict (phone_acc_*, phone_gyro_*)
        - wrist_features: full 66-feature dict (wrist_acc_*, wrist_gyro_*) if connected & worn, else None
        """
        max_len = max(len(self._phone_windows), len(self._wrist_windows))
        if self._win_idx >= max_len:
            ok = self._advance_file()
            if not ok:
                max_len = max(len(self._phone_windows), len(self._wrist_windows))
                if self._win_idx >= max_len:
                    return None, None

        idx = self._win_idx
        self._win_idx += 1

        phone_f = dict(self._phone_windows[idx]) if idx < len(self._phone_windows) else None
        wrist_f = dict(self._wrist_windows[idx]) if idx < len(self._wrist_windows) else None

        if phone_f:
            phone_f["source_file"] = self.active_source
        if wrist_f:
            wrist_f["source_file"] = self.active_source

        wrist_out = wrist_f if (self.connected and self.worn) else None
        return phone_f, wrist_out

    def get_next_window(self) -> Optional[dict]:
        """
        Return next wrist window feature dict if connected and worn, else None.
        Backward-compatible with original simulator signature.
        """
        _, wrist_f = self.get_next_pair()
        return wrist_f

    def get_current_raw_samples(self, n_samples: int = 100) -> dict:
        """
        Return recent raw accelerometer and gyroscope samples for live plotting.
        """
        result = {
            "wrist_acc":  {"x": [], "y": [], "z": []},
            "wrist_gyro": {"x": [], "y": [], "z": []},
            "phone_acc":  {"x": [], "y": [], "z": []},
            "phone_gyro": {"x": [], "y": [], "z": []},
        }
        if self._current_df is None or self._current_df.empty:
            return result

        df = self._current_df
        # Wrist acc (sensor_id=3, sensor_type=0)
        w_acc = df[(df["sensor_id"] == 3) & (df["sensor_type"] == 0)].tail(n_samples)
        if not w_acc.empty:
            result["wrist_acc"] = {"x": w_acc["x"].tolist(), "y": w_acc["y"].tolist(), "z": w_acc["z"].tolist()}

        # Wrist gyro (sensor_id=3, sensor_type=1)
        w_gyr = df[(df["sensor_id"] == 3) & (df["sensor_type"] == 1)].tail(n_samples)
        if not w_gyr.empty:
            result["wrist_gyro"] = {"x": w_gyr["x"].tolist(), "y": w_gyr["y"].tolist(), "z": w_gyr["z"].tolist()}

        # Phone acc (sensor_id=0, sensor_type=0)
        p_acc = df[(df["sensor_id"] == 0) & (df["sensor_type"] == 0)].tail(n_samples)
        if not p_acc.empty:
            result["phone_acc"] = {"x": p_acc["x"].tolist(), "y": p_acc["y"].tolist(), "z": p_acc["z"].tolist()}

        # Phone gyro (sensor_id=0, sensor_type=1)
        p_gyr = df[(df["sensor_id"] == 0) & (df["sensor_type"] == 1)].tail(n_samples)
        if not p_gyr.empty:
            result["phone_gyro"] = {"x": p_gyr["x"].tolist(), "y": p_gyr["y"].tolist(), "z": p_gyr["z"].tolist()}

        return result

    def state_summary(self) -> dict:
        """Return a snapshot of the current simulator state."""
        return {
            "connected":     self.connected,
            "worn":          self.worn,
            "active_source": self.active_source,
            "file_idx":      self._file_idx,
            "win_idx":       self._win_idx,
            "total_files":   len(self._files),
        }
