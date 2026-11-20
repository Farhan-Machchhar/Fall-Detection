"""
inference.py
============
ONNX model loading and real-time fall-detection inference.

The inference module loads exported ONNX models (produced by :mod:`train`)
and provides simple ``predict_*`` functions that accept a feature dict and
return a fall probability.

Feature ordering is enforced at load time using the metadata stored alongside
the model, so callers never need to worry about column alignment.

Public API
----------
load_onnx_model(path) → InferenceSession wrapper
predict_phone(features_dict, session) → float probability
predict_wrist(features_dict, session) → float probability
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── ONNX session wrapper ──────────────────────────────────────────────────────

class OnnxFallDetector:
    """
    Thin wrapper around an ONNX Runtime InferenceSession.

    Parameters
    ----------
    onnx_path : str or Path
        Path to the ``.onnx`` model file.
    feature_names : list[str]
        Ordered list of feature names; used to build the input array from a
        feature dict in the correct column order.
    fall_threshold : float
        Probability threshold above which a prediction is classified as *Fall*.
    """

    def __init__(
        self,
        onnx_path: str | Path,
        feature_names: list[str],
        fall_threshold: float = 0.5,
    ) -> None:
        import onnxruntime as ort  # type: ignore

        self.onnx_path      = Path(onnx_path)
        self.feature_names  = feature_names
        self.fall_threshold = fall_threshold

        self._session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict_proba(self, features_dict: dict) -> float:
        """
        Return fall probability for a feature dict.

        Missing features default to 0.0.
        """
        x = np.array(
            [[float(features_dict.get(k, 0.0)) for k in self.feature_names]],
            dtype=np.float32,
        )
        outputs = self._session.run(None, {self._input_name: x})
        # outputs[1] is the probability dict for classifiers; outputs[0] for regressors
        # skl2onnx classifiers: outputs[1] is a list of dicts {0: p0, 1: p1}
        try:
            prob_dict = outputs[1][0]
            if isinstance(prob_dict, dict):
                return float(prob_dict.get(1, 0.0))
            # Some backends return arrays
            return float(np.asarray(prob_dict).ravel()[1])
        except (IndexError, KeyError, TypeError):
            # Fallback: treat first output as probability
            return float(np.asarray(outputs[0]).ravel()[0])

    def predict(self, features_dict: dict) -> int:
        """Return 1 (Fall) or 0 (ADL)."""
        return int(self.predict_proba(features_dict) >= self.fall_threshold)


# ── Pickle-based fallback detector ───────────────────────────────────────────

class PickleFallDetector:
    """
    Fallback detector using a scikit-learn Pipeline loaded from pickle.

    Used when ONNX Runtime is unavailable.
    """

    def __init__(
        self,
        pkl_path: str | Path,
        fall_threshold: float = 0.5,
    ) -> None:
        pkl_path = Path(pkl_path)
        with open(pkl_path, "rb") as fh:
            payload = pickle.load(fh)
        self.pipeline      = payload["pipeline"]
        self.feature_names = payload["feature_names"]
        self.fall_threshold = fall_threshold

    def predict_proba(self, features_dict: dict) -> float:
        x = np.array(
            [[float(features_dict.get(k, 0.0)) for k in self.feature_names]],
            dtype=np.float32,
        )
        try:
            proba = self.pipeline.predict_proba(x)[0, 1]
        except AttributeError:
            proba = float(self.pipeline.predict(x)[0])
        return float(proba)

    def predict(self, features_dict: dict) -> int:
        return int(self.predict_proba(features_dict) >= self.fall_threshold)


# ── Factory functions ─────────────────────────────────────────────────────────

def load_onnx_model(
    onnx_path: str | Path,
    metadata_path: Optional[str | Path] = None,
    placement: str = "phone",
    model_name: str = "random_forest",
    fall_threshold: float = 0.5,
) -> OnnxFallDetector:
    """
    Load an ONNX model and return an :class:`OnnxFallDetector`.

    Feature names are read from *metadata_path* (``model_metadata.json``) if
    provided, otherwise inferred from the ONNX input shape (not recommended).

    Parameters
    ----------
    onnx_path : str or Path
    metadata_path : str or Path or None
        Path to ``model_metadata.json`` produced by :mod:`train`.
    placement : str
        ``"phone"`` or ``"wrist"`` — used to look up features in metadata.
    model_name : str
        ``"random_forest"`` or ``"mlp"``.
    fall_threshold : float

    Returns
    -------
    OnnxFallDetector
    """
    feature_names: list[str] = []

    if metadata_path is not None:
        try:
            with open(metadata_path, "r") as fh:
                meta = json.load(fh)
            feature_names = meta["placements"][placement]["feature_names"]
            fall_threshold = (
                meta["placements"][placement]["models"][model_name].get(
                    "fall_threshold", fall_threshold
                )
            )
        except (KeyError, json.JSONDecodeError, FileNotFoundError) as exc:
            logger.warning("Could not read feature names from metadata: %s", exc)

    return OnnxFallDetector(onnx_path, feature_names, fall_threshold)


def load_pickle_model(
    pkl_path: str | Path,
    fall_threshold: float = 0.5,
) -> PickleFallDetector:
    """Load a scikit-learn pipeline from pickle."""
    return PickleFallDetector(pkl_path, fall_threshold)


# ── Convenience top-level functions ──────────────────────────────────────────

def predict_phone(
    features_dict: dict,
    detector: OnnxFallDetector | PickleFallDetector,
) -> float:
    """
    Return fall probability for the phone (RIGHTPOCKET) sensor placement.

    Parameters
    ----------
    features_dict : dict
        Feature dict produced by :func:`feature_extraction.extract_phone_features`.
    detector : OnnxFallDetector or PickleFallDetector

    Returns
    -------
    float
        Fall probability in [0, 1].
    """
    return detector.predict_proba(features_dict)


def predict_wrist(
    features_dict: dict,
    detector: OnnxFallDetector | PickleFallDetector,
) -> float:
    """
    Return fall probability for the wrist sensor placement.

    Parameters
    ----------
    features_dict : dict
        Feature dict produced by :func:`feature_extraction.extract_wrist_features`.
    detector : OnnxFallDetector or PickleFallDetector

    Returns
    -------
    float
        Fall probability in [0, 1].
    """
    return detector.predict_proba(features_dict)
