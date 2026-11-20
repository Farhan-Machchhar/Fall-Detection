"""
sensor_fusion.py
================
Combine phone + wearable fall predictions using a decision-table approach.

Decision table (5 rows)
-----------------------
+-----------+-----------+---------------------------------------------------+
| Phone     | Wearable  | Action                                            |
+===========+===========+===================================================+
| Available | Available | Weighted average of both probabilities            |
+-----------+-----------+---------------------------------------------------+
| Available | No data   | Use phone probability alone                       |
+-----------+-----------+---------------------------------------------------+
| No data   | Available | Use wearable probability alone                    |
+-----------+-----------+---------------------------------------------------+
| No data   | No data   | Cannot determine → probability = None             |
+-----------+-----------+---------------------------------------------------+
| Available | Conflict  | If |p_phone − p_wearable| > conflict_tol,          |
|           |           | trust the higher-confidence source                |
+-----------+-----------+---------------------------------------------------+

Public API
----------
FusionEngine(w_phone, w_wearable, threshold, conflict_tolerance)
    .combine_predictions(p_phone, p_wearable) → FusionResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class FusionResult:
    """
    Result from the fusion engine.

    Attributes
    ----------
    probability : float or None
        Combined fall probability (None when no data is available from either
        source).
    source_label : str
        Human-readable description of which source(s) contributed.
    is_fall : bool or None
        ``True`` if ``probability >= threshold``, else ``False``; ``None``
        when ``probability`` is None.
    """
    probability:  Optional[float]
    source_label: str
    is_fall:      Optional[bool]


# ── FusionEngine ──────────────────────────────────────────────────────────────

class FusionEngine:
    """
    Implements the 5-row fall-detection decision table.

    Parameters
    ----------
    w_phone : float
        Weight assigned to the phone prediction (default 0.5).
    w_wearable : float
        Weight assigned to the wearable prediction (default 0.5).
        Note: ``w_phone + w_wearable`` need not equal 1; they are normalised
        internally.
    threshold : float
        Probability threshold for ``is_fall`` classification (default 0.5).
    conflict_tolerance : float
        Absolute difference in probabilities above which sources are considered
        conflicting (default 0.3).  In conflict mode the higher-confidence
        (further from 0.5) prediction is used.
    """

    def __init__(
        self,
        w_phone: float = 0.5,
        w_wearable: float = 0.5,
        threshold: float = 0.5,
        conflict_tolerance: float = 0.3,
    ) -> None:
        self.w_phone           = w_phone
        self.w_wearable        = w_wearable
        self.threshold         = threshold
        self.conflict_tolerance = conflict_tolerance

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _confidence(p: float) -> float:
        """Distance from the decision boundary (0.5) — higher = more certain."""
        return abs(p - 0.5)

    def _weighted_avg(self, p_phone: float, p_wearable: float) -> float:
        total_w = self.w_phone + self.w_wearable
        if total_w == 0:
            return 0.5
        return (self.w_phone * p_phone + self.w_wearable * p_wearable) / total_w

    def _is_fall_from_prob(self, prob: Optional[float]) -> Optional[bool]:
        if prob is None:
            return None
        return prob >= self.threshold

    # ── Public API ────────────────────────────────────────────────────────

    def combine_predictions(
        self,
        p_phone: Optional[float],
        p_wearable: Optional[float],
    ) -> FusionResult:
        """
        Fuse phone and wearable fall probabilities.

        Parameters
        ----------
        p_phone : float or None
            Fall probability from the phone model, or ``None`` if unavailable.
        p_wearable : float or None
            Fall probability from the wearable model, or ``None`` if
            unavailable.

        Returns
        -------
        FusionResult
        """
        # Row 4: No data from either source
        if p_phone is None and p_wearable is None:
            logger.debug("Fusion: no data from either source")
            return FusionResult(
                probability=None,
                source_label="no_data",
                is_fall=None,
            )

        # Row 2: Phone only
        if p_phone is not None and p_wearable is None:
            logger.debug("Fusion: phone only (p=%.3f)", p_phone)
            return FusionResult(
                probability=p_phone,
                source_label="phone_only",
                is_fall=self._is_fall_from_prob(p_phone),
            )

        # Row 3: Wearable only
        if p_phone is None and p_wearable is not None:
            logger.debug("Fusion: wearable only (p=%.3f)", p_wearable)
            return FusionResult(
                probability=p_wearable,
                source_label="wearable_only",
                is_fall=self._is_fall_from_prob(p_wearable),
            )

        # Both sources available — check for conflict
        assert p_phone is not None and p_wearable is not None

        diff = abs(p_phone - p_wearable)
        if diff > self.conflict_tolerance:
            # Row 5: Conflict — trust the more confident source
            if self._confidence(p_phone) >= self._confidence(p_wearable):
                prob         = p_phone
                source_label = "conflict_phone_wins"
            else:
                prob         = p_wearable
                source_label = "conflict_wearable_wins"

            logger.debug(
                "Fusion: conflict (diff=%.3f) → %s (p=%.3f)", diff, source_label, prob
            )
            return FusionResult(
                probability=prob,
                source_label=source_label,
                is_fall=self._is_fall_from_prob(prob),
            )

        # Row 1: Both available, no conflict — weighted average
        prob = self._weighted_avg(p_phone, p_wearable)
        logger.debug("Fusion: weighted avg (p=%.3f)", prob)
        return FusionResult(
            probability=prob,
            source_label="weighted_fusion",
            is_fall=self._is_fall_from_prob(prob),
        )
