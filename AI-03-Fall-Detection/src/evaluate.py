"""
evaluate.py
===========
Evaluation pipeline for the AI-03 Fall Detection project.

Loads test-set predictions (produced during training or inference), computes
a full set of classification metrics, and writes reports to *reports_dir*:

- ``reports/confusion_matrix_<placement>_<model>.png``
- ``reports/model_comparison.png``
- ``reports/evaluation_results.json``

False-alarm rate
----------------
Consecutive positive-prediction windows within a 5-second cooldown are merged
into a single alert event, so a rapid sequence of overlapping windows counts
as one false alarm rather than many.

Usage (CLI)
-----------
    python src/evaluate.py \\
        --predictions-dir data/predictions \\
        --reports-dir reports
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
COOLDOWN_SECONDS   = 5          # merge consecutive positives within 5 s
WINDOW_STRIDE_MS   = 1000       # used for false-alarm-rate computation


# ── Metric helpers ────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute accuracy, precision, recall, specificity, and F1.

    Parameters
    ----------
    y_true : np.ndarray of int (0/1)
    y_pred : np.ndarray of int (0/1)

    Returns
    -------
    dict
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "tp":          int(tp), "fp": int(fp),
        "tn":          int(tn), "fn": int(fn),
    }


def compute_false_alarm_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window_start_ms: Optional[np.ndarray] = None,
    cooldown_ms: int = COOLDOWN_SECONDS * 1000,
    total_adl_seconds: Optional[float] = None,
) -> float:
    """
    Compute false alarms per hour by merging consecutive FP windows.

    Consecutive false-positive windows within *cooldown_ms* of each other are
    counted as a single false-alarm event.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    window_start_ms : np.ndarray or None
        Timestamps of window starts.  If None, indices × WINDOW_STRIDE_MS are used.
    cooldown_ms : int
        Merge threshold in milliseconds.
    total_adl_seconds : float or None
        Duration of ADL-labelled data in seconds.  If None, estimated from
        the number of ADL windows.

    Returns
    -------
    float
        False alarms per hour.
    """
    if window_start_ms is None:
        window_start_ms = np.arange(len(y_pred)) * WINDOW_STRIDE_MS

    # Only look at ADL (true negatives + false positives)
    adl_mask = y_true == 0
    adl_starts = window_start_ms[adl_mask]
    adl_preds  = y_pred[adl_mask]

    # Sort by time
    order = np.argsort(adl_starts)
    adl_starts = adl_starts[order]
    adl_preds  = adl_preds[order]

    # Merge consecutive positives within cooldown window
    fa_events = 0
    last_event_end = -math.inf

    for t, pred in zip(adl_starts, adl_preds):
        if pred == 1:
            if t > last_event_end + cooldown_ms:
                fa_events += 1
                last_event_end = t
            else:
                # Still within cooldown — extend event
                last_event_end = max(last_event_end, t + WINDOW_STRIDE_MS)

    # Estimate total ADL time
    if total_adl_seconds is None:
        # Each ADL window is WINDOW_STRIDE_MS ms
        total_adl_ms = len(adl_starts) * WINDOW_STRIDE_MS
        total_adl_seconds = total_adl_ms / 1000.0

    fa_per_hour = (fa_events / total_adl_seconds * 3600) if total_adl_seconds > 0 else float("nan")
    return fa_per_hour


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Save a confusion-matrix PNG."""
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        display_labels=["ADL", "Fall"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", out_path)


def plot_model_comparison(all_metrics: dict, out_path: Path) -> None:
    """
    Bar chart comparing key metrics across all placement × model combinations.

    Parameters
    ----------
    all_metrics : dict
        ``{label: metrics_dict}`` where label is e.g. ``"phone / random_forest"``.
    out_path : Path
        Output PNG path.
    """
    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1"]
    labels  = list(all_metrics.keys())
    n_bars  = len(labels)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (label, metrics) in enumerate(all_metrics.items()):
        vals = [metrics.get(k, 0) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=label)

    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(metric_keys)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Comparison plot saved → %s", out_path)


# ── Main evaluation pipeline ──────────────────────────────────────────────────

def run_evaluation(
    predictions_dir: str | Path,
    reports_dir: str | Path,
) -> dict:
    """
    Load prediction CSVs and produce evaluation artefacts.

    Each prediction CSV must have columns:
    ``y_true``, ``y_pred``, and optionally ``window_start_ms``.

    File naming convention: ``<placement>_<model>_predictions.csv``
    e.g. ``phone_random_forest_predictions.csv``

    Parameters
    ----------
    predictions_dir : str or Path
    reports_dir : str or Path

    Returns
    -------
    dict
        Nested evaluation results (also written to ``evaluation_results.json``).
    """
    predictions_dir = Path(predictions_dir)
    reports_dir     = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(predictions_dir.glob("*_predictions.csv"))
    if not pred_files:
        logger.warning("No prediction CSVs found in %s", predictions_dir)
        return {}

    all_metrics: dict = {}
    results: dict = {}

    for pred_path in pred_files:
        # Derive label from filename
        stem  = pred_path.stem.replace("_predictions", "")
        parts = stem.split("_", 1)
        placement  = parts[0] if len(parts) > 0 else "unknown"
        model_name = parts[1] if len(parts) > 1 else "model"
        label      = f"{placement} / {model_name}"

        try:
            df = pd.read_csv(pred_path)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", pred_path.name, exc)
            continue

        if "y_true" not in df or "y_pred" not in df:
            logger.warning("Missing y_true/y_pred in %s", pred_path.name)
            continue

        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)
        t_ms   = df["window_start_ms"].to_numpy() if "window_start_ms" in df.columns else None

        metrics = compute_metrics(y_true, y_pred)
        fa_hr   = compute_false_alarm_rate(y_true, y_pred, window_start_ms=t_ms)
        metrics["false_alarms_per_hour"] = float(fa_hr)

        all_metrics[label] = metrics
        results[label]     = metrics

        # Confusion matrix PNG
        cm_path = reports_dir / f"confusion_matrix_{stem}.png"
        plot_confusion_matrix(y_true, y_pred, title=label, out_path=cm_path)

        logger.info("[%s] %s | FA/hr=%.2f", label, metrics, fa_hr)

    # Comparison plot
    if all_metrics:
        plot_model_comparison(all_metrics, reports_dir / "model_comparison.png")

    # Save JSON
    out_json = reports_dir / "evaluation_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Evaluation results → %s", out_json)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate fall detection models.")
    parser.add_argument("--predictions-dir", default="data/predictions")
    parser.add_argument("--reports-dir",     default="reports")
    args = parser.parse_args()

    run_evaluation(args.predictions_dir, args.reports_dir)
