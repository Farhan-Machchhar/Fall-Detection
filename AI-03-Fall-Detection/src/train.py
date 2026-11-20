"""
train.py
========
Training pipeline for the AI-03 Fall Detection project.

Workflow
--------
1. Load ``phone_features.csv`` and ``wrist_features.csv`` from *features_dir*.
2. Split participants into train / test sets (no data leakage).
3. Train a ``RandomForestClassifier`` and an ``MLPClassifier`` for each
   placement (phone + wrist) → 4 models total.
4. Export each model as a pickle (``.pkl``) and an ONNX file (``.onnx``).
5. Write ``model_metadata.json`` with feature names, thresholds, and scores.

Usage (CLI)
-----------
    python src/train.py \\
        --features-dir data/features \\
        --models-dir models \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
NON_FEATURE_COLS = {"label", "subject_id", "window_start_ms", "filename"}
SEED = 42
FALL_THRESHOLD = 0.5


# ── Data loading ─────────────────────────────────────────────────────────────

def load_features(csv_path: Path) -> pd.DataFrame:
    """Load a feature CSV, validate required columns, and return the DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError(f"'label' column missing in {csv_path}")
    if "subject_id" not in df.columns:
        raise ValueError(f"'subject_id' column missing in {csv_path}")
    df["label"] = df["label"].astype(int)
    return df


# ── Participant-level split ───────────────────────────────────────────────────

def participant_split(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split *df* into train / test sets at the participant level.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table with a ``subject_id`` column.
    test_ratio : float
        Fraction of unique participants assigned to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    subjects = sorted(df["subject_id"].unique())
    rng = np.random.default_rng(seed)
    n_test = max(1, int(len(subjects) * test_ratio))
    test_subjects = set(rng.choice(subjects, size=n_test, replace=False))

    train_df = df[~df["subject_id"].isin(test_subjects)].copy()
    test_df  = df[ df["subject_id"].isin(test_subjects)].copy()

    logger.info(
        "Split: %d train subjects (%d rows) | %d test subjects (%d rows)",
        len(subjects) - n_test, len(train_df), n_test, len(test_df),
    )
    return train_df, test_df


# ── Model builders ────────────────────────────────────────────────────────────

def _rf_pipeline(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )),
    ])


def _mlp_pipeline(seed: int = SEED) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )),
    ])


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(pipeline: Pipeline, feature_names: list[str], onnx_path: Path) -> bool:
    """
    Export *pipeline* to ONNX format.

    Returns ``True`` on success, ``False`` if skl2onnx is unavailable.
    """
    try:
        from skl2onnx import convert_sklearn           # type: ignore
        from skl2onnx.common.data_types import FloatTensorType  # type: ignore
    except ImportError:
        logger.warning("skl2onnx not installed — ONNX export skipped.")
        return False

    n_features = len(feature_names)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        with open(onnx_path, "wb") as fh:
            fh.write(onnx_model.SerializeToString())
        logger.info("ONNX model saved → %s", onnx_path)
        return True
    except Exception as exc:
        logger.warning("ONNX export failed for %s: %s", onnx_path.name, exc)
        return False


# ── Training orchestration ────────────────────────────────────────────────────

def train_placement(
    df: pd.DataFrame,
    placement: str,
    models_dir: Path,
    seed: int = SEED,
) -> dict:
    """
    Train RandomForest and MLP for one sensor placement.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table for this placement.
    placement : str
        ``"phone"`` or ``"wrist"``.
    models_dir : Path
        Directory to save model files.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Metadata dict for this placement (feature names, scores, paths).
    """
    # ── Feature / target split ────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=int)

    train_df, test_df = participant_split(df, seed=seed)
    X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test  = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    y_test  = test_df["label"].to_numpy(dtype=int)

    placement_meta: dict = {
        "placement":    placement,
        "feature_names": feature_cols,
        "n_train":      int(len(X_train)),
        "n_test":       int(len(X_test)),
        "models":       {},
    }

    for model_name, pipeline_fn in [("random_forest", _rf_pipeline), ("mlp", _mlp_pipeline)]:
        pipeline = pipeline_fn(seed=seed)
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            auc = float("nan")

        scores = {
            "accuracy":  float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc":   auc,
        }
        logger.info("[%s / %s] %s", placement, model_name, scores)

        # ── Save test predictions CSV ─────────────────────────────────────────
        pred_dir = models_dir.parent / "data" / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
        if "window_start_ms" in test_df.columns:
            pred_df["window_start_ms"] = test_df["window_start_ms"].values
        pred_csv = pred_dir / f"{placement}_{model_name}_predictions.csv"
        pred_df.to_csv(pred_csv, index=False)
        logger.info("Predictions saved → %s", pred_csv)

        # ── Pickle ────────────────────────────────────────────────────────────
        pkl_path = models_dir / f"{placement}_{model_name}.pkl"
        with open(pkl_path, "wb") as fh:
            pickle.dump({"pipeline": pipeline, "feature_names": feature_cols}, fh)
        logger.info("Pickle saved → %s", pkl_path)

        # Also save with standard naming expected by inference/app
        alias_pkl = models_dir / f"{placement}_rf_model.pkl" if model_name == "random_forest" else models_dir / f"{placement}_mlp_model.pkl"
        with open(alias_pkl, "wb") as fh:
            pickle.dump({"pipeline": pipeline, "feature_names": feature_cols}, fh)

        # ── ONNX ──────────────────────────────────────────────────────────────
        onnx_path = models_dir / f"{placement}_{model_name}.onnx"
        onnx_ok   = export_onnx(pipeline, feature_cols, onnx_path)

        # Standard naming alias
        if model_name == "random_forest":
            alias_onnx = models_dir / ("wearable_model.onnx" if placement == "wrist" else "phone_model.onnx")
            export_onnx(pipeline, feature_cols, alias_onnx)

        placement_meta["models"][model_name] = {
            "scores":        scores,
            "pkl_path":      str(pkl_path),
            "onnx_path":     str(onnx_path) if onnx_ok else None,
            "fall_threshold": FALL_THRESHOLD,
        }

    return placement_meta


def run_training(
    features_dir: str | Path,
    models_dir: str | Path,
    seed: int = SEED,
) -> dict:
    """
    Full training pipeline for phone + wrist placements.

    Parameters
    ----------
    features_dir : str or Path
        Directory containing ``phone_features.csv`` and ``wrist_features.csv``.
    models_dir : str or Path
        Directory where model files and metadata will be written.
    seed : int
        Random seed (default 42).

    Returns
    -------
    dict
        Nested metadata dict (also written to ``model_metadata.json``).
    """
    features_dir = Path(features_dir)
    models_dir   = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict = {"seed": seed, "placements": {}}

    for placement in ("phone", "wrist"):
        csv_path = features_dir / f"{placement}_features.csv"
        try:
            df = load_features(csv_path)
        except FileNotFoundError:
            logger.warning("Feature file not found, skipping %s: %s", placement, csv_path)
            continue

        logger.info("Training models for placement: %s (%d rows)", placement, len(df))
        pmeta = train_placement(df, placement, models_dir, seed=seed)
        metadata["placements"][placement] = pmeta

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info("Metadata saved → %s", meta_path)

    return metadata


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Train fall detection models.")
    parser.add_argument("--features-dir", default="data/features", help="Feature CSV directory")
    parser.add_argument("--models-dir",   default="models",        help="Output model directory")
    parser.add_argument("--seed",         type=int, default=42,    help="Random seed")
    args = parser.parse_args()

    run_training(args.features_dir, args.models_dir, seed=args.seed)
