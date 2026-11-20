"""
scripts/run_full_pipeline.py
============================
Complete end-to-end pipeline runner for AI-03 Fall Detection.

Handles:
- Loading all UMAFall CSV files
- Extracting phone and wrist features
- Training RF + MLP models
- Saving predictions
- Running evaluation
- Creating all required output files with correct naming

Required output files:
- data/phone_features.csv
- data/wrist_features.csv
- models/phone_rf_model.pkl
- models/wrist_rf_model.pkl
- models/phone_model.onnx
- models/wearable_model.onnx
- reports/evaluation_results.json
- reports/phone_confusion_matrix.png
- reports/wearable_confusion_matrix.png
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR      = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("full_pipeline")

DATASET_DIR  = Path(r"d:\Farhan\Assignment\Dataset")
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
REPORTS_DIR  = PROJECT_ROOT / "reports"
PRED_DIR     = DATA_DIR / "predictions"

SEED = 42


def step1_load(dataset_dir: Path, max_files: int = None):
    """Load all CSV files from the dataset directory."""
    from src.data_loader import load_all_files

    logger.info("=" * 60)
    logger.info("STEP 1: Loading dataset from %s", dataset_dir)
    t0 = time.time()
    dfs = load_all_files(str(dataset_dir), verbose=True)
    if max_files:
        dfs = dfs[:max_files]
    logger.info("Loaded %d files in %.1f s", len(dfs), time.time() - t0)
    return dfs


def step2_extract(dfs: list, data_dir: Path):
    """Extract phone and wrist features and save to data/ directory."""
    import pandas as pd
    from src.feature_extraction import build_feature_dataset, extract_phone_features, extract_wrist_features

    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 2: Extracting features → %s", data_dir)

    results = {}
    for name, extractor in [("phone", extract_phone_features), ("wrist", extract_wrist_features)]:
        t0 = time.time()
        csv_path = data_dir / f"{name}_features.csv"
        df = build_feature_dataset(dfs, extractor, output_csv=str(csv_path))
        logger.info("[%s] %d feature rows in %.1f s → %s", name, len(df), time.time() - t0, csv_path)
        results[name] = df

    return results


def step3_train(features: dict, models_dir: Path, seed: int = SEED):
    """Train RF + MLP for phone and wrist, save models and predictions."""
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models_dir.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    NON_FEATURE_COLS = {"label", "subject_id", "window_start_ms", "filename"}

    metadata = {"seed": seed, "placements": {}}
    all_scores = {}

    for placement, df in features.items():
        if df is None or df.empty:
            logger.warning("No features for %s, skipping", placement)
            continue

        df = df.copy()
        df["label"] = df["label"].astype(int)

        feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        logger.info("=" * 60)
        logger.info("STEP 3: Training %s (%d rows, %d features)", placement, len(df), len(feature_cols))

        # Participant-level split: subjects 1-11 train, 12-17 test
        subjects = sorted(df["subject_id"].unique())
        logger.info("Subjects: %s", subjects)
        # Use last 25% of subjects as test
        n_test = max(1, int(len(subjects) * 0.25))
        rng = np.random.default_rng(seed)
        test_subjects = set(rng.choice(subjects, size=n_test, replace=False))
        train_df = df[~df["subject_id"].isin(test_subjects)]
        test_df  = df[ df["subject_id"].isin(test_subjects)]

        logger.info("Train: %d rows | Test: %d rows | Test subjects: %s", len(train_df), len(test_df), test_subjects)

        X_train = train_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_train = train_df["label"].to_numpy(dtype=int)
        X_test  = test_df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_test  = test_df["label"].to_numpy(dtype=int)

        placement_meta = {"feature_names": feature_cols, "models": {}}
        wrist_label = "wrist" if placement == "wrist" else "phone"

        for model_name, pipeline in [
            ("random_forest", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, min_samples_leaf=2,
                    class_weight="balanced", random_state=seed, n_jobs=-1
                )),
            ])),
            ("mlp", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64), activation="relu", solver="adam",
                    max_iter=500, early_stopping=True, validation_fraction=0.1,
                    random_state=seed
                )),
            ])),
        ]:
            t0 = time.time()
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            scores = {
                "accuracy":  float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall":    float(recall_score(y_test, y_pred, zero_division=0)),
                "f1":        float(f1_score(y_test, y_pred, zero_division=0)),
            }
            logger.info("[%s / %s] acc=%.3f prec=%.3f rec=%.3f f1=%.3f (%.1fs)",
                        placement, model_name, scores["accuracy"], scores["precision"],
                        scores["recall"], scores["f1"], time.time() - t0)

            # Save predictions CSV for evaluate.py
            import pandas as pd
            pred_df = pd.DataFrame({
                "y_true": y_test,
                "y_pred": y_pred,
            })
            if "window_start_ms" in test_df.columns:
                pred_df["window_start_ms"] = test_df["window_start_ms"].values
            pred_path = PRED_DIR / f"{placement}_{model_name}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

            # Save pickle with standard naming (phone_rf_model.pkl, wrist_rf_model.pkl)
            if model_name == "random_forest":
                pkl_path = models_dir / f"{placement}_rf_model.pkl"
            else:
                pkl_path = models_dir / f"{placement}_mlp_model.pkl"
            with open(pkl_path, "wb") as fh:
                pickle.dump({"pipeline": pipeline, "feature_names": feature_cols}, fh)
            logger.info("Saved %s", pkl_path)

            # Also save with original naming convention
            orig_pkl = models_dir / f"{placement}_{model_name}.pkl"
            with open(orig_pkl, "wb") as fh:
                pickle.dump({"pipeline": pipeline, "feature_names": feature_cols}, fh)

            # ONNX export
            if model_name == "random_forest":
                onnx_name = "wearable_model.onnx" if placement == "wrist" else "phone_model.onnx"
            else:
                onnx_name = f"{placement}_{model_name}.onnx"

            onnx_path = models_dir / onnx_name
            _export_onnx(pipeline, feature_cols, onnx_path)

            # Also export with orig naming
            orig_onnx = models_dir / f"{placement}_{model_name}.onnx"
            if orig_onnx != onnx_path:
                _export_onnx(pipeline, feature_cols, orig_onnx)

            placement_meta["models"][model_name] = {
                "scores": scores,
                "pkl_path": str(pkl_path),
            }
            all_scores[f"{placement}/{model_name}"] = scores

        metadata["placements"][placement] = placement_meta

    # Save metadata
    meta_path = models_dir / "model_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    return all_scores


def _export_onnx(pipeline, feature_names: list, onnx_path: Path) -> bool:
    """Export sklearn pipeline to ONNX."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except ImportError:
        logger.warning("skl2onnx not installed — skipping ONNX export for %s", onnx_path.name)
        return False

    n_features = len(feature_names)
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        with open(onnx_path, "wb") as fh:
            fh.write(onnx_model.SerializeToString())
        logger.info("ONNX saved → %s", onnx_path)
        return True
    except Exception as exc:
        logger.warning("ONNX export failed for %s: %s", onnx_path.name, exc)
        return False


def step4_evaluate(reports_dir: Path):
    """Run evaluation and generate reports with correct naming."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
        f1_score, precision_score, recall_score
    )

    reports_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))
    if not pred_files:
        logger.warning("No prediction files found in %s", PRED_DIR)
        return {}

    all_metrics = {}
    results = {}

    for pred_path in pred_files:
        stem = pred_path.stem.replace("_predictions", "")
        parts = stem.split("_", 1)
        placement = parts[0]
        model_name = parts[1] if len(parts) > 1 else "model"
        label = f"{placement} / {model_name}"

        df = pd.read_csv(pred_path)
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        metrics = {
            "accuracy":    float(accuracy_score(y_true, y_pred)),
            "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(specificity),
            "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        }

        # Compute false alarm rate
        t_ms = df["window_start_ms"].to_numpy() if "window_start_ms" in df.columns else np.arange(len(y_pred)) * 1000
        adl_mask = y_true == 0
        adl_starts = t_ms[adl_mask]
        adl_preds = y_pred[adl_mask]
        order = np.argsort(adl_starts)
        adl_starts = adl_starts[order]
        adl_preds = adl_preds[order]
        import math
        fa_events = 0
        last_event_end = -math.inf
        cooldown_ms = 5000
        for t, pred in zip(adl_starts, adl_preds):
            if pred == 1:
                if t > last_event_end + cooldown_ms:
                    fa_events += 1
                    last_event_end = t
                else:
                    last_event_end = max(last_event_end, t + 1000)
        total_adl_ms = len(adl_starts) * 1000
        fa_per_hour = (fa_events / (total_adl_ms / 1000) * 3600) if total_adl_ms > 0 else float("nan")
        metrics["false_alarms_per_hour"] = float(fa_per_hour)

        logger.info("[%s] acc=%.3f rec=%.3f f1=%.3f FA/hr=%.2f",
                    label, metrics["accuracy"], metrics["recall"], metrics["f1"], fa_per_hour)

        all_metrics[label] = metrics
        results[label] = metrics

        # Confusion matrix PNG - with required naming
        if model_name == "random_forest":
            if placement == "phone":
                cm_name = "phone_confusion_matrix.png"
            else:
                cm_name = "wearable_confusion_matrix.png"
        else:
            cm_name = f"confusion_matrix_{stem}.png"

        fig, ax = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix(y_true, y_pred, labels=[0, 1]),
            display_labels=["ADL", "Fall"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(reports_dir / cm_name, dpi=120)
        plt.close(fig)
        logger.info("Confusion matrix saved → %s", reports_dir / cm_name)

        # Also save with generic name for all models
        generic_cm = reports_dir / f"confusion_matrix_{stem}.png"
        if generic_cm != reports_dir / cm_name:
            fig, ax = plt.subplots(figsize=(5, 4))
            disp.plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(label)
            fig.tight_layout()
            # Recompute since plot state is exhausted
            disp2 = ConfusionMatrixDisplay(
                confusion_matrix(y_true, y_pred, labels=[0, 1]),
                display_labels=["ADL", "Fall"],
            )
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            disp2.plot(ax=ax2, colorbar=False, cmap="Blues")
            ax2.set_title(label)
            fig2.tight_layout()
            fig2.savefig(generic_cm, dpi=120)
            plt.close(fig2)

    # Comparison plot
    if all_metrics:
        metric_keys = ["accuracy", "precision", "recall", "specificity", "f1"]
        labels = list(all_metrics.keys())
        n_bars = len(labels)
        x = np.arange(len(metric_keys))
        width = 0.8 / max(n_bars, 1)
        fig, ax = plt.subplots(figsize=(12, 5))
        for i, (lbl, m) in enumerate(all_metrics.items()):
            vals = [m.get(k, 0) for k in metric_keys]
            ax.bar(x + i * width, vals, width, label=lbl)
        ax.set_xticks(x + width * (n_bars - 1) / 2)
        ax.set_xticklabels(metric_keys)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        fig.savefig(reports_dir / "model_comparison.png", dpi=120)
        plt.close(fig)
        logger.info("Comparison plot saved")

    # Save JSON
    out_json = reports_dir / "evaluation_results.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Evaluation results → %s", out_json)

    return results


def verify_outputs():
    """Verify all required output files exist."""
    required = [
        DATA_DIR / "phone_features.csv",
        DATA_DIR / "wrist_features.csv",
        MODELS_DIR / "phone_rf_model.pkl",
        MODELS_DIR / "wrist_rf_model.pkl",
        MODELS_DIR / "phone_model.onnx",
        MODELS_DIR / "wearable_model.onnx",
        REPORTS_DIR / "evaluation_results.json",
        REPORTS_DIR / "phone_confusion_matrix.png",
        REPORTS_DIR / "wearable_confusion_matrix.png",
    ]
    all_ok = True
    for p in required:
        exists = p.exists() and p.stat().st_size > 0
        status = "✓" if exists else "✗ MISSING"
        logger.info("  %s  %s", status, p)
        if not exists:
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DATASET_DIR))
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of files (for testing)")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    t_total = time.time()
    logger.info("AI-03 Fall Detection Full Pipeline")

    dfs = []
    features = {}

    if not args.skip_load:
        dfs = step1_load(Path(args.dataset_dir), max_files=args.max_files)
    else:
        logger.info("STEP 1: Skipped")

    if not args.skip_extract:
        if not dfs:
            logger.error("No data loaded. Cannot extract features.")
            sys.exit(1)
        features = step2_extract(dfs, DATA_DIR)
    else:
        # Load existing feature CSVs
        import pandas as pd
        logger.info("STEP 2: Loading existing feature CSVs")
        for name in ("phone", "wrist"):
            p = DATA_DIR / f"{name}_features.csv"
            if p.exists():
                features[name] = pd.read_csv(p)
                logger.info("Loaded %s (%d rows)", p, len(features[name]))

    if not args.skip_train:
        if not features:
            logger.error("No features available. Cannot train.")
            sys.exit(1)
        all_scores = step3_train(features, MODELS_DIR, seed=SEED)
    else:
        logger.info("STEP 3: Skipped")
        all_scores = {}

    if not args.skip_evaluate:
        results = step4_evaluate(REPORTS_DIR)
    else:
        logger.info("STEP 4: Skipped")
        results = {}

    logger.info("=" * 60)
    logger.info("Pipeline finished in %.1f s", time.time() - t_total)
    logger.info("Verifying outputs...")
    verify_outputs()

    if results:
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY:")
        for label, m in results.items():
            logger.info("  [%s] acc=%.3f prec=%.3f rec=%.3f f1=%.3f FA/hr=%.2f",
                        label, m.get("accuracy", 0), m.get("precision", 0),
                        m.get("recall", 0), m.get("f1", 0), m.get("false_alarms_per_hour", 0))


if __name__ == "__main__":
    main()
