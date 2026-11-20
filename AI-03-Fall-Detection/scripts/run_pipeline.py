"""
scripts/run_pipeline.py
=======================
End-to-end orchestration script for the AI-03 Fall Detection pipeline.

Steps
-----
1. Load all UMAFall CSV files from DATASET_DIR.
2. Extract phone and wrist features → save to FEATURES_DIR.
3. Train RandomForest + MLP for phone and wrist → save to MODELS_DIR.
4. Generate evaluation reports → save to REPORTS_DIR.

Configuration is read from environment variables (see .env.example) with
sensible defaults.  You can also pass arguments on the command line.

Usage
-----
    # Using .env file:
    python scripts/run_pipeline.py

    # Explicit paths:
    python scripts/run_pipeline.py \\
        --dataset-dir  d:/Farhan/Assignment/Dataset \\
        --features-dir d:/Farhan/Assignment/AI-03-Fall-Detection/data/features \\
        --models-dir   d:/Farhan/Assignment/AI-03-Fall-Detection/models \\
        --reports-dir  d:/Farhan/Assignment/AI-03-Fall-Detection/reports \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR      = PROJECT_ROOT / "src"

# Add both PROJECT_ROOT and SRC_DIR to sys.path so both styles of import work:
# - `from src.data_loader import ...`
# - `from data_loader import ...`
for p in [str(SRC_DIR), str(PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Top-level imports with dual-fallback support
try:
    from src.data_loader import load_all_files
    from src.feature_extraction import (
        build_feature_dataset,
        extract_phone_features,
        extract_wrist_features,
    )
    from src.train import run_training
    from src.evaluate import run_evaluation
except ImportError:
    print("Import error")

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass   # python-dotenv not installed; env vars must be set externally

# ── Defaults from env ─────────────────────────────────────────────────────────
DEFAULT_DATASET_DIR  = os.getenv("DATASET_DIR",  str(Path(r"d:\Farhan\Assignment\Dataset") if Path(r"d:\Farhan\Assignment\Dataset").exists() else PROJECT_ROOT / "data" / "raw"))
DEFAULT_FEATURES_DIR = os.getenv("FEATURES_DIR", str(PROJECT_ROOT / "data"))
DEFAULT_MODELS_DIR   = os.getenv("MODELS_DIR",   str(PROJECT_ROOT / "models"))
DEFAULT_REPORTS_DIR  = os.getenv("REPORTS_DIR",  str(PROJECT_ROOT / "reports"))
DEFAULT_SEED         = int(os.getenv("RANDOM_SEED", "42"))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Pipeline steps ────────────────────────────────────────────────────────────

def step_load(dataset_dir: str) -> list:
    """Step 1 — Load all UMAFall CSV files."""
    logger.info("=" * 60)
    logger.info("STEP 1: Loading dataset from %s", dataset_dir)
    t0 = time.time()
    dfs = load_all_files(dataset_dir, verbose=True)
    logger.info("Loaded %d files in %.1f s", len(dfs), time.time() - t0)
    return dfs


def step_extract(dfs: list, features_dir: str) -> None:
    """Step 2 — Extract phone + wrist features."""
    import pandas as pd

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STEP 2: Extracting features → %s", features_dir)

    for name, extractor in [("phone", extract_phone_features), ("wrist", extract_wrist_features)]:
        t0 = time.time()
        csv_path = features_dir / f"{name}_features.csv"
        df = build_feature_dataset(dfs, extractor, output_csv=str(csv_path))
        logger.info(
            "[%s] %d feature rows extracted in %.1f s → %s",
            name, len(df), time.time() - t0, csv_path,
        )


def step_train(features_dir: str, models_dir: str, seed: int) -> None:
    """Step 3 — Train RandomForest + MLP for phone and wrist."""
    logger.info("=" * 60)
    logger.info("STEP 3: Training models (seed=%d)", seed)
    t0 = time.time()
    run_training(features_dir, models_dir, seed=seed)
    logger.info("Training complete in %.1f s", time.time() - t0)


def step_evaluate(predictions_dir: str, reports_dir: str) -> None:
    """Step 4 — Evaluate models and generate reports."""
    logger.info("=" * 60)
    logger.info("STEP 4: Evaluating models → %s", reports_dir)
    t0 = time.time()
    results = run_evaluation(predictions_dir, reports_dir)
    logger.info("Evaluation complete in %.1f s", time.time() - t0)
    for label, metrics in results.items():
        logger.info("  [%s] F1=%.3f  Recall=%.3f  FA/hr=%.2f",
                    label, metrics.get("f1", 0), metrics.get("recall", 0),
                    metrics.get("false_alarms_per_hour", float("nan")))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="AI-03 Fall Detection Pipeline")
    parser.add_argument("--dataset-dir",    default=DEFAULT_DATASET_DIR)
    parser.add_argument("--features-dir",   default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--models-dir",     default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir",    default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--predictions-dir",default=None,
                        help="Dir with *_predictions.csv; defaults to features-dir/predictions or data/predictions")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-load",     action="store_true")
    parser.add_argument("--skip-extract",  action="store_true")
    parser.add_argument("--skip-train",    action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    # Determine predictions directory dynamically
    if args.predictions_dir:
        predictions_dir = args.predictions_dir
    elif (Path(args.features_dir) / "predictions").exists():
        predictions_dir = str(Path(args.features_dir) / "predictions")
    elif (PROJECT_ROOT / "data" / "predictions").exists():
        predictions_dir = str(PROJECT_ROOT / "data" / "predictions")
    else:
        predictions_dir = str(Path(args.features_dir) / "predictions")

    t_total = time.time()
    logger.info("AI-03 Fall Detection Pipeline — starting")

    dfs = []

    if not args.skip_load:
        dfs = step_load(args.dataset_dir)
    else:
        logger.info("STEP 1: Skipped (--skip-load)")

    if not args.skip_extract:
        if not dfs:
            logger.warning("No DataFrames loaded; skipping feature extraction.")
        else:
            step_extract(dfs, args.features_dir)
    else:
        logger.info("STEP 2: Skipped (--skip-extract)")

    if not args.skip_train:
        step_train(args.features_dir, args.models_dir, seed=args.seed)
    else:
        logger.info("STEP 3: Skipped (--skip-train)")

    if not args.skip_evaluate:
        step_evaluate(predictions_dir, args.reports_dir)
    else:
        logger.info("STEP 4: Skipped (--skip-evaluate)")

    logger.info("=" * 60)
    logger.info("Pipeline finished in %.1f s", time.time() - t_total)


if __name__ == "__main__":
    main()
