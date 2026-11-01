"""
model_training.py — Orchestrates the full offline ML pipeline.

Run this script once to produce:
  models/best_model.joblib
  models/feature_columns.json
  models/model_metadata.json
  visualizations/*.png
  reports/model_comparison.csv
  reports/classification_report_<model>.txt
  reports/training_log.txt

Usage:
  python src/model_training.py
"""

import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_COLS,
    CV_FOLDS,
    FEATURE_COLS,
    FEATURE_COLUMNS_PATH,
    MODEL_DECISION,
    MODEL_LOGISTIC,
    MODEL_METADATA_PATH,
    MODEL_RANDOM,
    MODELS_DIR,
    NUMERIC_COLS,
    PROCESSED_DIR,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
)
from src.data_cleaning import clean_data, get_cleaning_report
from src.data_loader import load_raw_data
from src.eda import run_eda
from src.feature_selection import (
    plot_correlation_heatmap,
    plot_feature_importance,
    save_feature_columns,
)
from src.model_evaluation import (
    evaluate_model,
    plot_confusion_matrix,
    save_classification_report,
    save_comparison_csv,
)
from src.outlier_detection import cap_outliers, plot_boxplots
from src.preprocessing import build_preprocessor
from src.utils import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)


def main():
    set_seed(RANDOM_SEED)
    logger.info("=" * 60)
    logger.info("LOAN APPROVAL PREDICTION — OFFLINE TRAINING PIPELINE")
    logger.info(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    df_raw = load_raw_data()

    # ── 2. Clean data ─────────────────────────────────────────────────────────
    df_clean = clean_data(df_raw)
    cleaning_report = get_cleaning_report(df_raw, df_clean)
    logger.info(f"Cleaning: {cleaning_report}")

    # ── 3. EDA (on cleaned data) ───────────────────────────────────────────────
    run_eda(df_clean)

    # ── 4. Cap outliers ───────────────────────────────────────────────────────
    df_before_cap = df_clean.copy()
    df_capped, bounds_report = cap_outliers(df_clean)
    plot_boxplots(df_before_cap, df_capped)

    # ── 5. Train/test split (stratified) ──────────────────────────────────────
    X = df_capped[FEATURE_COLS]
    y = df_capped[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )
    logger.info(
        f"Split: train={len(X_train)}, test={len(X_test)}, "
        f"train_pos_rate={y_train.mean():.3f}"
    )

    # ── 6. Save processed splits ──────────────────────────────────────────────
    ensure_dir(PROCESSED_DIR)
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    # ── 7. Build preprocessor ─────────────────────────────────────────────────
    preprocessor = build_preprocessor()

    # ── 8. Correlation heatmap (numeric) ──────────────────────────────────────
    plot_correlation_heatmap(df_capped)

    # ── 9. Define classifiers ─────────────────────────────────────────────────
    classifiers = {
        MODEL_LOGISTIC: LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        ),
        MODEL_DECISION: DecisionTreeClassifier(
            max_depth=6,
            random_state=RANDOM_SEED,
        ),
        MODEL_RANDOM: RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    # ── 10. Train, cross-validate, and evaluate each model ────────────────────
    all_metrics = []
    fitted_pipelines = {}
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for name, clf in classifiers.items():
        logger.info(f"\n{'─'*40}")
        logger.info(f"Training: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ]
        )

        # Cross-validation on training set (F1 scoring)
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        logger.info(f"CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Fit final model on full training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Evaluate on test set
        metrics = evaluate_model(name, pipeline, X_test, y_test, cv_scores)
        all_metrics.append(metrics)

        # Save individual reports
        save_classification_report(name, y_test, metrics["y_pred"])
        plot_confusion_matrix(name, y_test, metrics["y_pred"])

    # ── 11. Select best model by F1 (tie-break: recall) ──────────────────────
    best_metrics = max(all_metrics, key=lambda m: (m["f1"], m["recall"]))
    best_name    = best_metrics["name"]
    best_pipeline = fitted_pipelines[best_name]

    logger.info(f"\n{'='*40}")
    logger.info(f"BEST MODEL: {best_name}")
    logger.info(f"  F1        = {best_metrics['f1']:.4f}")
    logger.info(f"  Recall    = {best_metrics['recall']:.4f}")
    logger.info(f"  Accuracy  = {best_metrics['accuracy']:.4f}")
    logger.info(f"Selection rationale: Highest test F1; tie-break by recall.")

    # ── 12. Feature importance (using RF — or best model if RF) ───────────────
    rf_pipeline = fitted_pipelines.get(MODEL_RANDOM)
    if rf_pipeline is not None:
        # Get preprocessed feature names
        prep = rf_pipeline.named_steps["preprocessor"]
        try:
            feature_names = prep.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(
                rf_pipeline.named_steps["classifier"].n_features_in_
            )]

        importances = pd.Series(
            rf_pipeline.named_steps["classifier"].feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)
        plot_feature_importance(importances)

    # ── 13. Save artifacts ────────────────────────────────────────────────────
    ensure_dir(MODELS_DIR)
    joblib.dump(best_pipeline, BEST_MODEL_PATH)
    logger.info(f"Saved best model: {BEST_MODEL_PATH}")

    save_feature_columns(FEATURE_COLS)

    # Model metadata
    metadata = {
        "model_name":    best_name,
        "training_date": datetime.now().isoformat(),
        "random_seed":   RANDOM_SEED,
        "test_size":     TEST_SIZE,
        "cv_folds":      CV_FOLDS,
        "metrics": {
            "accuracy":  best_metrics["accuracy"],
            "precision": best_metrics["precision"],
            "recall":    best_metrics["recall"],
            "f1":        best_metrics["f1"],
            "cv_mean":   best_metrics["cv_mean"],
            "cv_std":    best_metrics["cv_std"],
        },
        "all_models": [
            {k: v for k, v in m.items() if k != "y_pred"}
            for m in all_metrics
        ],
    }
    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved model metadata: {MODEL_METADATA_PATH}")

    # ── 14. Save model comparison CSV ─────────────────────────────────────────
    save_comparison_csv(all_metrics)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
