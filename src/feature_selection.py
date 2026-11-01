"""
feature_selection.py — Correlation matrix, near-zero-variance filter,
and Random Forest feature importance ranking.

Outputs:
  - visualizations/correlation_heatmap.png
  - visualizations/feature_importance.png
  - models/feature_columns.json  (written by model_training.py after selection)
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    FEATURE_COLUMNS_PATH,
    MODELS_DIR,
    NUMERIC_COLS,
    RANDOM_SEED,
    TARGET_COL,
    VIZ_DIR,
)
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


# ── Correlation Analysis ──────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Compute and plot a Pearson correlation heatmap for numeric columns + target.
    Saves to visualizations/correlation_heatmap.png.
    """
    ensure_dir(VIZ_DIR)

    numeric_df = df[NUMERIC_COLS + [TARGET_COL]].copy()

    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, pad=12)
    plt.tight_layout()

    out_path = VIZ_DIR / "correlation_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation heatmap: {out_path}")


# ── Near-Zero Variance Filter ────────────────────────────────────────────────

def remove_low_variance_features(
    df: pd.DataFrame, threshold: float = 0.01
) -> list:
    """
    Return list of numeric features whose variance exceeds threshold.
    Logs any dropped features. Categorical features are kept as-is.
    """
    selected = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        var = df[col].var()
        if var > threshold:
            selected.append(col)
        else:
            logger.warning(f"Low-variance feature '{col}' (var={var:.5f}) — consider dropping.")

    return selected + CATEGORICAL_COLS


# ── Random Forest Importance ─────────────────────────────────────────────────

def compute_rf_feature_importance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.Series:
    """
    Fit a quick Random Forest on encoded/scaled data and return feature importances.

    Parameters
    ----------
    X_train : pd.DataFrame  Preprocessed training features (after ColumnTransformer)
    y_train : pd.Series

    Returns
    -------
    pd.Series sorted descending, index = feature names
    """
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    importances = pd.Series(rf.feature_importances_, index=X_train.columns if hasattr(X_train, "columns") else None)
    importances.sort_values(ascending=False, inplace=True)
    logger.info(f"Top 5 features by RF importance:\n{importances.head()}")
    return importances


def plot_feature_importance(importances: pd.Series) -> None:
    """
    Bar chart of RF feature importances.
    Saves to visualizations/feature_importance.png.
    """
    ensure_dir(VIZ_DIR)

    top_n = min(20, len(importances))
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    top.sort_values().plot(kind="barh", ax=ax, color="#1976D2")
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title("Feature Importance (Random Forest)", fontsize=14)
    plt.tight_layout()

    out_path = VIZ_DIR / "feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved feature importance plot: {out_path}")


# ── Save Feature Columns ──────────────────────────────────────────────────────

def save_feature_columns(feature_cols: list = FEATURE_COLS) -> None:
    """
    Persist the ordered feature list to models/feature_columns.json.
    This file is loaded by ml_service.py to order prediction inputs correctly.
    """
    ensure_dir(MODELS_DIR)
    payload = {
        "feature_columns": feature_cols,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "target_col": TARGET_COL,
    }
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved feature columns: {FEATURE_COLUMNS_PATH}")
