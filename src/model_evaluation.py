"""
model_evaluation.py — Compute metrics and persist reports.

Functions:
  evaluate_model()    → dict of metrics for a single model
  save_model_reports() → write classification_report_<name>.txt
  save_comparison_csv() → write reports/model_comparison.csv
  plot_confusion_matrix() → visualizations/confusion_matrix_<name>.png
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import REPORTS_DIR, VIZ_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def evaluate_model(
    name: str,
    model,
    X_test,
    y_test,
    cv_scores: np.ndarray = None,
) -> dict:
    """
    Evaluate a fitted pipeline on the test set.

    Parameters
    ----------
    name       : Model identifier string
    model      : Fitted sklearn Pipeline
    X_test     : Test features (raw — pipeline handles preprocessing)
    y_test     : Test labels (0/1)
    cv_scores  : Array of CV fold scores (optional)

    Returns
    -------
    dict with keys: name, accuracy, precision, recall, f1,
                    cv_mean, cv_std, y_pred
    """
    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    cv_mean   = float(np.mean(cv_scores)) if cv_scores is not None else 0.0
    cv_std    = float(np.std(cv_scores))  if cv_scores is not None else 0.0

    metrics = {
        "name":      name,
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "cv_mean":   round(cv_mean, 4),
        "cv_std":    round(cv_std, 4),
        "y_pred":    y_pred,
    }

    logger.info(
        f"[{name}] Acc={accuracy:.4f}  P={precision:.4f}  "
        f"R={recall:.4f}  F1={f1:.4f}  CV={cv_mean:.4f}±{cv_std:.4f}"
    )

    return metrics


def save_classification_report(name: str, y_test, y_pred) -> None:
    """Write sklearn classification_report to reports/classification_report_<name>.txt."""
    ensure_dir(REPORTS_DIR)
    report = classification_report(y_test, y_pred, target_names=["Rejected", "Approved"])
    out_path = REPORTS_DIR / f"classification_report_{name}.txt"
    with open(out_path, "w") as f:
        f.write(f"Classification Report — {name}\n")
        f.write("=" * 50 + "\n")
        f.write(report)
    logger.info(f"Saved classification report: {out_path}")


def plot_confusion_matrix(name: str, y_test, y_pred) -> None:
    """Save confusion matrix heatmap to visualizations/confusion_matrix_<name>.png."""
    ensure_dir(VIZ_DIR)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Rejected", "Approved"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {name}", fontsize=13)
    plt.tight_layout()

    out_path = VIZ_DIR / f"confusion_matrix_{name}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix: {out_path}")


def save_comparison_csv(all_metrics: list) -> None:
    """
    Write reports/model_comparison.csv from a list of metric dicts.
    This file is imported by the Django management command load_model_metrics.
    """
    ensure_dir(REPORTS_DIR)
    fieldnames = ["name", "accuracy", "precision", "recall", "f1", "cv_mean", "cv_std", "is_selected"]
    rows = []
    # Determine best model (highest F1, tie-break recall)
    best = max(all_metrics, key=lambda m: (m["f1"], m["recall"]))

    for m in all_metrics:
        rows.append({
            "name":       m["name"],
            "accuracy":   m["accuracy"],
            "precision":  m["precision"],
            "recall":     m["recall"],
            "f1":         m["f1"],
            "cv_mean":    m["cv_mean"],
            "cv_std":     m["cv_std"],
            "is_selected": "True" if m["name"] == best["name"] else "False",
        })

    out_path = REPORTS_DIR / "model_comparison.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved model comparison CSV: {out_path}")
    logger.info(f"Best model: {best['name']} (F1={best['f1']:.4f})")
