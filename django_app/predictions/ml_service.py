"""
predictions/ml_service.py — Singleton model loader for Django.

Loads best_model.joblib once at first call (not at import time, so Django
startup stays fast even before training). Thread-safe for development server.
Exposes predict() and predict_proba() for use by views.
"""

import json
import threading

import joblib
import numpy as np
import pandas as pd

from django.conf import settings

_lock      = threading.Lock()
_pipeline  = None
_feat_cols = None


def _load_artifacts():
    """Load pipeline and feature columns from disk (called once, thread-safe)."""
    global _pipeline, _feat_cols

    models_dir = settings.ML_MODELS_DIR
    model_path  = models_dir / "best_model.joblib"
    feat_path   = models_dir / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run 'python src/model_training.py' from the project root first."
        )

    _pipeline = joblib.load(model_path)

    if feat_path.exists():
        with open(feat_path) as f:
            data = json.load(f)
        _feat_cols = data["feature_columns"]
    else:
        # Fallback: import from src config
        from src.config import FEATURE_COLS
        _feat_cols = FEATURE_COLS


def _get_pipeline():
    """Return loaded pipeline, loading if necessary."""
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:   # double-checked locking
                _load_artifacts()
    return _pipeline


def _get_feature_cols():
    global _feat_cols
    if _feat_cols is None:
        _get_pipeline()             # triggers _load_artifacts
    return _feat_cols


def _build_dataframe(input_dict: dict) -> pd.DataFrame:
    """
    Build a single-row DataFrame in the expected feature column order.
    Missing keys → np.nan (pipeline imputer handles them).
    """
    from src.config import NUMERIC_COLS

    feat_cols = _get_feature_cols()
    row = {col: input_dict.get(col, np.nan) for col in feat_cols}
    X = pd.DataFrame([row])

    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    return X


def predict(input_dict: dict) -> tuple:
    """
    Predict loan approval for a single applicant dict.

    Returns
    -------
    (label, prediction_probability, approved_probability)
      label                  : "Approved" or "Rejected"
      prediction_probability : probability of predicted class (float)
      approved_probability   : probability of Approved / class-1 (float)

    Raises
    ------
    FileNotFoundError if model not trained yet.
    """
    from src.config import INV_LABEL_MAP

    pipeline = _get_pipeline()
    X = _build_dataframe(input_dict)

    y_pred = int(pipeline.predict(X)[0])
    probas = pipeline.predict_proba(X)[0]   # [p_rejected, p_approved]

    label                  = INV_LABEL_MAP.get(y_pred, str(y_pred))
    approved_probability   = float(probas[1]) if len(probas) > 1 else float(probas[0])
    prediction_probability = float(probas[y_pred])

    return label, prediction_probability, approved_probability


def is_model_loaded() -> bool:
    """Check whether the model artifact exists without loading it."""
    models_dir = settings.ML_MODELS_DIR
    return (models_dir / "best_model.joblib").exists()


def get_model_info() -> dict:
    """Return metadata from model_metadata.json or an empty dict."""
    path = settings.ML_MODELS_DIR / "model_metadata.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}
