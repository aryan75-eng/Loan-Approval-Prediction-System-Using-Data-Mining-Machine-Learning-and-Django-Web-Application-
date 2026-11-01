"""
prediction.py — CLI / functional predictor for a single input instance.

Loads best_model.joblib, orders input features correctly, and returns
label + probability. Validated by tests/test_ml_pipeline.py.

Usage (CLI):
  python src/prediction.py

Usage (import):
  from src.prediction import predict_single
  label, proba = predict_single({...})
"""

import json

import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    FEATURE_COLUMNS_PATH,
    INV_LABEL_MAP,
    NUMERIC_COLS,
)
from src.utils import get_logger

logger = get_logger(__name__)


def load_model():
    """Load the serialised pipeline from disk. Raises FileNotFoundError if missing."""
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {BEST_MODEL_PATH}. "
            "Run 'python src/model_training.py' first."
        )
    pipeline = joblib.load(BEST_MODEL_PATH)
    logger.debug(f"Loaded model from {BEST_MODEL_PATH}")
    return pipeline


def load_feature_columns() -> list:
    """Load expected feature order from models/feature_columns.json."""
    if not FEATURE_COLUMNS_PATH.exists():
        # Fallback to config definition
        return FEATURE_COLS
    with open(FEATURE_COLUMNS_PATH) as f:
        data = json.load(f)
    return data["feature_columns"]


def predict_single(input_dict: dict) -> tuple:
    """
    Predict loan approval for a single applicant.

    Parameters
    ----------
    input_dict : dict
        Keys must match FEATURE_COLS. Missing keys will be filled with np.nan
        (the pipeline's imputer handles them).

    Returns
    -------
    (label, prediction_probability, approved_probability) : (str, float, float)
      label                  : "Approved" or "Rejected"
      prediction_probability : probability of the predicted class
      approved_probability   : probability of approval (class 1)
    """
    pipeline = load_model()
    feature_cols = load_feature_columns()

    # Build a single-row DataFrame in the expected column order
    row = {col: input_dict.get(col, np.nan) for col in feature_cols}
    X = pd.DataFrame([row])

    # Type coercions — numeric cols must be numeric
    for col in NUMERIC_COLS:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    # Predict
    y_pred = pipeline.predict(X)[0]                         # 0 or 1
    probas = pipeline.predict_proba(X)[0]                   # [p_reject, p_approve]

    label                  = INV_LABEL_MAP.get(int(y_pred), str(y_pred))
    approved_probability   = float(probas[1]) if len(probas) > 1 else float(probas[0])
    prediction_probability = float(probas[int(y_pred)])     # proba of predicted class

    logger.info(
        f"Prediction: {label} | P(predicted)={prediction_probability:.4f} "
        f"| P(approve)={approved_probability:.4f}"
    )

    return label, prediction_probability, approved_probability


def validate_input(input_dict: dict) -> list:
    """
    Basic input validation. Returns a list of error strings (empty = OK).
    The Django form handles most validation; this is a defence-in-depth check.
    """
    errors = []
    feature_cols = load_feature_columns()

    for col in feature_cols:
        if col not in input_dict:
            errors.append(f"Missing field: {col}")

    for col in NUMERIC_COLS:
        val = input_dict.get(col)
        if val is not None:
            try:
                fval = float(val)
                if fval < 0:
                    errors.append(f"{col} must be non-negative, got {fval}")
            except (ValueError, TypeError):
                errors.append(f"{col} must be a number, got {val!r}")

    return errors


# ── CLI Demo ───────────────────────────────────────────────────────────────────

SAMPLE_INPUT = {
    "ApplicantIncome":   5000,
    "CoapplicantIncome": 1500,
    "LoanAmount":        120,
    "Loan_Amount_Term":  360,
    "Credit_History":    1.0,
    "Gender":            "Male",
    "Married":           "Yes",
    "Dependents":        "0",
    "Education":         "Graduate",
    "Self_Employed":     "No",
    "Property_Area":     "Urban",
}


if __name__ == "__main__":
    print("\n--- Loan Approval Prediction (Demo) ---")
    errors = validate_input(SAMPLE_INPUT)
    if errors:
        print("Input errors:", errors)
    else:
        label, pred_proba, app_proba = predict_single(SAMPLE_INPUT)
        print(f"Result       : {label}")
        print(f"P(predicted) : {pred_proba:.4f}")
        print(f"P(approved)  : {app_proba:.4f}")
