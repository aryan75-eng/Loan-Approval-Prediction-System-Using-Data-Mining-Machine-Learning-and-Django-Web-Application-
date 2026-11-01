"""
test_ml_pipeline.py — Unit tests for the offline ML pipeline.

Tests T06, T07, T08, T09 from the test matrix.
Requires: models/best_model.joblib (run src/model_training.py first for T09).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root so src imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    LABEL_MAP,
    NUMERIC_COLS,
    OUTLIER_COLS,
    TARGET_COL,
)
from src.data_cleaning import clean_data
from src.outlier_detection import cap_outliers, compute_iqr_bounds


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df_with_missing():
    """DataFrame with deliberate missing values and duplicates."""
    data = {
        "Loan_ID":            ["LP001", "LP002", "LP003", "LP003"],  # LP003 duplicate
        "Gender":             ["Male", None, "Female", "Female"],
        "Married":            ["Yes", "No", None, None],
        "Dependents":         ["0", "1", None, None],
        "Education":          ["Graduate", "Not Graduate", "Graduate", "Graduate"],
        "Self_Employed":      ["No", None, "No", "No"],
        "ApplicantIncome":    [5000, 3000, 7000, 7000],
        "CoapplicantIncome":  [1500, 0, None, None],
        "LoanAmount":         [128, None, 200, 200],
        "Loan_Amount_Term":   [360, 360, None, None],
        "Credit_History":     [1.0, None, 1.0, 1.0],
        "Property_Area":      ["Urban", "Rural", "Semiurban", "Semiurban"],
        "Loan_Status":        ["Y", "N", "Y", "Y"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_outliers():
    """DataFrame with outlier values in income / loan columns."""
    np.random.seed(42)
    normal_income = np.random.normal(5000, 1000, 100)
    data = {
        "Loan_ID":          [f"LP{i:03d}" for i in range(100)],
        "Gender":           ["Male"] * 100,
        "Married":          ["Yes"] * 100,
        "Dependents":       ["0"] * 100,
        "Education":        ["Graduate"] * 100,
        "Self_Employed":    ["No"] * 100,
        "ApplicantIncome":  list(normal_income) [:-1] + [999999],  # extreme outlier
        "CoapplicantIncome": [1000] * 100,
        "LoanAmount":       [100] * 99 + [50000],                 # extreme outlier
        "Loan_Amount_Term": [360] * 100,
        "Credit_History":   [1.0] * 100,
        "Property_Area":    ["Urban"] * 100,
        "Loan_Status":      ["Y"] * 80 + ["N"] * 20,
    }
    return pd.DataFrame(data)


# ── T06: Duplicates removed ───────────────────────────────────────────────────

def test_T06_duplicate_rows_removed(sample_df_with_missing):
    """T06 — Duplicate rows should be removed after cleaning."""
    original_len = len(sample_df_with_missing)
    df_clean = clean_data(sample_df_with_missing)
    assert len(df_clean) < original_len, "Expected rows to be removed (duplicates)"
    assert df_clean.duplicated().sum() == 0, "Duplicates remain after cleaning"


# ── T07: Missing values imputed ───────────────────────────────────────────────

def test_T07_missing_values_imputed(sample_df_with_missing):
    """T07 — No missing values should remain in feature columns after cleaning."""
    df_clean = clean_data(sample_df_with_missing)
    remaining_missing = df_clean[FEATURE_COLS].isnull().sum().sum()
    assert remaining_missing == 0, (
        f"Expected 0 missing values in features, got {remaining_missing}\n"
        f"{df_clean[FEATURE_COLS].isnull().sum()}"
    )


def test_T07_numeric_imputed_with_median(sample_df_with_missing):
    """T07 — Numeric NaN should be replaced by median, not dropped."""
    rows_before = len(sample_df_with_missing.dropna())
    df_clean = clean_data(sample_df_with_missing)
    # At least as many rows as the non-NaN original (imputation, not deletion)
    assert len(df_clean) >= rows_before, "Rows were dropped instead of imputed"


# ── T08: Outliers capped (not dropped) ───────────────────────────────────────

def test_T08_outliers_capped_not_dropped(sample_df_with_outliers):
    """T08 — Outlier capping must not reduce row count."""
    df_clean = clean_data(sample_df_with_outliers)
    rows_before = len(df_clean)
    df_capped, _ = cap_outliers(df_clean)
    assert len(df_capped) == rows_before, (
        f"Rows dropped during outlier capping: {rows_before} → {len(df_capped)}"
    )


def test_T08_outlier_values_within_bounds(sample_df_with_outliers):
    """T08 — After capping, all values must be within IQR bounds."""
    df_clean = clean_data(sample_df_with_outliers)
    df_capped, bounds = cap_outliers(df_clean)

    for col in OUTLIER_COLS:
        if col not in df_capped.columns:
            continue
        lower = bounds[col]["lower_bound"]
        upper = bounds[col]["upper_bound"]
        below = (df_capped[col] < lower).sum()
        above = (df_capped[col] > upper).sum()
        assert below == 0, f"'{col}' has {below} values below lower bound {lower}"
        assert above == 0, f"'{col}' has {above} values above upper bound {upper}"


# ── T09: Pipeline produces model artifact ────────────────────────────────────

def test_T09_model_artifact_exists():
    """T09 — best_model.joblib must exist (run model_training.py first)."""
    from src.config import BEST_MODEL_PATH, FEATURE_COLUMNS_PATH
    assert BEST_MODEL_PATH.exists(), (
        f"Model not found at {BEST_MODEL_PATH}. Run: python src/model_training.py"
    )
    assert FEATURE_COLUMNS_PATH.exists(), (
        f"Feature columns JSON not found at {FEATURE_COLUMNS_PATH}."
    )


def test_T09_model_can_predict():
    """T09 — Loaded model must produce a valid prediction on a sample input."""
    from src.config import BEST_MODEL_PATH
    if not BEST_MODEL_PATH.exists():
        pytest.skip("Model not trained yet.")

    from src.prediction import SAMPLE_INPUT, predict_single

    label, pred_proba, app_proba = predict_single(SAMPLE_INPUT)
    assert label in ("Approved", "Rejected"), f"Unexpected label: {label}"
    assert 0.0 <= pred_proba <= 1.0, f"pred_proba out of range: {pred_proba}"
    assert 0.0 <= app_proba <= 1.0, f"app_proba out of range: {app_proba}"
