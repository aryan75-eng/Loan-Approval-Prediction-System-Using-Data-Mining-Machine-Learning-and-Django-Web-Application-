"""
config.py — Single source of truth for paths, constants, and feature lists.
All other src/ modules import from here.
"""

import os
from pathlib import Path

# ── Root Directories ────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # loan_prediction_system/

DATA_DIR        = ROOT_DIR / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
MODELS_DIR      = ROOT_DIR / "models"
VIZ_DIR         = ROOT_DIR / "visualizations"
REPORTS_DIR     = ROOT_DIR / "reports"

# ── File Paths ───────────────────────────────────────────────────────────────
RAW_DATA_PATH          = RAW_DATA_DIR / "loan_train.csv"
BEST_MODEL_PATH        = MODELS_DIR / "best_model.joblib"
FEATURE_COLUMNS_PATH   = MODELS_DIR / "feature_columns.json"
MODEL_METADATA_PATH    = MODELS_DIR / "model_metadata.json"
MODEL_COMPARISON_PATH  = REPORTS_DIR / "model_comparison.csv"
TRAINING_LOG_PATH      = REPORTS_DIR / "training_log.txt"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Dataset Schema ───────────────────────────────────────────────────────────
TARGET_COL = "Loan_Status"
ID_COL     = "Loan_ID"

# Numeric features fed to the ML pipeline
NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

# Categorical features fed to the ML pipeline
CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

# All feature columns (numeric first, then categorical) — defines input order
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# Columns that must be present in the raw CSV
REQUIRED_COLUMNS = [ID_COL] + FEATURE_COLS + [TARGET_COL]

# ── Outlier Capping ──────────────────────────────────────────────────────────
# IQR multiplier for Winsorization (cap, not drop)
IQR_MULTIPLIER = 1.5
OUTLIER_COLS   = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]

# ── Train/Test Split ─────────────────────────────────────────────────────────
TEST_SIZE = 0.20

# ── Cross-Validation ─────────────────────────────────────────────────────────
CV_FOLDS = 5

# ── Model Names ──────────────────────────────────────────────────────────────
MODEL_LOGISTIC    = "LogisticRegression"
MODEL_DECISION    = "DecisionTree"
MODEL_RANDOM      = "RandomForest"
ALL_MODEL_NAMES   = [MODEL_LOGISTIC, MODEL_DECISION, MODEL_RANDOM]

# ── Label Mapping ─────────────────────────────────────────────────────────────
# Y → 1 (Approved), N → 0 (Rejected)
LABEL_MAP     = {"Y": 1, "N": 0}
INV_LABEL_MAP = {1: "Approved", 0: "Rejected"}

# ── Upload Constraints (mirrored in Django settings) ─────────────────────────
MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx"}
