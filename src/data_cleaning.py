"""
data_cleaning.py — Missing-value imputation, duplicate removal, dtype fixes.

Strategy:
  - Drop Loan_ID (identifier, no predictive value)
  - Numeric columns: impute with median
  - Categorical columns: impute with mode (most_frequent)
  - Drop exact duplicate rows
  - Encode target: Y→1, N→0
"""

import pandas as pd

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    ID_COL,
    LABEL_MAP,
    NUMERIC_COLS,
    TARGET_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame  Raw dataframe from data_loader.

    Returns
    -------
    pd.DataFrame  Cleaned dataframe with feature columns + encoded target.
    """
    df = df.copy()
    original_shape = df.shape

    # 1. Drop identifier column
    if ID_COL in df.columns:
        df.drop(columns=[ID_COL], inplace=True)
        logger.info(f"Dropped identifier column '{ID_COL}'")

    # 2. Encode target variable: Y→1, N→0
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map(LABEL_MAP)
        invalid_targets = df[TARGET_COL].isnull().sum()
        if invalid_targets > 0:
            logger.warning(
                f"{invalid_targets} rows have unrecognised target values — dropping."
            )
            df.dropna(subset=[TARGET_COL], inplace=True)
        df[TARGET_COL] = df[TARGET_COL].astype(int)
        logger.info(f"Target encoded: Y→1, N→0. Distribution:\n{df[TARGET_COL].value_counts().to_dict()}")

    # 3. Impute numeric columns with median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            missing_count = df[col].isnull().sum()
            df[col].fillna(median_val, inplace=True)
            logger.debug(f"Numeric impute '{col}': {missing_count} values → median {median_val:.2f}")

    # 4. Impute categorical columns with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            missing_count = df[col].isnull().sum()
            df[col].fillna(mode_val, inplace=True)
            logger.debug(f"Categorical impute '{col}': {missing_count} values → mode '{mode_val}'")

    # 5. Remove duplicate rows
    before_dedup = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = before_dedup - len(df)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")

    # 6. Final dtype cleanup — ensure numeric columns are float
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        f"Cleaning complete: {original_shape} → {df.shape} "
        f"(removed {original_shape[0] - df.shape[0]} rows)"
    )

    # Log remaining missing values
    remaining_missing = df[FEATURE_COLS].isnull().sum().sum()
    if remaining_missing > 0:
        logger.warning(f"Still {remaining_missing} missing values after imputation — check data.")
    else:
        logger.info("No missing values remain after cleaning.")

    return df


def get_cleaning_report(df_before: pd.DataFrame, df_after: pd.DataFrame) -> dict:
    """
    Generate a before/after cleaning summary for display in Django analytics.

    Returns
    -------
    dict
    """
    return {
        "rows_before": df_before.shape[0],
        "rows_after": df_after.shape[0],
        "rows_removed": df_before.shape[0] - df_after.shape[0],
        "missing_before": int(df_before.isnull().sum().sum()),
        "missing_after": int(df_after[FEATURE_COLS].isnull().sum().sum()),
        "duplicates_removed": int(df_before.duplicated().sum()),
    }
