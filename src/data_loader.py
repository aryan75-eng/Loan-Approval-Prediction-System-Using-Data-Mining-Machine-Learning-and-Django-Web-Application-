"""
data_loader.py — Load the raw dataset, validate schema, and identify target.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    ID_COL,
    RAW_DATA_PATH,
    REQUIRED_COLUMNS,
    TARGET_COL,
)
from src.utils import get_logger

logger = get_logger(__name__)


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load loan dataset from CSV or XLSX.

    Parameters
    ----------
    filepath : Path
        Path to the raw data file. Defaults to data/raw/loan_train.csv.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with all original columns.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing or dataset is empty.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            "Download loan_train.csv from Kaggle and place it at data/raw/."
        )

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    logger.info(f"Loaded {filepath.name}: {df.shape[0]} rows × {df.shape[1]} columns")

    _validate_schema(df)

    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Assert all required columns are present."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    logger.info(f"Schema validated. Target column: '{TARGET_COL}'")
    logger.debug(f"Columns: {list(df.columns)}")


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a lightweight summary dict — used by Django analytics views.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict with keys: shape, missing_counts, duplicate_count, dtypes, target_distribution
    """
    return {
        "shape": df.shape,
        "missing_counts": df.isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_distribution": (
            df[TARGET_COL].value_counts().to_dict() if TARGET_COL in df.columns else {}
        ),
    }
