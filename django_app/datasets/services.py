"""
datasets/services.py — File validation and dataset profiling helpers.

Shared by upload view and test suite.
"""

import io
import os
from pathlib import Path

import pandas as pd

from django.conf import settings

# Allowed extensions (lowercase, with dot)
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_UPLOAD_SIZE    = getattr(settings, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024)

# Minimum columns expected in a loan dataset (not enforced strictly — just a hint)
MIN_COLUMNS = 2


class FileValidationError(Exception):
    """Raised when an uploaded file fails validation."""
    pass


def validate_file_extension(filename: str) -> None:
    """Raise FileValidationError if extension is not in ALLOWED_EXTENSIONS."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )


def validate_file_size(file) -> None:
    """Raise FileValidationError if file exceeds MAX_UPLOAD_SIZE."""
    file.seek(0, 2)         # seek to end
    size = file.tell()
    file.seek(0)            # reset
    if size > MAX_UPLOAD_SIZE:
        mb_limit = MAX_UPLOAD_SIZE / (1024 * 1024)
        raise FileValidationError(
            f"File too large ({size / (1024*1024):.1f} MB). Maximum allowed: {mb_limit:.0f} MB."
        )


def validate_mime_type(file, filename: str) -> None:
    """
    MIME-type sniff validation using file magic bytes.
    Falls back to extension check if python-magic is unavailable.
    """
    ext = Path(filename).suffix.lower()
    file.seek(0)
    header = file.read(8)
    file.seek(0)

    MAGIC_CSV  = None          # CSV has no magic bytes; validated by content
    MAGIC_XLSX = b"PK\x03\x04" # ZIP-based format
    MAGIC_XLS  = b"\xd0\xcf\x11\xe0"

    if ext in (".xlsx", ".xls"):
        if not (header[:4] == MAGIC_XLSX or header[:4] == MAGIC_XLS):
            # Try python-magic if available
            try:
                import magic
                mime = magic.from_buffer(header, mime=True)
                if mime not in (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "application/vnd.ms-excel",
                    "application/zip",
                ):
                    raise FileValidationError(
                        f"File content does not match extension '{ext}'."
                    )
            except ImportError:
                pass  # python-magic not available; trust extension
    # CSV: no magic bytes, content will be validated in profile_dataset


def read_dataframe(file, filename: str) -> pd.DataFrame:
    """
    Read uploaded file into a pandas DataFrame.
    Supports CSV and XLSX/XLS.
    """
    file.seek(0)
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".csv":
            df = pd.read_csv(file, encoding="utf-8", low_memory=False)
        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file)
        else:
            raise FileValidationError(f"Cannot read file with extension '{ext}'.")
    except Exception as e:
        raise FileValidationError(f"Could not parse file: {e}")

    if df.empty:
        raise FileValidationError("Uploaded file contains no data.")

    return df


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Generate a summary profile of the uploaded dataset.

    Returns
    -------
    dict with:
      total_records, total_columns, column_names, dtypes,
      missing_counts, missing_pct, duplicate_count,
      first_rows (list of dicts, first 10), target_present
    """
    missing_counts = df.isnull().sum().to_dict()
    missing_pct    = {k: round(v / len(df) * 100, 1) for k, v in missing_counts.items()}
    dtypes         = df.dtypes.astype(str).to_dict()

    return {
        "total_records":  len(df),
        "total_columns":  len(df.columns),
        "column_names":   list(df.columns),
        "dtypes":         dtypes,
        "missing_counts": {k: int(v) for k, v in missing_counts.items()},
        "missing_pct":    missing_pct,
        "duplicate_count": int(df.duplicated().sum()),
        "first_rows":     df.head(10).fillna("").to_dict(orient="records"),
        "target_present": "Loan_Status" in df.columns,
        "numeric_summary": df.describe().round(2).to_dict(),
    }


def validate_and_profile(file, filename: str) -> dict:
    """
    Run all validations and return a profile.
    Raises FileValidationError on any failure.
    """
    validate_file_extension(filename)
    validate_file_size(file)
    validate_mime_type(file, filename)
    df = read_dataframe(file, filename)
    return profile_dataset(df)
