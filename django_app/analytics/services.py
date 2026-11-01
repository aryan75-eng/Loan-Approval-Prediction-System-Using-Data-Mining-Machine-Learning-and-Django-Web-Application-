"""
analytics/services.py — Compute chart data from an uploaded dataset.

Returns JSON-serialisable dicts consumed by Chart.js in the browser.
No retraining happens here — purely descriptive analytics.
"""

import json

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types during json.dumps."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj) -> str:
    return json.dumps(obj, cls=NumpyEncoder)


def loan_status_chart(df: pd.DataFrame) -> dict:
    """Pie/bar chart data for Loan_Status distribution."""
    if TARGET_COL not in df.columns:
        return {}
    counts = df[TARGET_COL].value_counts()
    return {
        "labels": list(counts.index),
        "data":   [int(v) for v in counts.values],
        "colors": ["#43A047", "#E53935"],
    }


def numeric_distributions(df: pd.DataFrame) -> list:
    """Histogram bin data for each numeric column."""
    result = []
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        counts, bin_edges = np.histogram(series, bins=20)
        result.append({
            "label":  col,
            "labels": [f"{e:.0f}" for e in bin_edges[:-1]],
            "data":   counts.tolist(),
        })
    return result


def categorical_distributions(df: pd.DataFrame) -> list:
    """Bar chart data for each categorical column."""
    result = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        vc = df[col].value_counts()
        result.append({
            "label":  col,
            "labels": list(vc.index),
            "data":   [int(v) for v in vc.values],
        })
    return result


def missing_values_chart(df: pd.DataFrame) -> dict:
    """Bar chart of missing value counts per column."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return {
        "labels": list(missing.index),
        "data":   [int(v) for v in missing.values],
    }


def credit_history_by_status(df: pd.DataFrame) -> dict:
    """Stacked bar: Credit_History vs Loan_Status."""
    if TARGET_COL not in df.columns or "Credit_History" not in df.columns:
        return {}
    ct = pd.crosstab(df["Credit_History"], df[TARGET_COL])
    return {
        "labels":   [str(x) for x in ct.index.tolist()],
        "approved": ct.get("Y", ct.get(1, pd.Series([]))).tolist(),
        "rejected": ct.get("N", ct.get(0, pd.Series([]))).tolist(),
    }


def income_vs_loan_scatter(df: pd.DataFrame) -> dict:
    """Scatter plot data: ApplicantIncome vs LoanAmount coloured by status."""
    cols_needed = ["ApplicantIncome", "LoanAmount", TARGET_COL]
    if not all(c in df.columns for c in cols_needed):
        return {}

    approved = df[df[TARGET_COL].isin(["Y", 1])]
    rejected = df[df[TARGET_COL].isin(["N", 0])]

    def to_points(subset):
        return [{"x": float(r["ApplicantIncome"]), "y": float(r["LoanAmount"])}
                for _, r in subset[["ApplicantIncome", "LoanAmount"]].dropna().iterrows()]

    return {
        "approved": to_points(approved),
        "rejected": to_points(rejected),
    }


def preprocessing_summary(df: pd.DataFrame) -> dict:
    """Before-processing summary shown on the preprocessing analytics page."""
    total       = len(df)
    missing_map = df.isnull().sum().to_dict()
    total_miss  = sum(missing_map.values())
    dup_count   = int(df.duplicated().sum())

    numeric_stats = {}
    for col in NUMERIC_COLS:
        if col in df.columns:
            s = df[col].dropna()
            numeric_stats[col] = {
                "mean":   round(float(s.mean()), 2),
                "median": round(float(s.median()), 2),
                "std":    round(float(s.std()), 2),
                "min":    round(float(s.min()), 2),
                "max":    round(float(s.max()), 2),
            }

    return {
        "total_rows":      total,
        "missing_total":   total_miss,
        "missing_per_col": {k: int(v) for k, v in missing_map.items() if v > 0},
        "duplicate_count": dup_count,
        "numeric_stats":   numeric_stats,
    }


def get_all_chart_data(df: pd.DataFrame) -> dict:
    """Convenience wrapper — returns all chart datasets as a single dict."""
    return {
        "loan_status":      loan_status_chart(df),
        "numeric_dists":    numeric_distributions(df),
        "cat_dists":        categorical_distributions(df),
        "missing":          missing_values_chart(df),
        "credit_by_status": credit_history_by_status(df),
        "scatter":          income_vs_loan_scatter(df),
        "preprocessing":    preprocessing_summary(df),
    }
