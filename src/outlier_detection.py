"""
outlier_detection.py — IQR-based outlier detection and Winsorization (capping).

Policy: Cap outliers at IQR bounds — do NOT drop rows.
Only applied to OUTLIER_COLS: ApplicantIncome, CoapplicantIncome, LoanAmount.
"""

import pandas as pd

from src.config import IQR_MULTIPLIER, OUTLIER_COLS, VIZ_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)


def compute_iqr_bounds(series: pd.Series, multiplier: float = IQR_MULTIPLIER) -> tuple:
    """
    Compute lower and upper Winsorization bounds using IQR.

    Returns
    -------
    (lower_bound, upper_bound) : tuple of float
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return lower, upper


def cap_outliers(df: pd.DataFrame) -> tuple:
    """
    Cap outliers in OUTLIER_COLS to IQR-derived bounds.

    Parameters
    ----------
    df : pd.DataFrame  Cleaned dataframe (post data_cleaning.py)

    Returns
    -------
    (df_capped, bounds_report) : (pd.DataFrame, dict)
      bounds_report — per-column dict of {lower, upper, n_lower_capped, n_upper_capped}
    """
    df = df.copy()
    bounds_report = {}

    for col in OUTLIER_COLS:
        if col not in df.columns:
            logger.warning(f"Outlier column '{col}' not found — skipping.")
            continue

        lower, upper = compute_iqr_bounds(df[col])
        n_lower = int((df[col] < lower).sum())
        n_upper = int((df[col] > upper).sum())

        df[col] = df[col].clip(lower=lower, upper=upper)

        bounds_report[col] = {
            "lower_bound": round(lower, 2),
            "upper_bound": round(upper, 2),
            "n_lower_capped": n_lower,
            "n_upper_capped": n_upper,
        }

        logger.info(
            f"Outlier cap '{col}': lower={lower:.2f}, upper={upper:.2f} "
            f"| capped {n_lower} low + {n_upper} high values"
        )

    return df, bounds_report


def plot_boxplots(df_before: pd.DataFrame, df_after: pd.DataFrame) -> None:
    """
    Save side-by-side boxplots (before/after capping) to visualizations/.

    Parameters
    ----------
    df_before : pd.DataFrame  Pre-capping dataframe
    df_after  : pd.DataFrame  Post-capping dataframe
    """
    import matplotlib.pyplot as plt

    ensure_dir(VIZ_DIR)
    fig, axes = plt.subplots(
        nrows=len(OUTLIER_COLS),
        ncols=2,
        figsize=(12, 4 * len(OUTLIER_COLS)),
    )

    for i, col in enumerate(OUTLIER_COLS):
        if col not in df_before.columns:
            continue
        # Before
        axes[i, 0].boxplot(df_before[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#90CAF9"))
        axes[i, 0].set_title(f"{col} — Before Capping", fontsize=11)
        axes[i, 0].set_ylabel(col)

        # After
        axes[i, 1].boxplot(df_after[col].dropna(), vert=True, patch_artist=True,
                           boxprops=dict(facecolor="#A5D6A7"))
        axes[i, 1].set_title(f"{col} — After Capping", fontsize=11)
        axes[i, 1].set_ylabel(col)

    fig.suptitle("Outlier Detection — IQR Capping (Before vs After)", fontsize=14, y=1.01)
    plt.tight_layout()

    out_path = VIZ_DIR / "boxplots_outliers.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved boxplot: {out_path}")
