"""
eda.py — Generate all EDA visualizations to visualizations/.

Plots produced (10 total):
  loan_status_distribution.png
  numerical_distributions.png
  categorical_counts.png
  boxplots_outliers.png          (produced by outlier_detection.py)
  correlation_heatmap.png        (produced by feature_selection.py)
  applicant_income_analysis.png
  loan_amount_analysis.png
  credit_history_analysis.png
  scatter_income_vs_loan.png
  feature_importance.png         (produced by feature_selection.py after training)
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL, VIZ_DIR
from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# Global style
sns.set_theme(style="whitegrid", palette="Blues_d")
BLUE = "#1976D2"
ACCENT = "#0D47A1"
APPROVED_COLOR = "#43A047"
REJECTED_COLOR = "#E53935"


def run_eda(df: pd.DataFrame) -> None:
    """
    Execute all EDA plots in sequence.
    df should be the *cleaned* dataframe (target already encoded as 0/1).
    """
    ensure_dir(VIZ_DIR)
    logger.info("Starting EDA visualisation generation...")

    plot_loan_status_distribution(df)
    plot_numerical_distributions(df)
    plot_categorical_counts(df)
    plot_applicant_income_analysis(df)
    plot_loan_amount_analysis(df)
    plot_credit_history_analysis(df)
    plot_scatter_income_vs_loan(df)

    logger.info("EDA visualisations complete.")


# ── 1. Loan Status Distribution ───────────────────────────────────────────────

def plot_loan_status_distribution(df: pd.DataFrame) -> None:
    labels = ["Approved (Y)", "Rejected (N)"]
    counts = [int((df[TARGET_COL] == 1).sum()), int((df[TARGET_COL] == 0).sum())]
    colors = [APPROVED_COLOR, REJECTED_COLOR]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    ax1.bar(labels, counts, color=colors, edgecolor="white", width=0.5)
    ax1.set_title("Loan Status — Count", fontsize=13)
    ax1.set_ylabel("Number of Applications")
    for i, v in enumerate(counts):
        ax1.text(i, v + 3, str(v), ha="center", fontweight="bold")

    # Pie chart
    ax2.pie(counts, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, wedgeprops={"edgecolor": "white"})
    ax2.set_title("Loan Status — Proportion", fontsize=13)

    fig.suptitle("Loan Approval Status Distribution", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save("loan_status_distribution.png")


# ── 2. Numerical Distributions ────────────────────────────────────────────────

def plot_numerical_distributions(df: pd.DataFrame) -> None:
    n = len(NUMERIC_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, color=BLUE, edgecolor="white", alpha=0.85)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # Hide spare subplot
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numerical Feature Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save("numerical_distributions.png")


# ── 3. Categorical Counts ─────────────────────────────────────────────────────

def plot_categorical_counts(df: pd.DataFrame) -> None:
    n = len(CATEGORICAL_COLS)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(CATEGORICAL_COLS):
        ax = axes[i]
        value_counts = df[col].value_counts()
        ax.bar(value_counts.index, value_counts.values, color=BLUE, edgecolor="white")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=20)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Categorical Feature Distributions", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save("categorical_counts.png")


# ── 4. Applicant Income Analysis ──────────────────────────────────────────────

def plot_applicant_income_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram + KDE by approval status
    for label, color, name in [(1, APPROVED_COLOR, "Approved"), (0, REJECTED_COLOR, "Rejected")]:
        subset = df[df[TARGET_COL] == label]["ApplicantIncome"].dropna()
        axes[0].hist(subset, bins=30, alpha=0.6, color=color, label=name, edgecolor="white")

    axes[0].set_title("Applicant Income by Loan Status", fontsize=12)
    axes[0].set_xlabel("Applicant Income")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Box plot by status
    approved = df[df[TARGET_COL] == 1]["ApplicantIncome"].dropna()
    rejected = df[df[TARGET_COL] == 0]["ApplicantIncome"].dropna()
    axes[1].boxplot([approved, rejected], labels=["Approved", "Rejected"],
                    patch_artist=True,
                    boxprops=dict(facecolor=BLUE, alpha=0.6))
    axes[1].set_title("Income Distribution by Status (Box)", fontsize=12)
    axes[1].set_ylabel("Applicant Income")

    fig.suptitle("Applicant Income Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("applicant_income_analysis.png")


# ── 5. Loan Amount Analysis ───────────────────────────────────────────────────

def plot_loan_amount_analysis(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["LoanAmount"].dropna(), bins=35, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(df["LoanAmount"].median(), color=ACCENT, linestyle="--", label=f"Median: {df['LoanAmount'].median():.0f}")
    ax.axvline(df["LoanAmount"].mean(), color=REJECTED_COLOR, linestyle="--", label=f"Mean: {df['LoanAmount'].mean():.0f}")
    ax.set_title("Loan Amount Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Loan Amount (thousands)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    _save("loan_amount_analysis.png")


# ── 6. Credit History Analysis ────────────────────────────────────────────────

def plot_credit_history_analysis(df: pd.DataFrame) -> None:
    # Cross-tab: Credit_History vs Loan_Status
    ct = pd.crosstab(df["Credit_History"], df[TARGET_COL])
    ct.columns = ["Rejected", "Approved"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ct.plot(kind="bar", ax=ax1, color=[REJECTED_COLOR, APPROVED_COLOR],
            edgecolor="white", rot=0)
    ax1.set_title("Credit History vs Loan Status (Count)", fontsize=12)
    ax1.set_xlabel("Credit History (1=meets guidelines)")
    ax1.set_ylabel("Count")
    ax1.legend(title="Status")

    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot(kind="bar", ax=ax2, color=[REJECTED_COLOR, APPROVED_COLOR],
                edgecolor="white", rot=0)
    ax2.set_title("Credit History vs Loan Status (%)", fontsize=12)
    ax2.set_xlabel("Credit History")
    ax2.set_ylabel("Percentage (%)")
    ax2.legend(title="Status")

    fig.suptitle("Credit History Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("credit_history_analysis.png")


# ── 7. Scatter — Income vs Loan Amount ───────────────────────────────────────

def plot_scatter_income_vs_loan(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, color, name in [(1, APPROVED_COLOR, "Approved"), (0, REJECTED_COLOR, "Rejected")]:
        subset = df[df[TARGET_COL] == label]
        ax.scatter(
            subset["ApplicantIncome"],
            subset["LoanAmount"],
            alpha=0.45,
            color=color,
            label=name,
            s=30,
            edgecolors="none",
        )

    ax.set_title("Applicant Income vs Loan Amount", fontsize=14, fontweight="bold")
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount (thousands)")
    ax.legend(title="Loan Status")
    plt.tight_layout()
    _save("scatter_income_vs_loan.png")


# ── Helper ─────────────────────────────────────────────────────────────────────

def _save(filename: str) -> None:
    path = VIZ_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {path}")
